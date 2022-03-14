# Copyright 2021 DeepMind Technologies Limited.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Units-based components, acting on lists of 1d vectors."""

import enum
import math
from typing import Mapping, Optional, Sequence, Tuple

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import util
from alphastar.architectures.components.static_data import camera_masks
from alphastar.architectures.components.static_data import unit_encoder_data
from alphastar.commons import sample
import chex
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from pysc2.env.converter.cc.game_data.python import uint8_lookup
from pysc2.lib.features import FeatureUnit

from s2clientprotocol import raw_pb2 as sc_raw


class UnitTagsMasking(enum.Enum):
  NONE = "none"
  NON_EMPTY = "non_empty"
  TARGETABLE = "targetable"
  TARGETABLE_WITH_CAMERA = "targetable_with_camera"
  SELECTABLE = "selectable"


def get_unit_tags_mask(raw_units: chex.Array,
                       mode: UnitTagsMasking) -> chex.Array:
  """Get a mask over unit tags.

  Args:
    raw_units: The raw_units observation from the environment, of shape
      [..., num_raw_unit_features] and dtype int32.
    mode: The masking mode, a UnitTagsMasking.

  Returns:
    A boolean mask of size [...] specifying which units are masked given the
      mode.
  """
  if mode is UnitTagsMasking.NONE:
    return jnp.ones(shape=raw_units.shape[0], dtype=jnp.bool_)
  if mode is UnitTagsMasking.NON_EMPTY:
    return jnp.not_equal(
        raw_units[..., int(FeatureUnit.alliance)], 0)
  elif mode is UnitTagsMasking.TARGETABLE:
    return raw_units[..., int(FeatureUnit.tag)] > 0
  elif mode is UnitTagsMasking.SELECTABLE:
    my_units = jnp.equal(
        raw_units[..., int(FeatureUnit.alliance)], int(sc_raw.Alliance.Self))
    # In theory, units in cargo could be selected, but currently it is not
    # supported by the transforms and actions.
    not_in_cargo = jnp.equal(
        raw_units[..., int(FeatureUnit.is_in_cargo)], 0)
    return jnp.logical_and(
        get_unit_tags_mask(raw_units, UnitTagsMasking.TARGETABLE),
        jnp.logical_and(my_units, not_in_cargo))
  else:
    raise ValueError(f"Unsupported mode: {mode}.")


def get_unit_tags_camera_mask(raw_units: chex.Array,
                              function_arg: chex.Array) -> chex.Array:
  """Get which units can be targeted considering the camera.

  For some function arguments of the action, we are not allowed to target
  enemy units outside the camera. This function returns a mask over the units
  which specifies which units can be targeted given the function argument.

  Args:
    raw_units: The raw_units observation from the environment,
      of shape [max_num_observed_units, num_raw_unit_features] and dtype int32.
    function_arg: The `function` argument of the current action, of shape []
      and dtype int32.

  Returns:
    A boolean mask of size [max_num_observed_units] specifying which units are
      masked.
  """
  chex.assert_rank(raw_units, 2)
  chex.assert_type(raw_units, jnp.int32)
  chex.assert_rank(function_arg, 0)
  chex.assert_type(function_arg, jnp.int32)
  # 1) Is the unit on the camera?
  is_on_camera = raw_units[:, int(FeatureUnit.is_on_screen)]
  mask = jnp.equal(is_on_camera, 1)
  # 2) Is the unit owned by us?. We can target our own units outside the camera
  unit_owner = raw_units[:, int(FeatureUnit.alliance)]
  my_units = jnp.equal(unit_owner, int(sc_raw.Alliance.Self))
  mask = jnp.logical_or(mask, my_units)
  # 3) Can this function argument target units outside the camera?
  all_camera_only_functions = camera_masks.get_on_camera_only_functions_unit()
  all_not_camera_only_functions = jnp.asarray(
      np.logical_not(all_camera_only_functions))
  is_not_camera_only = all_not_camera_only_functions[function_arg]
  is_not_camera_only = jnp.broadcast_to(
      is_not_camera_only[jnp.newaxis], [raw_units.shape[0]])
  mask = jnp.logical_or(mask, is_not_camera_only)
  # 4) Is it targetable? (this should remove effects)
  targetable_units = get_unit_tags_mask(raw_units, UnitTagsMasking.TARGETABLE)
  mask = jnp.logical_and(mask, targetable_units)
  return mask


_UnitEncoderOneHot = Tuple[chex.Array, int]


class UnitsEncoder(modular.BatchedComponent):
  """Encode the units input."""

  def __init__(
      self,
      output_name: types.StreamType,
      max_num_observed_units: int,
      num_raw_unit_features: int,
      units_stream_size: int,
      action_spec: types.ActionSpec,
      num_unit_types: int,
      num_buff_types: int,
      name: Optional[str] = None
  ):
    """Initializes UnitsEncoder module.

    Args:
      output_name: The name to give to the output of this module, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      num_raw_unit_features: The number of features per unit,
        ie. obs_spec["raw_units"].shape[1].
      units_stream_size: The size of the output encoding each unit.
      action_spec: The action spec.
      num_unit_types: The number of different units.
      num_buff_types: The number of different buffs.
      name: The name of this component.
    """
    super().__init__(name)
    self._output_name = output_name
    self._units_stream_size = units_stream_size
    self._max_num_observed_units = max_num_observed_units
    self._num_raw_unit_features = num_raw_unit_features
    self._world_dim = int(math.sqrt(action_spec["world"].maximum + 1))
    self._action_spec = action_spec
    self._num_unit_types = num_unit_types
    self._num_buff_types = num_buff_types

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        ("observation", "raw_units"): specs.Array(
            (self._max_num_observed_units, self._num_raw_unit_features),
            jnp.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32),
        "non_empty_units": specs.Array(
            (self._max_num_observed_units,), jnp.bool_)})

  def _encode_unit(self, raw_unit: chex.Array) -> chex.Array:
    chex.assert_rank(raw_unit, 1)
    chex.assert_type(raw_unit, jnp.int32)

    # Lookup tables:
    attributes_lookup = unit_encoder_data.get_attribute_lookup(
        self._num_unit_types)
    function_list = util.get_function_list(self._action_spec)
    function_names = [f.name for f in function_list]
    order_id_lookup = unit_encoder_data.get_order_id_lookup(function_names)
    build_queue_order_id_lookup = (
        unit_encoder_data.get_build_queue_order_id_lookup(function_names))
    addon_lookup = unit_encoder_data.get_addon_lookup(self._num_unit_types)

    # Embeddings (to feed to hk.Linear modules):
    embeddings = [
        jax.nn.one_hot(raw_unit[FeatureUnit.unit_type], self._num_unit_types),
        jnp.asarray(attributes_lookup)[raw_unit[FeatureUnit.unit_type]],
        _binary_scale_embedding(raw_unit[FeatureUnit.x], self._world_dim),
        _binary_scale_embedding(raw_unit[FeatureUnit.y], self._world_dim),
        _features_embedding(raw_unit, {
            FeatureUnit.build_progress: 1. / 100,
            FeatureUnit.health_ratio: 1. / 255,
            FeatureUnit.shield_ratio: 1. / 255,
            FeatureUnit.energy_ratio: 1. / 255,
            FeatureUnit.order_progress_0: 1. / 100,
            FeatureUnit.order_progress_1: 1. / 100}),
        _remap_and_one_hot_embedding(
            raw_unit[FeatureUnit.order_id_0], order_id_lookup),
        _remap_and_one_hot_embedding(
            raw_unit[FeatureUnit.order_id_1], build_queue_order_id_lookup),
        _remap_and_one_hot_embedding(
            raw_unit[FeatureUnit.order_id_2], build_queue_order_id_lookup),
        _remap_and_one_hot_embedding(
            raw_unit[FeatureUnit.order_id_3], build_queue_order_id_lookup),
        _remap_and_one_hot_embedding(
            raw_unit[FeatureUnit.order_id_3], build_queue_order_id_lookup),
    ]

    # Encoded one-hots (to pass to jax.nn.one_hot then hk.Linear):
    one_hot_encoded = [
        _encode_one_hot(raw_unit, FeatureUnit.alliance),
        _encode_sqrt_one_hot(raw_unit, FeatureUnit.health),
        _encode_sqrt_one_hot(raw_unit, FeatureUnit.shield),
        _encode_sqrt_one_hot(raw_unit, FeatureUnit.energy),
        _encode_capped_one_hot(raw_unit, FeatureUnit.cargo_space_taken),
        _encode_one_hot(raw_unit, FeatureUnit.display_type),
        _encode_one_hot(raw_unit, FeatureUnit.cloak),
        _encode_one_hot(raw_unit, FeatureUnit.is_powered),
        _encode_divided_one_hot(raw_unit, FeatureUnit.mineral_contents, 100),
        _encode_divided_one_hot(raw_unit, FeatureUnit.vespene_contents, 100),
        _encode_mined_resource_one_hot(
            raw_unit, FeatureUnit.mineral_contents, self._num_unit_types),
        _encode_mined_resource_one_hot(
            raw_unit, FeatureUnit.vespene_contents, self._num_unit_types),
        _encode_capped_one_hot(raw_unit, FeatureUnit.cargo_space_max),
        _encode_capped_one_hot(raw_unit, FeatureUnit.assigned_harvesters),
        _encode_capped_one_hot(raw_unit, FeatureUnit.ideal_harvesters),
        _encode_capped_one_hot(raw_unit, FeatureUnit.weapon_cooldown),
        _encode_capped_one_hot(raw_unit, FeatureUnit.order_length),
        _encode_lookup(raw_unit[FeatureUnit.addon_unit_type], addon_lookup),
        _encode_one_hot(raw_unit, FeatureUnit.hallucination),
        # Since there are up to 2 buffs, we want to sum the one-hots. That's
        # done by setting the length of the first one_hot to 0.
        (raw_unit[FeatureUnit.buff_id_0], 0),
        (raw_unit[FeatureUnit.buff_id_1], self._num_buff_types),
        _encode_one_hot(raw_unit, FeatureUnit.active),
        _encode_one_hot(raw_unit, FeatureUnit.is_on_screen),
        _encode_one_hot(raw_unit, FeatureUnit.is_blip),
        _encode_divided_one_hot(raw_unit, FeatureUnit.order_progress_0, 10),
        _encode_divided_one_hot(raw_unit, FeatureUnit.order_progress_1, 10),
        _encode_one_hot(raw_unit, FeatureUnit.is_in_cargo),
        _encode_sqrt_one_hot(raw_unit, FeatureUnit.buff_duration_remain),
        _encode_one_hot(raw_unit, FeatureUnit.attack_upgrade_level),
        _encode_one_hot(raw_unit, FeatureUnit.armor_upgrade_level),
        _encode_one_hot(raw_unit, FeatureUnit.shield_upgrade_level),
        # Previous arguments, they are always the last two entries:
        _encode_one_hot(raw_unit, -2),
        _encode_one_hot(raw_unit, -1),
    ]

    # Put all the encoded one-hots in a single boolean vector:
    sum_offsets = np.cumsum([0] + [offset for _, offset in one_hot_encoded])
    indices = jnp.stack([idx + offset for (idx, _), offset
                         in zip(one_hot_encoded, sum_offsets[:-1])])
    boolean_code = jnp.matmul(
        jnp.ones((len(indices),), jnp.float32),
        indices[:, jnp.newaxis] == jnp.arange(sum_offsets[-1]))
    embeddings.append(util.astype(boolean_code, jnp.float32))

    embedding = sum([hk.Linear(self._units_stream_size)(x) for x in embeddings])
    mask = get_unit_tags_mask(raw_unit, UnitTagsMasking.NON_EMPTY)
    embedding = jnp.where(mask, embedding, 0)
    return embedding, mask

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    raw_units = inputs["observation", "raw_units"]
    embedding, mask = jax.vmap(self._encode_unit)(raw_units)
    outputs = types.StreamDict({
        self._output_name: embedding,
        "non_empty_units": mask})
    return outputs, {}


def _encode_one_hot(raw_unit: chex.Array,
                    feature_idx: int) -> _UnitEncoderOneHot:
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  return raw_unit[feature_idx], unit_encoder_data.MAX_VALUES[feature_idx] + 1


def _encode_capped_one_hot(raw_unit: chex.Array,
                           feature_idx: int) -> _UnitEncoderOneHot:
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  max_value = unit_encoder_data.MAX_VALUES[feature_idx]
  return jnp.minimum(raw_unit[feature_idx], max_value), max_value + 1


def _encode_sqrt_one_hot(raw_unit: chex.Array,
                         feature_idx: int) -> _UnitEncoderOneHot:
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  max_value = unit_encoder_data.MAX_VALUES[feature_idx]
  max_sqrt_value = int(math.floor(math.sqrt(max_value)))
  x = jnp.floor(jnp.sqrt(util.astype(raw_unit[feature_idx], jnp.float32)))
  x = jnp.minimum(util.astype(x, jnp.int32), max_sqrt_value)
  return x, max_sqrt_value + 1


def _encode_divided_one_hot(raw_unit: chex.Array,
                            feature_idx: int,
                            divisor: int) -> _UnitEncoderOneHot:
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  max_value = unit_encoder_data.MAX_VALUES[feature_idx]
  max_divided_value = max_value // divisor
  x = jnp.floor_divide(raw_unit[feature_idx], divisor)
  x = jnp.minimum(x, max_divided_value)
  return x, max_divided_value + 1


def _encode_mined_resource_one_hot(raw_unit: chex.Array,
                                   feature_idx: int,
                                   num_unit_types: int) -> _UnitEncoderOneHot:
  """Encode the amount of mined resource."""
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  unit_type = raw_unit[FeatureUnit.unit_type]
  initial_resource_lookup = np.zeros((num_unit_types,), dtype=np.int32)
  for unit, resource in unit_encoder_data.INITIAL_RESOURCE_CONTENTS.items():
    unit_id = uint8_lookup.PySc2ToUint8(unit)
    initial_resource_lookup[unit_id] = resource
  initial_resource = jnp.asarray(initial_resource_lookup)[unit_type]
  mined_resource = initial_resource - raw_unit[feature_idx]
  max_value = unit_encoder_data.MAX_VALUES[feature_idx]
  max_sqrt_value = int(math.floor(math.sqrt(max_value)))
  x = jnp.floor(jnp.sqrt(util.astype(mined_resource, jnp.float32)))
  x = jnp.clip(util.astype(x, jnp.int32), 0, max_sqrt_value)
  return x, max_sqrt_value + 1


def _encode_lookup(to_encode: chex.Array,
                   lookup_table: np.ndarray) -> chex.Array:
  chex.assert_rank(to_encode, 0)
  chex.assert_type(to_encode, jnp.int32)
  return jnp.asarray(lookup_table)[to_encode], max(lookup_table) + 1


def _features_embedding(raw_unit: chex.Array,
                        rescales: Mapping[int, float]) -> chex.Array:
  """Select features in `rescales`, rescale and concatenate them."""
  chex.assert_rank(raw_unit, 1)
  chex.assert_type(raw_unit, jnp.int32)
  assert rescales
  selected_features = []
  feature_indices = sorted(rescales.keys())
  i_min = 0
  while i_min < len(feature_indices):
    i_max = i_min
    while ((i_max < len(feature_indices) - 1) and
           (feature_indices[i_max + 1] == feature_indices[i_max] + 1)):
      i_max += 1
    consecutive_features = raw_unit[
        feature_indices[i_min]:feature_indices[i_max] + 1]
    consecutive_rescales = jnp.asarray(
        [rescales[feature_indices[i]] for i in range(i_min, i_max + 1)],
        jnp.float32)
    i_min = i_max + 1
    rescaled_features = jnp.multiply(consecutive_features, consecutive_rescales)
    selected_features.append(rescaled_features)
  return util.astype(jnp.concatenate(selected_features, axis=0), jnp.float32)


def _binary_scale_embedding(to_encode: chex.Array,
                            world_dim: int) -> chex.Array:
  """Encode the feature using its binary representation."""
  chex.assert_rank(to_encode, 0)
  chex.assert_type(to_encode, jnp.int32)
  num_bits = (world_dim - 1).bit_length()
  bit_mask = 1 << np.arange(num_bits)
  pos = jnp.broadcast_to(to_encode[jnp.newaxis], num_bits)
  result = jnp.not_equal(jnp.bitwise_and(pos, bit_mask), 0)
  return util.astype(result, jnp.float32)


def _remap_and_one_hot_embedding(to_encode: chex.Array,
                                 lookup_table: np.ndarray) -> chex.Array:
  remapped, num_classes = _encode_lookup(to_encode, lookup_table)
  return jax.nn.one_hot(remapped, num_classes)


class Transformer(modular.BatchedComponent):
  """Apply unit-wise resblocks, and transformer layers, to the units."""

  def __init__(
      self,
      max_num_observed_units: int,
      units_stream_size: int,
      transformer_num_layers: int,
      transformer_num_heads: int,
      transformer_key_size: int,
      transformer_value_size: int,
      resblocks_num_before: int,
      resblocks_num_after: int,
      resblocks_hidden_size: Optional[int] = None,
      use_layer_norm: bool = True,
      input_name: types.StreamType = "units_stream",
      output_name: types.StreamType = "units_stream",
      name: Optional[str] = None):
    """Initializes Transformer module.

    Args:
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      units_stream_size: The size of the output encoding each unit.
      transformer_num_layers: Number of consecutive transformer layers.
      transformer_num_heads: Number of heads in the transformers.
      transformer_key_size: Size of the keys in the transformers.
      transformer_value_size: Per-head output size of the transformers.
      resblocks_num_before: Number of per-unit fully connected resblocks before
        the transformers.
      resblocks_num_after: Number of per-unit fully connected resblocks after
        the transformers.
      resblocks_hidden_size: Number of hidden units in the resblocks.
      use_layer_norm: Whether to use layer normalization.
      input_name: The name of the input to use, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      output_name: The name to give to the output, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._max_num_observed_units = max_num_observed_units
    self._units_stream_size = units_stream_size
    self._transformer_num_layers = transformer_num_layers
    self._transformer_num_heads = transformer_num_heads
    self._transformer_key_size = transformer_key_size
    self._transformer_value_size = transformer_value_size
    self._resblocks_num_before = resblocks_num_before
    self._resblocks_num_after = resblocks_num_after
    self._resblocks_hidden_size = resblocks_hidden_size
    self._use_layer_norm = use_layer_norm
    self._input_name = input_name
    self._output_name = output_name or input_name

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        "non_empty_units": specs.Array(
            (self._max_num_observed_units,), jnp.bool_),
        self._input_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    mask = inputs["non_empty_units"]

    for _ in range(self._resblocks_num_before):
      x = util.UnitsResblock(
          hidden_size=self._resblocks_hidden_size,
          use_layer_norm=self._use_layer_norm)(x)
    for _ in range(self._transformer_num_layers):
      x1 = x
      if self._use_layer_norm:
        x1 = util.units_layer_norm(x1)
      x1 = jax.nn.relu(x1)
      # The logits mask has shape [num_heads, num_units, num_units]:
      logits_mask = mask[jnp.newaxis, jnp.newaxis]
      x1 = hk.MultiHeadAttention(
          num_heads=self._transformer_num_heads,
          key_size=self._transformer_key_size,
          w_init_scale=1.,
          value_size=self._transformer_value_size,
          model_size=self._units_stream_size)(
              query=x1, key=x1, value=x1, mask=logits_mask)
      # Mask here mostly for safety:
      x1 = jnp.where(mask[:, jnp.newaxis], x1, 0)
      x = x + x1
    for _ in range(self._resblocks_num_after):
      x = util.UnitsResblock(
          hidden_size=self._resblocks_hidden_size,
          use_layer_norm=self._use_layer_norm)(x)
    x = jnp.where(mask[:, jnp.newaxis], x, 0)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class MLP(modular.BatchedComponent):
  """Apply unit-wise linear layers to the units."""

  def __init__(
      self,
      max_num_observed_units: int,
      units_stream_size: int,
      layer_sizes: Sequence[int],
      use_layer_norm: bool = True,
      input_name: types.StreamType = "units_stream",
      output_name: types.StreamType = "units_stream",
      name: Optional[str] = None):
    """Initializes MLP module.

    Args:
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      units_stream_size: The size of the input encoding each unit.
      layer_sizes: The size of each output of layer of the MLP.
      use_layer_norm: Whether to use layer normalization.
      input_name: The name of the input to use, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      output_name: The name to give to the output, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._max_num_observed_units = max_num_observed_units
    self._units_stream_size = units_stream_size
    if not layer_sizes:
      raise ValueError("layer_sizes must contain at least one element.")
    self._layer_sizes = layer_sizes
    self._use_layer_norm = use_layer_norm
    self._input_name = input_name
    self._output_name = output_name or input_name

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array(
            (self._max_num_observed_units, self._layer_sizes[-1]), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for size in self._layer_sizes:
      if self._use_layer_norm:
        x = util.units_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Linear(size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class ToVector(modular.BatchedComponent):
  """Per-unit processing then average over the units dimension."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               max_num_observed_units: int,
               units_stream_size: int,
               units_hidden_sizes: Sequence[int],
               vector_stream_size: int,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes ToVector module.

    Args:
      input_name: The name of the input to use, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      output_name: The name to give to the output, of shape
        [vector_stream_size] and dtype float32.
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      units_stream_size: The size of the output encoding each unit.
      units_hidden_sizes: The list of sizes of the hidden layers processing
        each unit independently, before merging the unit representations.
      vector_stream_size: The size of the output (1d vector representation).
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._max_num_observed_units = max_num_observed_units
    self._units_stream_size = units_stream_size
    self._units_hidden_sizes = units_hidden_sizes
    self._vector_stream_size = vector_stream_size
    self._use_layer_norm = use_layer_norm

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._vector_stream_size,), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for size in self._units_hidden_sizes:
      if self._use_layer_norm:
        x = util.units_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Linear(output_size=size)(x)
    x = x.mean(axis=0)
    if self._use_layer_norm:
      x = util.vector_layer_norm(x)
    x = jax.nn.relu(x)
    x = hk.Linear(output_size=self._vector_stream_size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class ToVisualScatter(modular.BatchedComponent):
  """Scatter the units into their positions in the visual stream.

  This means that each element of the units stream will be embedded and placed
  in the visual stream, at the location corresponding to its (x, y) coordinate
  in the world map.
  """

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               max_num_observed_units: int,
               num_raw_unit_features: int,
               units_stream_size: int,
               units_world_dim: int,
               units_hidden_sizes: Sequence[int],
               output_spatial_size: int,
               output_features_size: int,
               kernel_size: int = 3,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes ToVisualScatter module.

    Args:
      input_name: The name of the input to use, of shape
        [max_num_observed_units, units_stream_size] and dtype float32.
      output_name: The name to give to the output, of shape
        [output_spatial_size, output_spatial_size, output_features_size] and
        dtype float32.
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      num_raw_unit_features: The number of features per unit,
        ie. obs_spec["raw_units"].shape[1].
      units_stream_size: The size of the output encoding each unit.
      units_world_dim: The size of the "world" reference frame of the units.
        This corresponds to the range of the x and y fields of the raw_units
        input (and can be different from the size of the visual stream).
      units_hidden_sizes: The list of sizes of the hidden layers processing
        each unit independently, before merging the unit representations.
      output_spatial_size: The spatial size of the output (2d feature maps).
      output_features_size: The number of the feature planes in the output (2d
        feature maps).
      kernel_size: The size of the convolution kernel to use after scattering.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._max_num_observed_units = max_num_observed_units
    self._num_raw_unit_features = num_raw_unit_features
    self._units_stream_size = units_stream_size
    if units_world_dim % output_spatial_size != 0:
      raise ValueError(f"units_world_dim (set to {units_world_dim}) must be a"
                       "multiple of output_spatial_size "
                       f"({output_spatial_size}).")
    self._units_world_dim = units_world_dim
    self._units_hidden_sizes = tuple(units_hidden_sizes)
    self._output_spatial_size = output_spatial_size
    self._output_features_size = output_features_size
    self._kernel_size = kernel_size
    self._use_layer_norm = use_layer_norm

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        ("observation", "raw_units"): specs.Array(
            (self._max_num_observed_units, self._num_raw_unit_features),
            jnp.int32),
        "non_empty_units": specs.Array(
            (self._max_num_observed_units,), jnp.bool_),
        self._input_name: specs.Array(
            (self._max_num_observed_units, self._units_stream_size),
            jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_spatial_size,
                                        self._output_spatial_size,
                                        self._output_features_size),
                                       jnp.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    raw_units = inputs["observation", "raw_units"]
    non_empty_units = inputs["non_empty_units"]
    z = inputs[self._input_name]

    non_empty_units = non_empty_units[:, jnp.newaxis]

    for size in self._units_hidden_sizes:
      if self._use_layer_norm:
        z = util.units_layer_norm(z)
      z = jax.nn.relu(z)
      z = hk.Linear(output_size=size)(z)
    z = jnp.where(non_empty_units, z, 0)

    ratio = self._units_world_dim // self._output_spatial_size
    unit_x = raw_units[:, FeatureUnit.x] // ratio
    unit_y = raw_units[:, FeatureUnit.y] // ratio
    unit_x = jnp.clip(unit_x, 0, self._output_spatial_size - 1)
    unit_y = jnp.clip(unit_y, 0, self._output_spatial_size - 1)

    one_hot_x = jax.nn.one_hot(unit_x, self._output_spatial_size)
    one_hot_y = jax.nn.one_hot(unit_y, self._output_spatial_size)
    z = jnp.einsum("uy,uf->uyf", one_hot_y, z)
    z = jnp.einsum("ux,uyf->yxf", one_hot_x, z)

    if self._use_layer_norm:
      z = util.visual_layer_norm(z)
    z = jax.nn.relu(z)
    z = hk.Conv2D(
        output_channels=self._output_features_size,
        kernel_shape=self._kernel_size)(z)

    outputs = types.StreamDict({self._output_name: z})
    return outputs, {}


class PointerEmbedding(modular.BatchedComponent):
  """Embeds a single int32 into a float32 vector, taking embeddings as inputs.

  This is similar to vector.Embedding, the difference is that the embeddings
  are not learned, they are passed as an input (embeddings).

  Given two inputs, embeddings and inputs, the output is embeddings[index].
  """

  def __init__(self,
               num_embeddings: int,
               embeddings_size: int,
               index_input_name: types.StreamType,
               embeddings_input_name: types.StreamType,
               output_name: types.StreamType,
               name: Optional[str] = None):
    """Initializes PointerEmbedding module.

    Args:
      num_embeddings: The number of embeddings, ie. the shape[0] of the
        embeddings input.
      embeddings_size: The size of each embedding, ie. the shape[1] of the
        embeddings input.
      index_input_name: The name of the index input to use, of shape []
        and dtype int32.
      embeddings_input_name: The name of the embeddings input, of shape
        [num_embeddings, embeddings_size] and dtype float32.
      output_name: The name to give to the output, of shape [embeddings_size]
        and dtype float32.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._num_embeddings = num_embeddings
    self._embeddings_size = embeddings_size
    self._index_input_name = index_input_name
    self._embeddings_input_name = embeddings_input_name
    self._output_name = output_name

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    spec[self._index_input_name] = specs.Array((), jnp.int32)
    spec[self._embeddings_input_name] = specs.Array(
        (self._num_embeddings, self._embeddings_size), jnp.float32)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    spec[self._output_name] = specs.Array((self._embeddings_size,), jnp.float32)
    return spec

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    embeddings = inputs[self._embeddings_input_name]
    index = inputs[self._index_input_name]
    index = jnp.minimum(index, embeddings.shape[0])
    output = types.StreamDict()
    output[self._output_name] = embeddings[index]
    return output, {}


class BinaryVectorPointerEmbedding(modular.BatchedComponent):
  """Embeds a boolean mask into a float32 vector, taking embeddings as inputs.

  This is similar to vector.BinaryVectorEmbedding, the difference is that the
  embeddings are not learned, they are passed as an input (embeddings).

  Given two inputs, embeddings and mask, the output is the sum of the unmasked
  embeddings.
  """

  def __init__(self,
               num_embeddings: int,
               embeddings_size: int,
               mask_input_name: types.StreamType,
               embeddings_input_name: types.StreamType,
               output_name: types.StreamType,
               name: Optional[str] = None):
    """Initializes BinaryVectorPointerEmbedding module.

    Args:
      num_embeddings: The number of embeddings, ie. the shape[0] of the
        embeddings input.
      embeddings_size: The size of each embedding, ie. the shape[1] of the
        embeddings input.
      mask_input_name: The name of the index input to use, of shape
        [num_embeddings] and dtype bool_.
      embeddings_input_name: The name of the embeddings input, of shape
        [num_embeddings, embeddings_size] and dtype float32.
      output_name: The name to give to the output, of shape [embeddings_size]
        and dtype float32.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._num_embeddings = num_embeddings
    self._embeddings_size = embeddings_size
    self._mask_input_name = mask_input_name
    self._embeddings_input_name = embeddings_input_name
    self._output_name = output_name

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    spec[self._mask_input_name] = specs.Array(
        (self._num_embeddings,), jnp.bool_)
    spec[self._embeddings_input_name] = specs.Array(
        (self._num_embeddings, self._embeddings_size), jnp.float32)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    spec[self._output_name] = specs.Array((self._embeddings_size,), jnp.float32)
    return spec

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    embeddings = inputs[self._embeddings_input_name]
    indices = util.astype(inputs[self._mask_input_name], jnp.float32)
    x = jnp.matmul(indices, embeddings)
    output = types.StreamDict()
    output[self._output_name] = x
    return output, {}


class PointerLogits(modular.BatchedComponent):
  """Produce logits using a pointer network.

  This is basically an attention mechanism between keys, coming from the units
  stream, and a single key, coming from the vector stream.
  """

  def __init__(
      self,
      max_num_observed_units: int,
      num_raw_unit_features: int,
      logits_output_name: types.StreamType,
      mask_output_name: types.StreamType,
      query_input_size: int,
      keys_input_size: int,
      unit_tags_masking: UnitTagsMasking,
      query_input_name: types.StreamType = "vector_stream",
      keys_input_name: types.StreamType = "units_stream",
      num_layers_query: int = 2,
      num_layers_keys: int = 2,
      key_size: int = 64,
      use_layer_norm: bool = True,
      name: Optional[str] = None):
    """Initializes PointerLogits module.

    Args:
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      num_raw_unit_features: The number of features per unit,
        ie. obs_spec["raw_units"].shape[1].
      logits_output_name: The name to give to the logits output, of shape
        [max_num_observed_units] and dtype float32.
      mask_output_name: The name to give to the mask output, of shape
        [max_num_observed_units] and dtype bool.
      query_input_size: The size of the input used for the query.
      keys_input_size: The size of the input used for each key.
      unit_tags_masking: The type of masking to use.
      query_input_name: The name of the input to use for the query (a 1d
        vector, typically the vector stream), of shape [query_input_size] and
        dtype float32.
      keys_input_name: The name of the input to use for the keys (a list of 1d
        vectors, typically the units stream), of shape
        [max_num_observed_units, keys_input_size] and dtype float32.
      num_layers_query: The number of layers used to process the query before
        the attention mechanism.
        num_layers_keys: The number of layers used to individually process the
        keys before the attention mechanism.
      key_size: The size of the keys (and query) in the attention mechanism.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._max_num_observed_units = max_num_observed_units
    self._num_raw_unit_features = num_raw_unit_features
    self._logits_output_name = logits_output_name
    self._mask_output_name = mask_output_name
    self._query_input_size = query_input_size
    self._keys_input_size = keys_input_size
    self._unit_tags_masking = unit_tags_masking
    self._query_input_name = query_input_name
    self._keys_input_name = keys_input_name
    self._num_layers_query = num_layers_query
    self._num_layers_keys = num_layers_keys
    self._key_size = key_size
    self._use_layer_norm = use_layer_norm
    if num_layers_query == 0 and key_size != query_input_size:
      raise ValueError("If num_layers_query is set to 0, key_size must be "
                       "equal to query_input_size, but they are set to "
                       f"{key_size} and {query_input_size} respectively.")
    if num_layers_keys == 0 and key_size != keys_input_size:
      raise ValueError("If num_layers_keys is set to 0, key_size must be "
                       "equal to keys_input_size, but they are set to "
                       f"{key_size} and {keys_input_size} respectively.")

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict({
        ("observation", "raw_units"): specs.Array(
            (self._max_num_observed_units, self._num_raw_unit_features),
            jnp.int32),
        self._keys_input_name: specs.Array(
            (self._max_num_observed_units, self._keys_input_size), jnp.float32),
        self._query_input_name: specs.Array(
            (self._query_input_size,), jnp.float32)})
    if self._unit_tags_masking is UnitTagsMasking.TARGETABLE_WITH_CAMERA:
      spec["action", "function"] = specs.Array((), jnp.int32)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._logits_output_name: specs.Array(
            (self._max_num_observed_units,), jnp.float32),
        self._mask_output_name: specs.Array(
            (self._max_num_observed_units,), jnp.bool_)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    # Query.
    query = inputs[self._query_input_name]
    for i in range(self._num_layers_query):
      if self._use_layer_norm:
        query = util.vector_layer_norm(query)
      query = jax.nn.relu(query)
      if i == self._num_layers_query - 1:
        query = hk.Linear(output_size=self._key_size)(query)
      else:
        query = hk.Linear(output_size=query.shape[-1])(query)

    # Keys.
    keys = inputs[self._keys_input_name]
    for i in range(self._num_layers_keys):
      if self._use_layer_norm:
        keys = util.units_layer_norm(keys)
      keys = jax.nn.relu(keys)
      if i == self._num_layers_keys - 1:
        keys = hk.Linear(output_size=self._key_size)(keys)
      else:
        keys = hk.Linear(output_size=keys.shape[-1])(keys)

    # Mask
    if self._unit_tags_masking is UnitTagsMasking.TARGETABLE_WITH_CAMERA:
      mask = get_unit_tags_camera_mask(inputs["observation", "raw_units"],
                                       inputs["action", "function"])
    else:
      mask = get_unit_tags_mask(inputs["observation", "raw_units"],
                                mode=self._unit_tags_masking)

    # Pointer
    logits = jnp.matmul(keys, query)  # ij,j->i
    logits = sample.mask_logits(logits, mask)

    outputs = types.StreamDict({self._logits_output_name: logits,
                                self._mask_output_name: mask})
    return outputs, {}


class FinalizeUnitTagsLogits(modular.BatchedComponent):
  """Compute full mask and add the end of selection bit to logits and mask."""

  def __init__(
      self,
      input_logits_name: types.StreamType,
      input_mask_name: types.StreamType,
      output_logits_name: types.StreamType,
      output_mask_name: types.StreamType,
      vector_input_name: types.StreamType,
      max_num_observed_units: int,
      vector_input_size: int,
      name: Optional[str] = None):
    """Initializes FinalizeUnitTagsLogits module.

    Args:
      input_logits_name: The name of the input to use for the logits, of shape
        [max_num_observed_units] and dtype float32.
      input_mask_name: The name of the input to use for the mask, of shape
        [max_num_observed_units] and dtype bool.
      output_logits_name: The name of the output for the logits, of shape
        [max_num_observed_units + 1] and dtype float32.
      output_mask_name: The name of the output for the mask, of shape
        [max_num_observed_units + 1] and dtype bool.
      vector_input_name: The name of the input to use to compute the EOS
        (End-Of-Selection) logit, of shape [vector_input_size] and dtype
        float32.
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      vector_input_size: The size of the vector stream input.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_logits_name = input_logits_name
    self._input_mask_name = input_mask_name
    self._output_logits_name = output_logits_name
    self._output_mask_name = output_mask_name
    self._vector_input_name = vector_input_name
    self._num_units = max_num_observed_units
    self._vector_input_size = vector_input_size

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_logits_name: specs.Array((self._num_units,), jnp.float32),
        self._input_mask_name: specs.Array((self._num_units,), jnp.bool_),
        self._vector_input_name: specs.Array(
            (self._vector_input_size,), jnp.float32),
        "selected_unit_tags": specs.Array((self._num_units + 1,), jnp.bool_)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_logits_name: specs.Array(
            (self._num_units + 1,), jnp.float32),
        self._output_mask_name: specs.Array((self._num_units + 1,), jnp.bool_)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    # Logits:
    units_logits = inputs[self._input_logits_name]
    eos_logit = hk.Linear(1)(jax.nn.relu(inputs[self._vector_input_name]))
    logits = jnp.concatenate([units_logits, eos_logit], axis=0)

    # Mask:
    units_mask = inputs[self._input_mask_name]
    selected_units = inputs["selected_unit_tags"]
    # We cannot select the same unit twice:
    units_mask = jnp.logical_and(units_mask,
                                 jnp.logical_not(selected_units[:-1]))
    # We must select at least 1 unit, so EOS is masked if no unit is selected:
    eos_mask = jnp.sum(selected_units, dtype=jnp.bool_)
    # Once EOS is selected, we can only select EOS:
    is_eos_selected = selected_units[-1:]
    units_mask = jnp.logical_and(units_mask, jnp.logical_not(is_eos_selected))
    mask = jnp.concatenate([units_mask, eos_mask[jnp.newaxis]], axis=0)

    logits = sample.mask_logits(logits, mask)
    outputs = types.StreamDict({self._output_logits_name: logits,
                                self._output_mask_name: mask})
    return outputs, {}


class UnitTagsHead(modular.BatchedComponent):
  """Unit tag head.

  This applies a `scan` operation to the inner component.
  It also builds a boolean vector `selected_unit_tags`
  For simplicity, we require the inner component to be stateless.

  This module is equivalent to the following pseudo-code:
  ```
  constant_inputs = extract constant_inputs from inputs
  carries = extract carries from inputs
  x = extract per_step_inputs from inputs
  y = []
  selected_units = [0] * (num_units + 1)
  for i in range(max_num_selected_units):
    comp_output = inner_component.unroll((x[i], carries, selected_units))
    carries = extract carries from comp_output
    y[i] = extract per_step_outputs from comp_outputs
    selected_units[y[i][action]] = 1
  return (y, carries, selected_units)```
  """

  def __init__(
      self,
      inner_component: modular.Component,
      constant_inputs: Sequence[types.StreamType],
      carries: Sequence[types.StreamType],
      per_step_inputs: Sequence[types.StreamType],
      per_step_outputs: Sequence[types.StreamType],
      max_num_selected_units: int,
      max_num_observed_units: int,
      action_output: types.StreamType,
      name: Optional[str] = None):
    """Initializes UnitTagsHead module.

    Args:
      inner_component: The component to wrap and unroll `max_num_selected_units`
        times inside this component. It must at least produce action for the
        currently selected unit tag, and in most cases should also produce
        logits and mask.
      constant_inputs: The list of input names to pass to the every unroll
        instance of the `inner_component`. They can have any shape and dtype.
      carries: The list of inputs to pass to the `inner_component` which values
        will be changed by the `inner_component` at each unroll step, before
        being passed to the next step. They are also returned as outputs of this
        module. They can have any shape and dtype.
      per_step_inputs: The list of inputs to pass to the `inner_component`
        per-step. These inputs shape be [max_num_selected_units, ...]. At the
        i-th unroll step, these inputs will be indexed by [i, ...]. They can
        have any dtype.
      per_step_outputs: The list of outputs of the `inner_component` to return
        for every unroll step. Since gathering these outputs can take up much
        memory, they must be specified explicitely. They will have shape
        [max_num_selected_units, ...], where the element indexed by [i, ...]
        comes from the i-th unroll step. The dtype is specified by the
        `inner_component` output.
      max_num_selected_units: The maximum number of selected units, ie. the
        first dimension of the unit_tags logit vector. This corresponds to the
        number of unroll steps performed by this component.
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      action_output: The name of the output of the `inner_component`
        corresponding to the action. It must be of shape [] and dtype int32.
        This is used to produce the `selected_unit_tags` carry (of shape
        [max_num_observed_units] and dtype bool).
      name: The name of this component.
    """
    super().__init__(name=name)
    self._inner_component = inner_component
    self._constant_inputs = constant_inputs
    self._carries = carries
    self._per_step_inputs = per_step_inputs
    self._per_step_outputs = per_step_outputs
    self._max_num_selected_units = max_num_selected_units
    self._max_num_observed_units = max_num_observed_units
    self._action_output = action_output
    # For simplicity, we request inner_component to be stateless:
    if inner_component.prev_state_spec or inner_component.next_state_spec:
      raise ValueError(f"{self.name}: Inner component must be stateless.")
    # Check that the input and output sets do not intersect:
    if set(constant_inputs).intersection(set(carries)):
      raise ValueError("constant_inputs and carries must "
                       "be disjoint sets but both contain "
                       f"{set(constant_inputs).intersection(set(carries))}.")
    if set(constant_inputs).intersection(set(per_step_inputs)):
      raise ValueError(
          "constant_inputs and per_step_inputs must be disjoint sets but both "
          f"contain {set(constant_inputs).intersection(set(per_step_inputs))}.")
    if set(per_step_inputs).intersection(set(carries)):
      raise ValueError("per_step_inputs and carries must "
                       "be disjoint sets but both contain "
                       f"{set(per_step_inputs).intersection(set(carries))}.")
    if set(per_step_outputs).intersection(set(carries)):
      raise ValueError("per_step_outputs and carries must "
                       "be disjoint sets but both contain "
                       f"{set(per_step_outputs).intersection(set(carries))}.")
    # Check that input and output sets are contained in the inner component spec
    input_spec_names = set(inner_component.input_spec.keys())
    output_spec_names = set(inner_component.output_spec.keys())
    if not set(constant_inputs).issubset(input_spec_names):
      raise ValueError(
          "constant_inputs must be a subset of inner_component.input_spec, but "
          f"{set(constant_inputs).difference(input_spec_names)} is not there.")
    if not set(carries).issubset(input_spec_names):
      raise ValueError(
          "carries must be a subset of inner_component.input_spec, but "
          f"{set(carries).difference(input_spec_names)} is not there.")
    if not set(per_step_inputs).issubset(input_spec_names):
      raise ValueError(
          "per_step_inputs must be a subset of inner_component.input_spec, but "
          f"{set(per_step_inputs).difference(input_spec_names)} is not there.")
    if not set(carries).issubset(output_spec_names):
      raise ValueError(
          "carries must be a subset of inner_component.output_spec, but "
          f"{set(carries).difference(output_spec_names)} is not there.")
    if not set(per_step_outputs).issubset(output_spec_names):
      raise ValueError(
          "per_step_outputs must be a subset of inner_component.output_spec, "
          f"but {set(per_step_inputs).difference(input_spec_names)} is not "
          "there.")
    if action_output not in per_step_outputs:
      raise ValueError(f"action_output ({action_output}) must be in "
                       "per_step_outputs.")
    # Check that the inner component does not change the spec of carries:
    for c in carries:
      if inner_component.input_spec[c] != inner_component.output_spec[c]:
        raise ValueError(f"{self.name}: Carry {c} changed spec.")
    # Check that the input and output sets are not larger than the inner
    # component spec:
    all_inputs = set(constant_inputs).union(carries).union(per_step_inputs)
    missing_inputs = all_inputs.difference(
        set(inner_component.input_spec.keys()))
    if missing_inputs:
      raise ValueError(f"{missing_inputs} are specified as inputs, but are not "
                       "in inner_component.input_spec.")
    all_outputs = set(carries).union(per_step_outputs)
    missing_outputs = all_outputs.difference(
        set(inner_component.output_spec.keys()))
    if missing_outputs:
      raise ValueError(f"{missing_outputs} are specified as outputs, but are "
                       "not in inner_component.output_spec.")
    # Check that all the inputs of the inner component are specified in this
    # init function arguments:
    missing_inputs2 = set(inner_component.input_spec.keys()).difference(
        all_inputs.union(set(["selected_unit_tags"])))
    if missing_inputs2:
      raise ValueError(f"{missing_inputs2} are in inner_component.input_spec, "
                       "but are not specified as inputs.")
    # Check that the action output has the right type:
    if inner_component.output_spec[action_output].shape:
      raise ValueError(
          "The output specified as action_output must have shape (), but has "
          f"shape {inner_component.output_spec[action_output].shape}.")
    if inner_component.output_spec[action_output].dtype != jnp.int32:
      raise ValueError(
          "The output specified as action_output must have dtype int32, but "
          f"has dtype {inner_component.output_spec[action_output].dtype}.")

  def _replicate_spec(self, spec: specs.Array) -> specs.Array:
    return spec.replace(shape=(self._max_num_selected_units,) + spec.shape)

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    for name in self._constant_inputs:
      spec[name] = self._inner_component.input_spec[name]
    for name in self._carries:
      spec[name] = self._inner_component.input_spec[name]
    for name in self._per_step_inputs:
      spec[name] = self._replicate_spec(self._inner_component.input_spec[name])
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    for name in self._carries:
      spec[name] = self._inner_component.output_spec[name]
    for name in self._per_step_outputs:
      spec[name] = self._replicate_spec(self._inner_component.output_spec[name])
    spec["selected_unit_tags"] = specs.Array(
        (self._max_num_observed_units,), jnp.bool_)
    return spec

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    # We need inner_component to be a hk.Module to use it with hk.scan:
    inner_module_unroll = hk.to_module(self._inner_component.unroll)(
        name="inner_module")
    constant_inputs = inputs.filter(self._constant_inputs)
    def iterate(carries, loop_inputs):
      comp_inputs = constant_inputs.copy()
      # scan replaces empty inputs with None:
      comp_inputs.update(carries or types.StreamDict())
      comp_inputs.update(loop_inputs or types.StreamDict())
      comp_inputs = jax.tree_map(lambda x: x[jnp.newaxis], comp_inputs)
      comp_outputs, _, _ = inner_module_unroll(comp_inputs, types.StreamDict())
      comp_outputs = jax.tree_map(lambda x: x[0], comp_outputs)
      carries.update(comp_outputs.filter(self._carries))
      loop_outputs = comp_outputs.filter(self._per_step_outputs)
      action = comp_outputs[self._action_output]
      action_one_hot = jax.nn.one_hot(
          action, self._max_num_observed_units + 1, dtype=jnp.bool_)
      carries["selected_unit_tags"] = jnp.logical_or(
          carries["selected_unit_tags"], action_one_hot)
      return carries, loop_outputs
    carries = inputs.filter(self._carries)
    carries["selected_unit_tags"] = jnp.zeros(
        (self._max_num_observed_units + 1,), jnp.bool_)
    loop_inputs = inputs.filter(self._per_step_inputs)
    carries, loop_outputs = hk.scan(
        iterate, carries, loop_inputs, length=self._max_num_selected_units)
    outputs = loop_outputs
    outputs.update(carries)
    outputs["selected_unit_tags"] = outputs["selected_unit_tags"][:-1]
    # Logs are ignored for simplicity
    return outputs, {}

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

"""Head (logits and sampling) blocks."""

from typing import Sequence

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures import util
from alphastar.architectures.components import common
from alphastar.architectures.components import merge
from alphastar.architectures.components import units
from alphastar.architectures.components import vector
from alphastar.architectures.components import visual
import jax.numpy as jnp
import ml_collections


def _get_sampling(argument_name: types.ArgumentName,
                  num_logits: int,
                  is_training: bool,
                  config: ml_collections.ConfigDict) -> modular.Component:
  """Sampling module during inference, or teacher forcing during training."""
  if is_training:
    return common.ActionFromBehaviourFeatures(
        name="action_from_behaviour_features",
        argument_name=argument_name,
        max_action_value=num_logits - 1)
  else:
    return common.Sample(
        name="sample",
        argument_name=argument_name,
        num_logits=num_logits,
        **config.sample)


def get_vector_head(argument_name: types.ArgumentName,
                    action_spec: types.ActionSpec,
                    vector_stream_size: int,
                    is_training: bool,
                    overlap_len: int,
                    config: ml_collections.ConfigDict
                    ) -> modular.Component:
  """Produce logits and action for an argument and embed it to vector_stream."""
  num_logits = action_spec[argument_name].maximum + 1
  component = modular.SequentialComponent(name=f"{argument_name}_head")
  component.append(vector.Resnet(
      name="resnet",
      input_size=vector_stream_size,
      input_name="vector_stream",
      output_name=f"{argument_name}_logits_vector_stream",
      **config.resnet))
  component.append(vector.Logits(
      name="logits",
      logits_output_name=("logits", argument_name),
      mask_output_name=("masks", argument_name),
      num_logits=num_logits,
      input_size=vector_stream_size,
      input_name=f"{argument_name}_logits_vector_stream",
      **config.logits))
  component.append(_get_sampling(
      argument_name=argument_name,
      num_logits=num_logits,
      is_training=is_training,
      config=config.sampling))
  if argument_name == util.Argument.FUNCTION:
    component.append(common.ArgumentMasks(
        name="argument_masks",
        action_spec=action_spec))
  component.append(common.FeatureToNextState(
      name="action_to_next_state",
      input_name=("action", argument_name),
      output_name=("action", argument_name),
      stream_shape=(),
      stream_dtype=jnp.int32,
      overlap_len=overlap_len))
  component.append(vector.Embedding(
      name="action_embedding",
      input_name=("action", argument_name),
      output_name=f"{argument_name}_embedding",
      mask_name=("argument_masks", argument_name),
      num_classes=num_logits,
      output_size=vector_stream_size))
  component.append(merge.VectorMerge(
      name="embedding_merge",
      input_sizes={"vector_stream": vector_stream_size,
                   f"{argument_name}_embedding": vector_stream_size},
      output_name="vector_stream",
      output_size=vector_stream_size,
      **config.embedding_merge))
  return component


def get_unit_tags_head(obs_spec: types.ObsSpec,
                       action_spec: types.ActionSpec,
                       vector_stream_size: int,
                       units_stream_size: int,
                       is_training: bool,
                       config: ml_collections.ConfigDict) -> modular.Component:
  """Produce logits and action for unit_tags and embed it to vector_stream."""
  num_logits = action_spec["unit_tags"].maximum + 1
  max_num_selected_units = action_spec["unit_tags"].shape[0]
  max_num_observed_units = obs_spec["raw_units"].shape[0]
  assert num_logits == max_num_observed_units + 1
  inner_component = modular.SequentialComponent(name="inner_component")
  inner_component.append(units.PointerLogits(
      name="logits",
      max_num_observed_units=max_num_observed_units,
      num_raw_unit_features=obs_spec["raw_units"].shape[1],
      logits_output_name=("pre_logits", util.Argument.UNIT_TAGS),
      mask_output_name=("pre_masks", util.Argument.UNIT_TAGS),
      query_input_size=vector_stream_size,
      keys_input_size=config.keys_mlp.layer_sizes[-1],
      key_size=config.keys_mlp.layer_sizes[-1],
      unit_tags_masking=units.UnitTagsMasking.SELECTABLE,
      query_input_name="unit_tags_query",
      keys_input_name="unit_tags_keys",
      **config.inner_component.logits))
  inner_component.append(units.FinalizeUnitTagsLogits(
      name="finalize_unit_tags_logits",
      input_logits_name=("pre_logits", util.Argument.UNIT_TAGS),
      input_mask_name=("pre_masks", util.Argument.UNIT_TAGS),
      output_logits_name=("logits", util.Argument.UNIT_TAGS),
      output_mask_name=("masks", util.Argument.UNIT_TAGS),
      max_num_observed_units=max_num_observed_units,
      vector_input_name="unit_tags_query",
      vector_input_size=vector_stream_size))
  inner_component.append(_get_sampling(
      argument_name=util.Argument.UNIT_TAGS,
      num_logits=num_logits,
      is_training=is_training,
      config=config.inner_component.sampling))
  inner_component.append(units.PointerEmbedding(
      name="embedding",
      num_embeddings=max_num_observed_units,
      embeddings_size=config.small_embeddings_mlp.layer_sizes[-1],
      index_input_name=("action", util.Argument.UNIT_TAGS),
      embeddings_input_name="small_unit_tag_embeddings",
      output_name="unit_tags_embedding"))
  inner_component.append(merge.VectorMerge(
      name="embedding_merge",
      input_sizes={
          "unit_tags_query": vector_stream_size,
          "unit_tags_embedding": config.small_embeddings_mlp.layer_sizes[-1]},
      output_name="unit_tags_query",
      output_size=vector_stream_size,
      **config.inner_component.embedding_merge))

  component = modular.SequentialComponent(name="unit_tag_head")
  component.append(vector.Resnet(
      name="query_resnet",
      input_size=vector_stream_size,
      input_name="vector_stream",
      output_name="unit_tags_query",
      **config.query_resnet))
  component.append(units.MLP(
      name="keys_mlp",
      max_num_observed_units=max_num_observed_units,
      units_stream_size=units_stream_size,
      input_name="units_stream",
      output_name="unit_tags_keys",
      **config.keys_mlp))
  component.append(units.MLP(
      name="small_embeddings_mlp",
      max_num_observed_units=max_num_observed_units,
      units_stream_size=units_stream_size,
      input_name="units_stream",
      output_name="small_unit_tag_embeddings",
      **config.small_embeddings_mlp))
  component.append(units.MLP(
      name="large_embeddings_mlp",
      max_num_observed_units=max_num_observed_units,
      units_stream_size=units_stream_size,
      input_name="units_stream",
      output_name="large_unit_tag_embeddings",
      **config.large_embeddings_mlp))
  unit_tags_head_per_step_inputs = []
  if is_training:
    unit_tags_head_per_step_inputs.append(
        ("behaviour_features", "action", util.Argument.UNIT_TAGS))
  component.append(units.UnitTagsHead(
      name="recurrent_unit_tags_head",
      inner_component=inner_component,
      constant_inputs=["unit_tags_keys",
                       "small_unit_tag_embeddings",
                       ("observation", "raw_units")],
      carries=["unit_tags_query"],
      per_step_inputs=unit_tags_head_per_step_inputs,
      per_step_outputs=[("logits", util.Argument.UNIT_TAGS),
                        ("masks", util.Argument.UNIT_TAGS),
                        ("action", util.Argument.UNIT_TAGS)],
      max_num_selected_units=max_num_selected_units,
      max_num_observed_units=max_num_observed_units,
      action_output=("action", util.Argument.UNIT_TAGS)))
  component.append(units.BinaryVectorPointerEmbedding(
      name="embedding",
      num_embeddings=max_num_observed_units,
      embeddings_size=config.large_embeddings_mlp.layer_sizes[-1],
      mask_input_name="selected_unit_tags",
      embeddings_input_name="large_unit_tag_embeddings",
      output_name="unit_tags_embedding"))
  component.append(merge.VectorMerge(
      name="embedding_merge",
      input_sizes={
          "vector_stream": vector_stream_size,
          "unit_tags_embedding": config.large_embeddings_mlp.layer_sizes[-1]},
      output_name="vector_stream",
      output_size=vector_stream_size,
      **config.embedding_merge))
  component.append(merge.UnitsMerge(
      name="selected_units_merge",
      input_sizes={"units_stream": units_stream_size,
                   "selected_unit_tags": None},
      input_dtypes={"selected_unit_tags": jnp.bool_},
      max_num_observed_units=max_num_observed_units,
      output_name="units_stream_after_embedding",
      output_size=units_stream_size,
      **config.selected_units_merge))
  return component


def get_target_unit_tag_head(obs_spec: types.ObsSpec,
                             action_spec: types.ActionSpec,
                             vector_stream_size: int,
                             units_stream_size: int,
                             is_training: bool,
                             config: ml_collections.ConfigDict
                             ) -> modular.Component:
  """Produce logits and action for target_unit_tag."""
  num_logits = action_spec["target_unit_tag"].maximum + 1
  assert num_logits == obs_spec["raw_units"].shape[0]
  component = modular.SequentialComponent(name="target_unit_tag_head")
  component.append(vector.Resnet(
      name="resnet",
      input_size=vector_stream_size,
      input_name="vector_stream",
      output_name="target_unit_tag_query",
      **config.resnet))
  component.append(units.PointerLogits(
      name="logits",
      max_num_observed_units=obs_spec["raw_units"].shape[0],
      num_raw_unit_features=obs_spec["raw_units"].shape[1],
      logits_output_name=("logits", util.Argument.TARGET_UNIT_TAG),
      mask_output_name=("masks", util.Argument.TARGET_UNIT_TAG),
      query_input_size=vector_stream_size,
      keys_input_size=units_stream_size,
      unit_tags_masking=units.UnitTagsMasking.TARGETABLE_WITH_CAMERA,
      query_input_name="target_unit_tag_query",
      keys_input_name="units_stream_after_embedding",
      **config.logits))
  component.append(_get_sampling(
      argument_name=util.Argument.TARGET_UNIT_TAG,
      num_logits=num_logits,
      is_training=is_training,
      config=config.sampling))
  return component


def get_world_head(obs_spec: types.ObsSpec,
                   action_spec: types.ActionSpec,
                   vector_stream_size: int,
                   visual_stream_sizes: Sequence[int],
                   is_training: bool,
                   config: ml_collections.ConfigDict
                   ) -> modular.Component:
  """Produce logits and action for an argument and embed it to vector stream."""
  num_logits = action_spec["world"].maximum + 1
  world_size = util.get_world_size(action_spec)
  spatial_size = obs_spec["minimap_height_map"].shape[0]
  component = modular.SequentialComponent(name="world_head")
  full_downscale_factor = (
      config.visual_upscale.upscale_factor ** len(visual_stream_sizes))
  component.append(vector.ToVisual(
      name="vector_to_visual",
      input_name="vector_stream",
      output_name=f"visual_stream_ds{full_downscale_factor}_from_vector",
      input_size=vector_stream_size,
      output_spatial_size=spatial_size // full_downscale_factor,
      output_features_size=visual_stream_sizes[-1],
      **config.vector_to_visual))
  component.append(merge.SumMerge(
      name="vector_to_visual_merge",
      input_names=[
          f"visual_stream_ds{full_downscale_factor}_from_vector",
          f"visual_stream_ds{full_downscale_factor}"],
      output_name=f"visual_stream_ds{full_downscale_factor}",
      stream_shape=(spatial_size // full_downscale_factor,
                    spatial_size // full_downscale_factor,
                    visual_stream_sizes[-1]),
      ))
  component.append(visual.Resnet(
      name="resnet",
      input_name=f"visual_stream_ds{full_downscale_factor}",
      output_name=f"visual_stream_ds{full_downscale_factor}",
      input_spatial_size=spatial_size // full_downscale_factor,
      input_features_size=visual_stream_sizes[-1],
      **config.resnet))
  for i in range(len(visual_stream_sizes) - 1, 0, -1):
    us = config.visual_upscale.upscale_factor
    ids = int(us**(i + 1))  # input downsampling factor
    ods = int(us**i)  # output downsampling factor
    component.append(visual.Upscale(
        name=f"downscale_ds{ods}",
        input_name=f"visual_stream_ds{ids}",
        output_name=f"visual_stream_ds{ods}_upsample",
        input_spatial_size=spatial_size // ids,
        input_features_size=visual_stream_sizes[i],
        output_features_size=visual_stream_sizes[i - 1],
        **config.visual_upscale))
    component.append(merge.SumMerge(
        name=f"visual_head_merge_ds{ods}",
        input_names=[
            f"visual_stream_ds{ods}", f"visual_stream_ds{ods}_upsample"],
        output_name=f"visual_stream_ds{ods}",
        stream_shape=(spatial_size // ods,
                      spatial_size // ods,
                      visual_stream_sizes[i - 1]),
        ))
  component.append(visual.Upscale(
      name="downscale_ds1",
      input_name=f"visual_stream_ds{ods}",
      output_name="visual_stream_ds1",
      input_spatial_size=spatial_size // ods,
      input_features_size=visual_stream_sizes[0],
      output_features_size=config.visual_feature_size_ds1,
      **config.visual_upscale))
  component.append(visual.Logits(
      name="logits",
      input_name="visual_stream_ds1",
      input_spatial_size=spatial_size,
      input_features_size=config.visual_feature_size_ds1,
      logits_output_name=("logits", util.Argument.WORLD),
      mask_output_name=("masks", util.Argument.WORLD),
      upscale_factor=world_size // spatial_size,
      kernel_size=world_size // spatial_size,
      **config.logits))
  component.append(_get_sampling(
      argument_name=util.Argument.WORLD,
      num_logits=num_logits,
      is_training=is_training,
      config=config.sampling))
  return component

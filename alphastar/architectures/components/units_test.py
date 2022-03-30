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

"""Tests for units."""

from typing import Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import common
from alphastar.architectures.components import merge
from alphastar.architectures.components import test_utils
from alphastar.architectures.components import units
from alphastar.architectures.components import util
from alphastar.architectures.components.static_data import unit_encoder_data
from alphastar.commons import sample
from dm_env import specs
import jax.numpy as jnp
import numpy as np


_UNIT_TAGS = 'unit_tags'


class UnitsTest(test_utils.ComponentTest):
  """Basic tests for the units components."""

  def test_encode_one_hot(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[1] = 1  # alliance
    self.assertEqual(units._encode_one_hot(jnp.asarray(raw_unit), 1), (1, 5))

  def test_encode_capped_one_hot(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[1] = 1  # alliance
    self.assertEqual(
        units._encode_capped_one_hot(jnp.asarray(raw_unit), 1), (1, 5))
    raw_unit[1] = 10_000  # alliance
    self.assertEqual(
        units._encode_capped_one_hot(jnp.asarray(raw_unit), 1), (4, 5))

  def test_encode_sqrt_one_hot(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[2] = 142  # health
    raw_unit[3] = 0  # shield
    self.assertEqual(
        units._encode_sqrt_one_hot(jnp.asarray(raw_unit), 2), (11, 39))
    self.assertEqual(
        units._encode_sqrt_one_hot(jnp.asarray(raw_unit), 3), (0, 32))
    raw_unit[2] = 10_000  # health
    self.assertEqual(
        units._encode_sqrt_one_hot(jnp.asarray(raw_unit), 2), (38, 39))

  def test_encode_divided_one_hot(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[2] = 142  # health
    self.assertEqual(
        units._encode_divided_one_hot(jnp.asarray(raw_unit), 2, 10), (14, 151))
    self.assertEqual(
        units._encode_divided_one_hot(jnp.asarray(raw_unit), 2, 9), (15, 167))
    self.assertEqual(
        units._encode_divided_one_hot(jnp.asarray(raw_unit), 2, 2), (71, 751))
    raw_unit[2] = 10_000  # health
    self.assertEqual(
        units._encode_divided_one_hot(jnp.asarray(raw_unit), 2, 10), (150, 151))

  def test_encode_mined_resource_one_hot(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[0] = 149  # unit_type (VespeneGeyser)
    raw_unit[21] = 1042  # vespene_contents
    self.assertEqual(
        units._encode_mined_resource_one_hot(jnp.asarray(raw_unit), 21, 256),
        (34, 51))
    raw_unit[0] = 18  # unit_type (Barracks)
    self.assertEqual(
        units._encode_mined_resource_one_hot(jnp.asarray(raw_unit), 21, 256),
        (0, 51))
    raw_unit[0] = 149  # unit_type (VespeneGeyser)
    raw_unit[21] = 10_000  # vespene_contents
    self.assertEqual(
        units._encode_mined_resource_one_hot(jnp.asarray(raw_unit), 21, 256),
        (0, 51))

  def test_encode_addon_lookup(self):
    addon_lookup = unit_encoder_data.get_addon_lookup(256)
    self.assertEqual(
        units._encode_lookup(jnp.asarray(34), addon_lookup), (1, 7))
    self.assertEqual(
        units._encode_lookup(jnp.asarray(0), addon_lookup), (0, 7))
    self.assertEqual(
        units._encode_lookup(jnp.asarray(10_000), addon_lookup), (0, 7))

  def test_encode_order_id_lookup(self):
    action_spec = types.SpecDict({
        'function': specs.BoundedArray((), jnp.int32, 0, 555)})
    function_list = util.get_function_list(action_spec)
    function_names = [f.name for f in function_list]
    order_id_lookup = unit_encoder_data.get_order_id_lookup(function_names)
    id_1 = [x.id for x in function_list if x.name == 'Patrol_Patrol_unit'][0]
    id_2 = [x.id for x in function_list if x.name == 'Patrol_unit'][0]
    id_3 = [x.id for x in function_list if x.name == 'Load_unit'][0]
    lookup_1, num1 = units._encode_lookup(jnp.asarray(id_1), order_id_lookup)
    lookup_2, _ = units._encode_lookup(jnp.asarray(id_2), order_id_lookup)
    lookup_3, _ = units._encode_lookup(jnp.asarray(id_3), order_id_lookup)
    self.assertEqual(lookup_1, lookup_2)
    self.assertEqual(num1, max(order_id_lookup) + 1)
    self.assertNotEqual(lookup_1, lookup_3)

    id_4 = [x.id for x in function_list if x.name == 'no_op'][0]
    id_5 = [x.id for x in function_list
            if x.name == 'Research_ZergMeleeWeaponsLevel2_quick'][0]
    build_queue_order_id_lookup = (
        unit_encoder_data.get_build_queue_order_id_lookup(function_names))
    lookup_1b, num_1b = units._encode_lookup(
        jnp.asarray(id_1), build_queue_order_id_lookup)
    lookup_3b, _ = units._encode_lookup(
        jnp.asarray(id_3), build_queue_order_id_lookup)
    lookup_4b, _ = units._encode_lookup(
        jnp.asarray(id_4), build_queue_order_id_lookup)
    lookup_5b, _ = units._encode_lookup(
        jnp.asarray(id_5), build_queue_order_id_lookup)
    self.assertEqual(lookup_1b, lookup_3b)
    self.assertEqual(num_1b, max(build_queue_order_id_lookup) + 1)
    self.assertEqual(lookup_1b, lookup_4b)
    self.assertNotEqual(lookup_4b, lookup_5b)

  def test_features_embedding(self):
    raw_unit = np.zeros(47, np.int32)
    raw_unit[6] = 42  # build_progress
    raw_unit[7] = 6  # health_ratio
    raw_unit[8] = 255  # shield_ratio
    raw_unit[9] = 0  # energy_ratio
    raw_unit[13] = 3  # y (not used)
    raw_unit[36] = 4  # order_progress_1
    raw_unit[37] = 78  # order_progress_1
    raw_unit[38] = 4  # order_id_2 (not used)
    rescales = {
        6: 1. / 100,
        7: 1. / 255,
        8: 1. / 255,
        9: 1. / 255,
        36: 1. / 100,
        37: 1. / 100}

    embeddings = units._features_embedding(jnp.asarray(raw_unit), rescales)
    np.testing.assert_allclose(embeddings, jnp.asarray([
        0.42, 6./255, 1., 0., 0.04, 0.78]))

  def test_binary_scale_embedding(self):
    np.testing.assert_array_equal(
        units._binary_scale_embedding(jnp.asarray(42), 256),
        jnp.asarray([0, 1, 0, 1, 0, 1, 0, 0], jnp.float32))
    np.testing.assert_array_equal(
        units._binary_scale_embedding(jnp.asarray(119), 128),
        jnp.asarray([1, 1, 1, 0, 1, 1, 1], jnp.float32))

  def test_remap_and_one_hot_embedding(self):
    action_spec = types.SpecDict({
        'function': specs.BoundedArray((), jnp.int32, 0, 555)})
    function_list = util.get_function_list(action_spec)
    function_names = [f.name for f in function_list]
    id_1 = [x.id for x in function_list if x.name == 'Halt_Building_quick'][0]
    id_2 = [x.id for x in function_list if x.name == 'Halt_quick'][0]
    order_id_lookup = unit_encoder_data.get_order_id_lookup(function_names)
    one_hot_1 = units._remap_and_one_hot_embedding(
        jnp.asarray(id_1), order_id_lookup)
    one_hot_2 = units._remap_and_one_hot_embedding(
        jnp.asarray(id_2), order_id_lookup)
    np.testing.assert_array_equal(one_hot_1, one_hot_2)

  @parameterized.product(
      is_training=[True, False],
      output_size=[1, 4])
  def test_UnitsEncoder(self,
                        is_training: bool,
                        output_size: int):
    input_spec, action_spec = test_utils.get_test_specs(is_training)
    component = units.UnitsEncoder(
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        output_name='output_stream',
        num_raw_unit_features=input_spec['observation', 'raw_units'].shape[1],
        units_stream_size=output_size,
        action_spec=action_spec,
        num_unit_types=256,
        num_buff_types=5)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      units_stream_size=[3],
      transformer_num_layers=[0, 2],
      transformer_num_heads=[1, 2],
      transformer_key_size=[4],
      transformer_value_size=[5],
      resblocks_num_before=[0, 1],
      resblocks_num_after=[0, 1],
      resblocks_hidden_size=[None, 3],
      use_layer_norm=[True, False])
  def test_Transformer(self,
                       is_training: bool,
                       units_stream_size: int,
                       transformer_num_layers: int,
                       transformer_num_heads: int,
                       transformer_key_size: int,
                       transformer_value_size: int,
                       resblocks_num_before: int,
                       resblocks_num_after: int,
                       resblocks_hidden_size: Optional[int],
                       use_layer_norm: bool):
    input_spec, _ = test_utils.get_test_specs(is_training)
    component = units.Transformer(
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        units_stream_size=units_stream_size,
        transformer_num_layers=transformer_num_layers,
        transformer_num_heads=transformer_num_heads,
        transformer_key_size=transformer_key_size,
        transformer_value_size=transformer_value_size,
        resblocks_num_before=resblocks_num_before,
        resblocks_num_after=resblocks_num_after,
        resblocks_hidden_size=resblocks_hidden_size,
        use_layer_norm=use_layer_norm,
        input_name='input_stream',
        output_name='output_stream')
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_size=[1, 4],
      layer_sizes=[[1], [3, 2]],
      use_layer_norm=[True, False])
  def test_MLP(self,
               is_training: bool,
               input_size: int,
               layer_sizes: Sequence[int],
               use_layer_norm: bool):
    input_spec, _ = test_utils.get_test_specs(is_training)
    component = units.MLP(
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        units_stream_size=input_size,
        layer_sizes=layer_sizes,
        use_layer_norm=use_layer_norm,
        input_name='input_stream',
        output_name='output_stream')
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      units_stream_size=[1, 3],
      units_hidden_sizes=[[], [1], [2, 4]],
      vector_stream_size=[1, 4],
      use_layer_norm=[True, False])
  def test_ToVector(self,
                    is_training: bool,
                    units_stream_size: int,
                    units_hidden_sizes: Sequence[int],
                    vector_stream_size: int,
                    use_layer_norm: bool = True):
    input_spec, _ = test_utils.get_test_specs(is_training)
    component = units.ToVector(
        input_name='input_stream',
        output_name='output_stream',
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        units_stream_size=units_stream_size,
        units_hidden_sizes=units_hidden_sizes,
        vector_stream_size=vector_stream_size,
        use_layer_norm=use_layer_norm)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      units_stream_size=[2],
      units_world_dim=[1, 4],
      units_hidden_sizes=[[], [1], [2, 4]],
      output_spatial_size=[1, 4],
      output_features_size=[1, 3],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False])
  def test_ToVisualScatter(self,
                           is_training: bool,
                           units_stream_size: int,
                           units_world_dim: int,
                           units_hidden_sizes: Sequence[int],
                           output_spatial_size: int,
                           output_features_size: int,
                           kernel_size: int,
                           use_layer_norm: bool):
    input_spec, _ = test_utils.get_test_specs(is_training)
    kwargs = dict(
        input_name='input_stream',
        output_name='output_stream',
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        num_raw_unit_features=input_spec['observation', 'raw_units'].shape[1],
        units_stream_size=units_stream_size,
        units_world_dim=units_world_dim,
        units_hidden_sizes=units_hidden_sizes,
        output_spatial_size=output_spatial_size,
        output_features_size=output_features_size,
        kernel_size=kernel_size,
        use_layer_norm=use_layer_norm)
    if units_world_dim % output_spatial_size != 0:
      with self.assertRaises(ValueError):
        units.ToVisualScatter(**kwargs)
    else:
      component = units.ToVisualScatter(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      query_input_size=[4],
      keys_input_size=[3],
      unit_tags_masking=list(units.UnitTagsMasking),
      num_layers_query=[0, 1],
      num_layers_keys=[0, 1],
      key_size=[3, 4, 5],
      use_layer_norm=[True, False])
  def test_PointerLogits(self,
                         is_training: bool,
                         unit_tags_masking: units.UnitTagsMasking,
                         query_input_size: int,
                         keys_input_size: int,
                         num_layers_query: int,
                         num_layers_keys: int,
                         key_size: int,
                         use_layer_norm: bool):
    input_spec, _ = test_utils.get_test_specs(is_training)
    kwargs = dict(
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        num_raw_unit_features=input_spec['observation', 'raw_units'].shape[1],
        logits_output_name=('logits', 'target_unit_tag'),
        mask_output_name=('masks', 'target_unit_tag'),
        query_input_size=query_input_size,
        keys_input_size=keys_input_size,
        unit_tags_masking=unit_tags_masking,
        query_input_name='vector_stream',
        keys_input_name='units_stream',
        num_layers_query=num_layers_query,
        num_layers_keys=num_layers_keys,
        key_size=key_size,
        use_layer_norm=use_layer_norm)
    expect_error = False
    if num_layers_query == 0 and key_size != query_input_size:
      expect_error = True
    if num_layers_keys == 0 and key_size != keys_input_size:
      expect_error = True
    if expect_error:
      with self.assertRaises(ValueError):
        units.PointerLogits(**kwargs)
    else:
      component = units.PointerLogits(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      vector_input_size=[1, 3])
  def test_FinalizeUnitTagsLogits(self,
                                  is_training: bool,
                                  vector_input_size: int):
    input_spec, _ = test_utils.get_test_specs(is_training)
    component = units.FinalizeUnitTagsLogits(
        input_logits_name='pre_logits_stream',
        input_mask_name='pre_mask_stream',
        output_logits_name='logits_stream',
        output_mask_name='mask_stream',
        vector_input_name='vector_input',
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        vector_input_size=vector_input_size)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.parameters(True, False)
  def test_UnitTagsHead(self, is_training: bool):
    input_spec, action_spec = test_utils.get_test_specs(is_training)
    inner_component = modular.SequentialComponent()
    num_units = input_spec['observation', 'raw_units'].shape[0]
    inner_component.append(units.PointerLogits(
        max_num_observed_units=num_units,
        num_raw_unit_features=input_spec['observation', 'raw_units'].shape[1],
        logits_output_name=('logits', _UNIT_TAGS),
        mask_output_name=('masks', _UNIT_TAGS),
        query_input_size=num_units,
        keys_input_size=6,
        unit_tags_masking=units.UnitTagsMasking.SELECTABLE,
        query_input_name='query_stream',
        keys_input_name='keys_stream',
        num_layers_query=1,
        num_layers_keys=0,
        key_size=6,
        use_layer_norm=True))
    inner_component.append(merge.SumMerge(
        input_names=[('logits', _UNIT_TAGS),
                     'per_step_input_stream'],
        output_name='query_stream',
        stream_shape=(num_units,)))
    inner_component.append(common.Sample(
        argument_name=_UNIT_TAGS,
        num_logits=input_spec['observation', 'raw_units'].shape[0],
        sample_fn=sample.sample))
    component = units.UnitTagsHead(
        inner_component=inner_component,
        constant_inputs=['keys_stream', ('observation', 'raw_units')],
        carries=['query_stream'],
        per_step_inputs=['per_step_input_stream'],
        per_step_outputs=[('masks', _UNIT_TAGS),
                          ('action', _UNIT_TAGS)],
        max_num_selected_units=action_spec['unit_tags'].shape[0],
        max_num_observed_units=num_units,
        action_output=('action', _UNIT_TAGS))
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)


if __name__ == '__main__':
  absltest.main()

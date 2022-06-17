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

"""Tests for agent."""

from absl.testing import absltest
from absl.testing import parameterized
from alphastar.architectures import architectures
from alphastar.commons import jax_utils
from alphastar.modules import agent
from alphastar.unplugged.data import util as data_util
from dm_env import specs
import jax.numpy as jnp


def setUpModule():
  # Disable JAX optimizations in order to speed up compilation.
  jax_utils.disable_jax_optimizations()


def tearDownModule():
  jax_utils.restore_jax_config()


def get_test_specs():
  max_num_selected_units = 4
  obs_spec = {
      'away_race_observed': specs.Array((), jnp.int32),
      'away_race_requested': specs.Array((), jnp.int32),
      'camera': specs.Array((256, 256), jnp.int32),
      'camera_position': specs.Array((2,), jnp.int32),
      'camera_size': specs.Array((2,), jnp.int32),
      'game_loop': specs.Array((), jnp.int32),
      'home_race_requested': specs.Array((), jnp.int32),
      'minimap_alerts': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=5),
      'minimap_buildable': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=1),
      'minimap_creep': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=2),
      'minimap_height_map': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=255),
      'minimap_pathable': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=1),
      'minimap_player_relative': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=3),
      'minimap_visibility_map': specs.BoundedArray(
          (128, 128), jnp.int32, minimum=0, maximum=2),
      'mmr': specs.Array((), jnp.int32),
      'player': specs.Array((7,), jnp.int32),
      'raw_units': specs.Array((10, 47), jnp.int32),
      'unit_counts_bow': specs.Array((5,), jnp.int32),
      'upgrades_fixed_length': specs.BoundedArray(
          (8,), jnp.int32, minimum=0, maximum=13)}
  action_spec = {
      'function': specs.BoundedArray((), jnp.int32, minimum=0, maximum=10),
      'delay': specs.BoundedArray((), jnp.int32, minimum=0, maximum=6),
      'queued': specs.BoundedArray((), jnp.int32, minimum=0, maximum=2),
      'repeat': specs.BoundedArray((), jnp.int32, minimum=0, maximum=4),
      'unit_tags': specs.BoundedArray(
          (max_num_selected_units,), jnp.int32, minimum=0, maximum=10),
      'target_unit_tag': specs.BoundedArray(
          (), jnp.int32, minimum=0, maximum=9),
      'world': specs.BoundedArray(
          (), jnp.int32, minimum=0, maximum=256**2 - 1)}
  return obs_spec, action_spec


class EvaluatorTest(parameterized.TestCase):

  @parameterized.parameters(('alphastar.dummy', False),
                            ('alphastar.dummy', True),
                            ('alphastar.lite', False),
                            ('alphastar.lite', True))
  def test_no_recompilation(self, architecture: str, is_training: bool):
    builder = architectures.get_architecture(architecture)
    obs_spec, action_spec = get_test_specs()
    behaviour_features_spec = {'action': action_spec} if is_training else None
    input_spec = data_util.get_input_spec(
        obs_spec=obs_spec, behaviour_features_spec=behaviour_features_spec)
    component = builder(input_spec, action_spec, is_training, 0, 0)
    alphastar_agent = agent.AlphaStarAgent(component)
    alphastar_agent.warmup()
    with jax_utils.no_jax_compilation_allowed():
      alphastar_agent.warmup()


if __name__ == '__main__':
  absltest.main()

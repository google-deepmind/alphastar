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

"""Utility functions to test components."""

from typing import Tuple

from alphastar import types
from alphastar.architectures import modular
from alphastar.unplugged.data import util as data_util
from dm_env import specs
import haiku as hk
import jax
from jax import test_util as jtu
import jax.numpy as jnp


def get_test_specs(is_training: bool
                   ) -> Tuple[types.InputSpec, types.ActionSpec]:
  """Return some input_spec and action_spec for testing."""
  max_num_selected_units = 4
  obs_spec = types.SpecDict()
  obs_spec['away_race_observed'] = specs.Array((), jnp.int32)
  obs_spec['away_race_requested'] = specs.Array((), jnp.int32)
  obs_spec['camera'] = specs.Array((256, 256), jnp.int32)
  obs_spec['camera_position'] = specs.Array((2,), jnp.int32)
  obs_spec['camera_size'] = specs.Array((2,), jnp.int32)
  obs_spec['game_loop'] = specs.Array((), jnp.int32)
  obs_spec['home_race_requested'] = specs.Array((), jnp.int32)
  obs_spec['minimap_alerts'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=5)
  obs_spec['minimap_buildable'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=1)
  obs_spec['minimap_creep'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=2)
  obs_spec['minimap_height_map'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=255)
  obs_spec['minimap_pathable'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=1)
  obs_spec['minimap_player_relative'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=3)
  obs_spec['minimap_visibility_map'] = specs.BoundedArray(
      (128, 128), jnp.int32, minimum=0, maximum=2)
  obs_spec['mmr'] = specs.Array((), jnp.int32)
  obs_spec['player'] = specs.Array((7,), jnp.int32)
  obs_spec['raw_units'] = specs.Array((10, 47), jnp.int32)
  obs_spec['unit_counts_bow'] = specs.Array((5,), jnp.int32)
  obs_spec['upgrades_fixed_length'] = specs.BoundedArray(
      (8,), jnp.int32, minimum=0, maximum=13)

  action_spec = types.SpecDict()
  action_spec['function'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=10)
  action_spec['delay'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=6)
  action_spec['queued'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=2)
  action_spec['repeat'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=4)
  action_spec['unit_tags'] = specs.BoundedArray(
      (max_num_selected_units,), jnp.int32, minimum=0, maximum=10)
  action_spec['target_unit_tag'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=9)
  action_spec['world'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=256**2 - 1)

  input_spec = types.SpecDict()
  input_spec['observation'] = obs_spec
  input_spec['step_type'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=2)
  if is_training:
    input_spec['behaviour_features', 'action'] = action_spec.copy()
    # Delay behaviour feature input is unbounded:
    input_spec['behaviour_features', 'action', 'delay'] = specs.BoundedArray(
        (), jnp.int32, minimum=0, maximum=1_000_000_000)
  return input_spec, action_spec

@jtu.with_config(jax_numpy_rank_promotion="allow")
class ComponentTest(jtu.JaxTestCase):
  """Basic class to test component input/output consistency."""

  def _test_component(self,
                      component: modular.Component,
                      batch_size: int = 2,
                      unroll_len: int = 3):
    """Test that the forward pass does not crash, and has correct shapes."""
    inputs = data_util.get_dummy_observation(
        component.input_spec, batch_size=batch_size, unroll_len=unroll_len)
    prev_state = data_util.get_dummy_observation(
        component.prev_state_spec, batch_size=batch_size, unroll_len=None)
    rng_key = jax.random.PRNGKey(42)
    initial_state_init, initial_state_apply = hk.transform(
        jax.vmap(component.initial_state, axis_size=batch_size))
    initial_state = initial_state_apply(initial_state_init(rng_key), rng_key)
    component.next_state_spec.validate(
        initial_state, num_leading_dims_to_ignore=1)
    forward_init, forward_apply = hk.transform_with_state(
        jax.vmap(component.unroll))
    params, hk_state = forward_init(rng_key, inputs, prev_state)
    (outputs, next_state, _), _ = forward_apply(
        params, hk_state, rng_key, inputs, prev_state)
    for v in outputs.values():
      self.assertEqual(v.shape[:2], (batch_size, unroll_len))
    component.output_spec.validate(outputs, num_leading_dims_to_ignore=2)
    for v in next_state.values():
      self.assertEqual(v.shape[0], batch_size)
    component.next_state_spec.validate(next_state, num_leading_dims_to_ignore=1)

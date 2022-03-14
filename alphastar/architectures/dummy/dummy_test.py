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

"""Tests for dummy."""

from absl.testing import absltest
from absl.testing import parameterized
from alphastar.architectures.dummy import dummy
from dm_env import specs
import haiku as hk
import jax
from jax import test_util as jtu
import jax.numpy as jnp


class DummyTest(jtu.JaxTestCase):
  """Basic tests for the dummy architecture."""

  @parameterized.parameters(True, False)
  def test_forward(self, is_training: bool):
    """Test that the forward pass does not crash, and has correct shapes."""

    batch_size = 2
    unroll_len = 3 if is_training else 1
    max_num_selected_units = 4
    obs_spec = {'player': specs.Array((7,), jnp.int32)}
    action_spec = {
        'function': specs.BoundedArray((), jnp.int32, minimum=0, maximum=10),
        'delay': specs.BoundedArray((), jnp.int32, minimum=0, maximum=6),
        'queued': specs.BoundedArray((), jnp.int32, minimum=0, maximum=2),
        'repeat': specs.BoundedArray((), jnp.int32, minimum=0, maximum=4),
        'unit_tags': specs.BoundedArray(
            (max_num_selected_units,), jnp.int32, minimum=0, maximum=10),
        'target_unit_tag': specs.BoundedArray(
            (), jnp.int32, minimum=0, maximum=10),
        'world': specs.BoundedArray((), jnp.int32, minimum=0, maximum=25)}
    input_spec = {
        'observation': obs_spec,
        'step_type': specs.BoundedArray((), jnp.int32, minimum=0, maximum=2)}
    if is_training:
      input_spec['behaviour_features'] = {'action': action_spec}
    alphastar = dummy.get_alphastar_dummy(
        input_spec=input_spec,
        action_spec=action_spec,
        is_training=is_training,
        overlap_len=0,
        burnin_len=0)
    def make_input(spec):
      return jnp.zeros((batch_size, unroll_len) + spec.shape, spec.dtype)
    inputs = jax.tree_map(make_input, alphastar.input_spec)
    rng_key = jax.random.PRNGKey(42)
    initial_state_init, initial_state_apply = hk.transform(
        jax.vmap(alphastar.initial_state, axis_size=batch_size))
    initial_state = initial_state_apply(initial_state_init(rng_key), rng_key)
    forward_init, forward_apply = hk.transform(
        jax.vmap(alphastar.unroll))
    params = forward_init(rng_key, inputs, initial_state)
    outputs, next_state, _ = forward_apply(
        params, rng_key, inputs, initial_state)
    for v in outputs.values():
      self.assertEqual(v.shape[:2], (batch_size, unroll_len))
    alphastar.output_spec.validate(outputs, num_leading_dims_to_ignore=2)
    for v in next_state.values():
      self.assertEqual(v.shape[0], batch_size)
    alphastar.next_state_spec.validate(next_state, num_leading_dims_to_ignore=1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

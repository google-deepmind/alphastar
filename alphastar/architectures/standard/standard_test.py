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
from alphastar.architectures.components import test_utils
from alphastar.architectures.standard import standard
from alphastar.architectures.standard.configs import lite as config_lite
from jax import test_util as jtu


class LiteTest(test_utils.ComponentTest):
  """Basic tests for the standard architecture."""

  @parameterized.parameters(True, False)
  def test_forward(self, is_training: bool):
    """Test that the forward pass does not crash, and has correct shapes."""
    batch_size = 2
    unroll_len = 3 if is_training else 1
    overlap_len = 1 if is_training else 0
    burnin_len = 0
    input_spec, action_spec = test_utils.get_test_specs(is_training)
    if is_training:
      input_spec['behaviour_features'] = {'action': action_spec}
    alphastar = standard.get_alphastar_standard(
        input_spec=input_spec,
        action_spec=action_spec,
        is_training=is_training,
        overlap_len=overlap_len,
        burnin_len=burnin_len,
        config=config_lite.get_config())
    self._test_component(
        alphastar, batch_size=batch_size, unroll_len=unroll_len)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

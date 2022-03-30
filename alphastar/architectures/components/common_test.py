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

"""Tests for common."""

from absl.testing import absltest
from absl.testing import parameterized
from alphastar import types
from alphastar.architectures import util as modular_util
from alphastar.architectures.components import common
from alphastar.architectures.components import test_utils
from alphastar.commons import sample


_ALL_ARGUMENT_NAMES = (
    modular_util.Argument.FUNCTION,
    modular_util.Argument.DELAY,
    modular_util.Argument.QUEUED,
    modular_util.Argument.REPEAT,
    modular_util.Argument.UNIT_TAGS,
    modular_util.Argument.TARGET_UNIT_TAG,
    modular_util.Argument.WORLD
)


class CommonTest(test_utils.ComponentTest):
  """Basic tests for the common components."""

  @parameterized.parameters(*_ALL_ARGUMENT_NAMES)
  def test_ActionFromBehaviourFeatures(self,
                                       argument_name: types.ArgumentName):
    component = common.ActionFromBehaviourFeatures(argument_name=argument_name)
    self._test_component(component, batch_size=2, unroll_len=3)

  @parameterized.product(
      is_training=[True, False],
      argument_name=_ALL_ARGUMENT_NAMES)
  def test_Sample(self,
                  is_training: bool,
                  argument_name: types.ArgumentName):
    _, action_spec = test_utils.get_test_specs(is_training)
    component = common.Sample(argument_name=argument_name,
                              num_logits=action_spec[argument_name].maximum + 1,
                              sample_fn=sample.sample)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.parameters(True, False)
  def test_ArgumentMasks(self, is_training: bool):
    _, action_spec = test_utils.get_test_specs(is_training)
    component = common.ArgumentMasks(action_spec=action_spec)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.parameters(True, False)
  def test_FeatureFromPrevState(self, is_training: bool):
    component = common.FeatureFromPrevState(input_name='input_stream',
                                            output_name='output_stream',
                                            is_training=is_training,
                                            stream_shape=(2, 4))
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.parameters((True, 0), (True, 2), (False, 0))
  def test_FeatureToNextState(self, is_training: bool, overlap_len: int):
    component = common.FeatureToNextState(input_name='input_stream',
                                          output_name='output_stream',
                                          stream_shape=(2, 4),
                                          overlap_len=overlap_len)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  def test_FeatureToNextState_error(self):
    component = common.FeatureToNextState(input_name='input_stream',
                                          output_name='output_stream',
                                          stream_shape=(2, 4),
                                          overlap_len=2)
    with self.assertRaises(ValueError):
      self._test_component(component, batch_size=2, unroll_len=1)


if __name__ == '__main__':
  absltest.main()

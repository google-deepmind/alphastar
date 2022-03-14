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

"""Tests for vector."""

from typing import Callable, Optional, Sequence, Tuple

from absl.testing import absltest
from absl.testing import parameterized
from alphastar import types
from alphastar.architectures.components import test_utils
from alphastar.architectures.components import vector
import chex
from jax import test_util as jtu
import jax.numpy as jnp


class VectorTest(test_utils.ComponentTest):
  """Basic tests for the vector components."""

  @parameterized.product(
      is_training=[True, False],
      num_features=[1, 3],
      output_size=[1, 4],
      fun=[jnp.sqrt, jnp.log1p])
  def test_VectorEncoder(self,
                         is_training: bool,
                         num_features: int,
                         output_size: int,
                         fun: Callable[[chex.Array], chex.Array]):
    component = vector.VectorEncoder(input_name='input_stream',
                                     output_name='output_stream',
                                     num_features=num_features,
                                     output_size=output_size,
                                     fun=fun)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      num_classes=[1, 3],
      output_size=[1, 4],
      mask_name=[None, 'mask_stream'],
      fun=[None, lambda x: x * 0.1])
  def test_Embedding(self,
                     is_training: bool,
                     num_classes: int,
                     output_size: int,
                     mask_name: Optional[types.StreamType] = None,
                     fun: Optional[float] = None):
    component = vector.Embedding(input_name='input_stream',
                                 output_name='output_stream',
                                 num_classes=num_classes,
                                 output_size=output_size,
                                 mask_name=mask_name,
                                 fun=fun)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_size=[1, 4],
      num_classes=[1, 3])
  def test_FixedLengthToMask(self,
                             is_training: bool,
                             input_size: int,
                             num_classes: int):
    component = vector.FixedLengthToMask(input_name='input_stream',
                                         output_name='output_stream',
                                         input_size=input_size,
                                         num_classes=num_classes)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_size=[1, 4],
      output_size=[1, 3],
      mask_name=[None, 'mask_stream'])
  def test_BinaryVectorEmbedding(self,
                                 is_training: bool,
                                 input_size: int,
                                 output_size: int,
                                 mask_name: Optional[types.StreamType] = None):
    component = vector.BinaryVectorEmbedding(input_name='input_stream',
                                             output_name='output_stream',
                                             input_size=input_size,
                                             output_size=output_size,
                                             mask_name=mask_name)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      encoding_size=[2, 4],
      output_size=[1, 3])
  def test_ClockFeatureEncoder(self,
                               is_training: bool,
                               encoding_size: int,
                               output_size: int):
    component = vector.ClockFeatureEncoder(input_name='input_stream',
                                           output_name='output_stream',
                                           encoding_size=encoding_size,
                                           output_size=output_size)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_size=[1, 4],
      num_resblocks=[0, 1, 2],
      use_layer_norm=[True, False])
  def test_Resnet(self,
                  is_training: bool,
                  input_size: int,
                  num_resblocks: int,
                  use_layer_norm: bool):
    component = vector.Resnet(input_name='input_stream',
                              output_name='output_stream',
                              input_size=input_size,
                              num_resblocks=num_resblocks,
                              use_layer_norm=use_layer_norm)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_size=[1, 4],
      output_features_size=[1, 3],
      features_size_and_upscale=[[3, [3], 1], [8, [5, 3], 2], [3, [3], 3]],
      use_layer_norm=[True, False],
      kernel_size=[1, 2, 3])
  def test_ToVisual(self,
                    is_training: bool,
                    input_size: int,
                    output_features_size: int,
                    features_size_and_upscale: Tuple[int, Sequence[int], int],
                    use_layer_norm: bool,
                    kernel_size: int):
    output_spatial_size, hidden_feature_sizes, upscale_factor = (
        features_size_and_upscale)
    component = vector.ToVisual(input_name='input_stream',
                                output_name='output_stream',
                                input_size=input_size,
                                output_spatial_size=output_spatial_size,
                                output_features_size=output_features_size,
                                hidden_feature_sizes=hidden_feature_sizes,
                                upscale_factor=upscale_factor,
                                use_layer_norm=use_layer_norm,
                                kernel_size=kernel_size)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      num_logits=[1, 4],
      input_size=[1, 3],
      num_linear_layers=[1, 2],
      use_layer_norm=[True, False])
  def test_Logits(self,
                  is_training: bool,
                  num_logits: int,
                  input_size: int,
                  num_linear_layers: int = 2,
                  use_layer_norm: bool = True):
    component = vector.Logits(num_logits=num_logits,
                              input_size=input_size,
                              logits_output_name=('logits', 'function'),
                              mask_output_name=('masks', 'function'),
                              input_name='input_stream',
                              num_linear_layers=num_linear_layers,
                              use_layer_norm=use_layer_norm)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

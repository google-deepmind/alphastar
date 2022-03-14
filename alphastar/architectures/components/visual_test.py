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

"""Tests for visual."""

from typing import Callable, Optional, Sequence

from absl.testing import absltest
from absl.testing import parameterized
from alphastar.architectures import util
from alphastar.architectures.components import test_utils
from alphastar.architectures.components import visual
import chex
from jax import test_util as jtu
import jax.numpy as jnp


class VisualTest(test_utils.ComponentTest):
  """Basic tests for the visual components."""

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 3],
      downscale_factor=[1, 2, 3],
      output_features_size=[1, 5],
      kernel_size=[1, 2, 3],
      fun=[jnp.sqrt, jnp.log1p])
  def test_SingleFeatureEncoder(self,
                                is_training: bool,
                                input_spatial_size: int,
                                downscale_factor: int,
                                output_features_size: int,
                                kernel_size: int,
                                fun: Callable[[chex.Array], chex.Array]):
    kwargs = dict(
        input_name='input_stream',
        output_name='output_stream',
        input_spatial_size=input_spatial_size,
        downscale_factor=downscale_factor,
        output_features_size=output_features_size,
        kernel_size=kernel_size,
        fun=fun)
    if input_spatial_size % downscale_factor != 0:
      with self.assertRaises(ValueError):
        _ = visual.SingleFeatureEncoder(**kwargs)
    else:
      component = visual.SingleFeatureEncoder(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 3],
      downscale_factor=[1, 2, 3],
      num_classes=[1, 4],
      output_features_size=[1, 5],
      kernel_size=[1, 2, 3])
  def test_Embedding(self,
                     is_training: bool,
                     input_spatial_size: int,
                     downscale_factor: int,
                     num_classes: int,
                     output_features_size: int,
                     kernel_size: int):
    kwargs = dict(
        input_name='input_stream',
        output_name='output_stream',
        input_spatial_size=input_spatial_size,
        downscale_factor=downscale_factor,
        num_classes=num_classes,
        output_features_size=output_features_size,
        kernel_size=kernel_size)
    if input_spatial_size % downscale_factor != 0:
      with self.assertRaises(ValueError):
        _ = visual.Embedding(**kwargs)
    else:
      component = visual.Embedding(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 3],
      downscale_factor=[1, 2, 3],
      output_features_size=[1, 5])
  def test_CameraEncoder(self,
                         is_training: bool,
                         input_spatial_size: int,
                         downscale_factor: int,
                         output_features_size: int):
    kwargs = dict(
        output_name='output_stream',
        input_spatial_size=input_spatial_size,
        downscale_factor=downscale_factor,
        output_features_size=output_features_size)
    if input_spatial_size % downscale_factor != 0:
      with self.assertRaises(ValueError):
        _ = visual.CameraEncoder(**kwargs)
    else:
      component = visual.CameraEncoder(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 4],
      input_features_size=[1, 3],
      output_features_size=[1, 5],
      downscale_factor=[1, 2, 3],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False])
  def test_Downscale(self,
                     is_training: bool,
                     input_spatial_size: int,
                     input_features_size: int,
                     output_features_size: int,
                     downscale_factor: int,
                     kernel_size: int,
                     use_layer_norm: bool):
    kwargs = dict(input_name='input_stream',
                  output_name='output_stream',
                  input_spatial_size=input_spatial_size,
                  input_features_size=input_features_size,
                  output_features_size=output_features_size,
                  downscale_factor=downscale_factor,
                  kernel_size=kernel_size,
                  use_layer_norm=use_layer_norm)
    if input_spatial_size % downscale_factor != 0:
      with self.assertRaises(ValueError):
        _ = visual.Downscale(**kwargs)
    else:
      component = visual.Downscale(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 4],
      input_features_size=[1, 3],
      output_features_size=[1, 5],
      downscale_factor=[1, 2, 3],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False])
  def test_Upscale(self,
                   is_training: bool,
                   input_spatial_size: int,
                   input_features_size: int,
                   output_features_size: int,
                   downscale_factor: int,
                   kernel_size: int,
                   use_layer_norm: bool):
    component = visual.Upscale(input_name='input_stream',
                               output_name='output_stream',
                               input_spatial_size=input_spatial_size,
                               input_features_size=input_features_size,
                               output_features_size=output_features_size,
                               upscale_factor=downscale_factor,
                               kernel_size=kernel_size,
                               use_layer_norm=use_layer_norm)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 4],
      input_features_size=[1, 3],
      num_resblocks=[0, 2],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False],
      num_hidden_feature_planes=[None, 3])
  def test_Resnet(self,
                  is_training: bool,
                  input_spatial_size: int,
                  input_features_size: int,
                  num_resblocks: int,
                  kernel_size: int,
                  use_layer_norm: bool,
                  num_hidden_feature_planes: Optional[int] = None):
    component = visual.Resnet(
        input_name='input_stream',
        output_name='output_stream',
        input_spatial_size=input_spatial_size,
        input_features_size=input_features_size,
        num_resblocks=num_resblocks,
        kernel_size=kernel_size,
        use_layer_norm=use_layer_norm,
        num_hidden_feature_planes=num_hidden_feature_planes)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 4],
      input_features_size=[1, 3],
      output_size=[3],
      hidden_feature_sizes=[[], [2, 3]],
      downscale_factor=[1, 2, 3],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False])
  def test_ToVector(self,
                    is_training: bool,
                    input_spatial_size: int,
                    input_features_size: int,
                    output_size: int,
                    hidden_feature_sizes: Sequence[int],
                    downscale_factor: int,
                    kernel_size: int,
                    use_layer_norm: bool):
    kwargs = dict(input_name='input_stream',
                  output_name='output_stream',
                  input_spatial_size=input_spatial_size,
                  input_features_size=input_features_size,
                  vector_stream_size=output_size,
                  hidden_feature_sizes=hidden_feature_sizes,
                  downscale_factor=downscale_factor,
                  kernel_size=kernel_size,
                  use_layer_norm=use_layer_norm)
    if (input_spatial_size %
        (downscale_factor ** len(hidden_feature_sizes)) != 0):
      with self.assertRaises(ValueError):
        _ = visual.ToVector(**kwargs)
    else:
      component = visual.ToVector(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_spatial_size=[1, 4],
      input_features_size=[1, 3],
      upscale_factor=[1, 2, 3],
      kernel_size=[1, 2, 3],
      use_layer_norm=[True, False],
      use_depth_to_space=[True, False])
  def test_Logits(self,
                  is_training: bool,
                  input_spatial_size: int,
                  input_features_size: int,
                  upscale_factor: int,
                  kernel_size: int,
                  use_layer_norm: bool,
                  use_depth_to_space: bool):
    kwargs = dict(input_name='input_stream',
                  input_spatial_size=input_spatial_size,
                  input_features_size=input_features_size,
                  logits_output_name=('logits', util.Argument.WORLD),
                  mask_output_name=('masks', util.Argument.WORLD),
                  upscale_factor=upscale_factor,
                  kernel_size=kernel_size,
                  use_layer_norm=use_layer_norm,
                  use_depth_to_space=use_depth_to_space)
    if use_depth_to_space and kernel_size % upscale_factor != 0:
      with self.assertRaises(ValueError):
        _ = visual.Logits(**kwargs)
    else:
      component = visual.Logits(**kwargs)
      self._test_component(
          component, batch_size=2, unroll_len=3 if is_training else 1)

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

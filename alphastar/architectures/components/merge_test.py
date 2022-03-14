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

"""Tests for merge."""

from typing import Sequence

from absl.testing import absltest
from absl.testing import parameterized
from alphastar import types
from alphastar.architectures.components import merge
from alphastar.architectures.components import test_utils
from jax import test_util as jtu
import jax.numpy as jnp


class MergeTest(test_utils.ComponentTest):
  """Basic tests for the merge components."""

  @parameterized.product(
      is_training=[True, False],
      input_names=[['stream'], ['stream1', 'stream2']],
      stream_shape=[(1,), (3, 4), (2, 1, 4)])
  def test_SumMerge(self,
                    is_training: bool,
                    input_names: Sequence[types.StreamType],
                    stream_shape: Sequence[int]):
    component = merge.SumMerge(
        input_names=input_names,
        output_name='stream',
        stream_shape=stream_shape)
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_sizes_and_dtypes=[
          ({'stream': 4}, None),
          ({'stream': 4}, {'stream': jnp.int32}),
          ({'stream1': 3, 'stream2': None}, None),
          ({'stream1': 3, 'stream2': 4, 'stream3': 1}, None),
          ({'stream1': None, 'stream2': None, 'stream3': 3},
           {'stream1': jnp.bool_, 'stream2': jnp.int32}),],
      output_size=[1, 3],
      gating_type=list(merge.GatingType),
      use_layer_norm=[True, False])
  def test_VectorMerge(
      self,
      is_training: bool,
      input_sizes_and_dtypes,
      output_size: int,
      gating_type: merge.GatingType,
      use_layer_norm: bool = True):
    component = merge.VectorMerge(
        input_sizes=input_sizes_and_dtypes[0],
        output_name='stream',
        output_size=output_size,
        gating_type=gating_type,
        use_layer_norm=use_layer_norm,
        input_dtypes=input_sizes_and_dtypes[1])
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)

  @parameterized.product(
      is_training=[True, False],
      input_sizes_and_dtypes=[
          ({'stream': 4}, None),
          ({'stream': 4}, {'stream': jnp.int32}),
          ({'stream1': 3, 'stream2': None}, None),
          ({'stream1': 3, 'stream2': 4, 'stream3': 1}, None),
          ({'stream1': None, 'stream2': None, 'stream3': 3},
           {'stream1': jnp.bool_, 'stream2': jnp.int32}),],
      output_size=[1, 3],
      gating_type=list(merge.GatingType),
      use_layer_norm=[True, False])
  def test_UnitsMerge(
      self,
      is_training: bool,
      input_sizes_and_dtypes,
      output_size: int,
      gating_type: merge.GatingType,
      use_layer_norm: bool = True):
    input_spec, _ = test_utils.get_test_specs(is_training)
    component = merge.UnitsMerge(
        max_num_observed_units=input_spec['observation', 'raw_units'].shape[0],
        input_sizes=input_sizes_and_dtypes[0],
        output_name='stream',
        output_size=output_size,
        gating_type=gating_type,
        use_layer_norm=use_layer_norm,
        input_dtypes=input_sizes_and_dtypes[1])
    self._test_component(
        component, batch_size=2, unroll_len=3 if is_training else 1)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

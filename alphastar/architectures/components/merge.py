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

"""Components to merge streams."""

import enum
from typing import List, Mapping, Optional, Sequence

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import util
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp


class GatingType(enum.Enum):
  """Defines how the tensors are gated and aggregated in modules."""
  NONE = 'none'
  GLOBAL = 'global'
  POINTWISE = 'pointwise'


class SumMerge(modular.BatchedComponent):
  """Merge streams using a simple sum (faster than Merge, for large stream).

  Streams must have the same size.
  This module can merge any type of stream (vector, units or visual).
  """

  def __init__(self,
               input_names: Sequence[types.StreamType],
               output_name: types.StreamType,
               stream_shape: Sequence[int],
               name: Optional[str] = None):
    """Initializes SumMerge module.

    Args:
      input_names: The name of the inputs to sum. They must all have shape
        `stream_shape` and dtype float32.
      output_name: The name to give to the output of this module, of shape
        `stream_shape` and dtype float32.
      stream_shape: The shape of the inputs and outputs.
      name: The name of this component.
    """
    super().__init__(name=name)
    if not input_names:
      raise ValueError('input_names cannot be empty')
    self._input_names = input_names
    self._output_name = output_name
    self._stream_spec = specs.Array(shape=stream_shape, dtype=jnp.float32)

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        name: self._stream_spec for name in self._input_names})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({self._output_name: self._stream_spec})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    outputs = types.StreamDict({
        self._output_name: sum([inputs[name] for name in self._input_names])})
    return outputs, {}


class VectorMerge(modular.BatchedComponent):
  """Merge vector streams.

  Streams are first transformed through layer normalization, relu and linear
  layers, then summed, so they don't need to have the same size.
  Gating can also be used before the sum.

  If gating_type is not none, the sum is weighted using a softmax
  of the intermediate activations labelled above.
  """

  def __init__(self,
               input_sizes: Mapping[types.StreamType, Optional[int]],
               output_name: types.StreamType,
               output_size: int,
               gating_type: GatingType = GatingType.NONE,
               use_layer_norm: bool = True,
               input_dtypes: Optional[Mapping[types.StreamType,
                                              jnp.dtype]] = None,
               name: Optional[str] = None):
    """Initializes VectorMerge module.

    Args:
      input_sizes: A dictionary mapping input names to their size (a single
        integer for 1d inputs, or None for 0d inputs).
        If an input size is None, we assume it's ().
      output_name: The name to give to the output of this module, of shape
        [output_size] and dtype float32.
      output_size: The size of the output vector.
      gating_type: The type of gating mechanism to use.
      use_layer_norm: Whether to use layer normalization.
      input_dtypes: An optional dictionary with the dtypes of the inputs. If an
        input is missing from this dictionary, its dtype is assumed to be
        float32.
      name: The name of this component.
    """
    super().__init__(name=name)
    if not input_sizes:
      raise ValueError('input_names cannot be empty')
    self._input_sizes = input_sizes
    self._output_name = output_name
    self._output_size = output_size
    self._gating_type = gating_type
    self._use_layer_norm = use_layer_norm
    self._input_dtypes = dict(input_dtypes or {})
    missing_inputs = set(self._input_dtypes).difference(set(self._input_sizes))
    if missing_inputs:
      raise ValueError(f'Inputs {missing_inputs} are in input_dtypes but not '
                       'in input_sizes')
    for name in self._input_sizes:
      if name not in self._input_dtypes:
        self._input_dtypes[name] = jnp.float32

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    for name, size in self._input_sizes.items():
      if size is not None:
        spec[name] = specs.Array((size,), self._input_dtypes[name])
      else:
        spec[name] = specs.Array((), self._input_dtypes[name])
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict(
        {self._output_name: specs.Array((self._output_size,), jnp.float32)})

  def _compute_gate(
      self,
      inputs_to_gate: List[types.StreamDict],
      init_gate: List[types.StreamDict]
    ):
    w_init = hk.initializers.RandomNormal(stddev=0.005)
    b_init = hk.initializers.Constant(0.)
    if self._gating_type is GatingType.GLOBAL:
      gate_size = 1
    elif self._gating_type is GatingType.POINTWISE:
      gate_size = self._output_size
    else:
      raise ValueError(f'Gating type {self._gating_type} is not supported')
    if len(inputs_to_gate) == 2:
      # more efficient than the general version below
      gate = [hk.Linear(gate_size, w_init=w_init, b_init=b_init)(y)
              for y in init_gate]
      gate = sum(gate)
      sigmoid = jax.nn.sigmoid(gate)
      gate = [sigmoid, 1. - sigmoid]
    else:
      gate = [
          hk.Linear(
              len(inputs_to_gate) * gate_size, w_init=w_init, b_init=b_init)(y)
          for y in init_gate
      ]
      gate = sum(gate)
      gate = jnp.reshape(gate, [len(inputs_to_gate), gate_size])
      gate = jax.nn.softmax(gate, axis=0)
      gate = [gate[i] for i in range(gate.shape[0])]
    return gate

  def _encode(self, inputs: types.StreamDict):
    gate, outputs = [], []
    for name, size in self._input_sizes.items():
      feature = inputs[name]
      if size is None:
        feature = feature[jnp.newaxis]
      feature = util.astype(feature, jnp.float32)
      if self._use_layer_norm:
        feature = util.vector_layer_norm(feature)
      feature = jax.nn.relu(feature)
      gate.append(feature)
      outputs.append(hk.Linear(self._output_size)(feature))
    return gate, outputs

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    gate, outputs = self._encode(inputs)
    if len(outputs) == 1:
      # Special case of 1-D inputs that do not need any gating.
      output = outputs[0]
    elif self._gating_type is GatingType.NONE:
      output = sum(outputs)
    else:
      gate = self._compute_gate(outputs, gate)
      data = [g * d for g, d in zip(gate, outputs)]
      output = sum(data)
    outputs = types.StreamDict({self._output_name: output})
    return outputs, {}


class UnitsMerge(VectorMerge):
  """Merge units streams. Applies VectorMerge unit-wise."""

  def __init__(
      self,
      input_sizes: Mapping[types.StreamType, Optional[int]],
      max_num_observed_units: int,
      output_name: types.StreamType,
      output_size: int,
      gating_type: GatingType = GatingType.NONE,
      use_layer_norm: bool = True,
      input_dtypes: Optional[Mapping[types.StreamType, jnp.dtype]] = None,
      name: Optional[str] = None):
    """Initializes UnitsMerge module.

    Args:
      input_sizes: A dictionary mapping input names to their number of feature
        per unit (a single integer n for inputs of shape
        [max_num_observed_units, n], or None for inputs of shape
        [max_num_observed_units]).
      max_num_observed_units: The maximum number of oberved units,
        ie. obs_spec["raw_units"].shape[0].
      output_name: The name to give to the output of this module, of shape
        [output_size] and dtype float32.
      output_size: The size of the output vector.
      gating_type: The type of gating mechanism to use.
      use_layer_norm: Whether to use layer normalization.
      input_dtypes: An optional dictionary with the dtypes of the inputs. If an
        input is missing from this dictionary, its dtype is assumed to be
        float32.
      name: The name of this component.
    """
    super().__init__(input_sizes=input_sizes,
                     output_name=output_name,
                     output_size=output_size,
                     gating_type=gating_type,
                     use_layer_norm=use_layer_norm,
                     input_dtypes=input_dtypes,
                     name=name)
    self._max_num_observed_units = max_num_observed_units

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict()
    for name, size in self._input_sizes.items():
      dtype = self._input_dtypes[name]
      if size is not None:
        spec[name] = specs.Array((self._max_num_observed_units, size,), dtype)
      else:
        spec[name] = specs.Array((self._max_num_observed_units,), dtype)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array(
            (self._max_num_observed_units, self._output_size,), jnp.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    return jax.vmap(super()._forward)(inputs)


# Note that we have not implemented VisualMerge because we use SumMerge for
# visual streams, to optimize speed.

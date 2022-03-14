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

"""Component common to all architecture streams(vector, visual and units)."""

from typing import Iterable, Optional

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import util
from alphastar.commons import sample
import dm_env
from dm_env import specs
import jax.numpy as jnp


class ActionFromBehaviourFeatures(modular.BatchedComponent):
  """Copies the action from the behaviour features (for training)."""

  def __init__(self,
               argument_name: types.ArgumentName,
               max_action_value: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes ActionFromBehaviourFeatures module.

    Args:
      argument_name: The name of the action argument to use.
      max_action_value: An optional clipping of the max value for the action.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._argument_name = argument_name
    self._max_action_value = max_action_value

  @property
  def input_spec(self) -> types.SpecDict:
    """Generates the specifications for the input."""
    return types.SpecDict({
        ("behaviour_features", "action", self._argument_name): specs.Array(
            (), jnp.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    """Generates the specifications for the output."""
    return types.SpecDict({
        ("action", self._argument_name): specs.Array((), jnp.int32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs["behaviour_features", "action", self._argument_name]
    if self._max_action_value is not None:
      x = jnp.minimum(x, self._max_action_value)
    outputs = types.StreamDict({("action", self._argument_name): x})
    return outputs, {}


class Sample(modular.BatchedComponent):
  """Samples the logits into an action (for inference)."""

  def __init__(self,
               argument_name: types.ArgumentName,
               num_logits: int,
               sample_fn: sample.SampleFn,
               name: Optional[str] = None):
    """Initializes Sample module.

    Args:
      argument_name: The name of the action argument to use.
      num_logits: The size of the logits 1d vector.
      sample_fn: The function to sample the logits, taking a float32 1d logits
        vector as input and returning a int32 0d action.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._argument_name = argument_name
    self._num_logits = num_logits
    self._sample_fn = sample_fn

  @property
  def input_spec(self) -> types.SpecDict:
    """Generates the specifications for the input."""
    return types.SpecDict({("logits", self._argument_name): specs.Array(
        (self._num_logits,), jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    """Generates the specifications for the output."""
    return types.SpecDict({
        ("action", self._argument_name): specs.Array((), jnp.int32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    action = self._sample_fn(inputs["logits", self._argument_name])
    outputs = types.StreamDict({("action", self._argument_name): action})
    return outputs, {}


class ArgumentMasks(modular.BatchedComponent):
  """Compute the argument masks from the function argument."""

  def __init__(self,
               action_spec: types.ActionSpec,
               input_name: types.StreamType = ("action", "function"),
               output_name: types.StreamType = "argument_masks",
               name: Optional[str] = None):
    """Initializes ArgumentMasks object.

    Args:
      action_spec: The action spec.
      input_name: The name of the input to use, of shape [] and dtype int32.
      output_name: The prefix of the name to give to the output. There is one
        output per argument name in action_spec, each one of shape [] and
        dtype bool.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._action_spec = action_spec
    self._input_name = input_name
    self._output_name = output_name

  @property
  def input_spec(self) -> types.SpecDict:
    """Generates the specifications for the input."""
    return types.SpecDict({
        self._input_name: specs.Array((), jnp.int32),
        "step_type": specs.Array((), jnp.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    """Generates the specifications for the output."""
    return types.SpecDict({(self._output_name, arg): specs.Array((), jnp.bool_)
                           for arg in self._action_spec})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    function_arg = inputs[self._input_name]
    full_argument_masks = util.get_full_argument_masks(self._action_spec)
    first_step_mask = jnp.not_equal(
        inputs["step_type"], int(dm_env.StepType.FIRST))
    outputs = types.StreamDict({
        (self._output_name, arg): jnp.logical_and(
            full_argument_masks[arg][function_arg], first_step_mask)
        for arg in self._action_spec})
    return outputs, {}


class FeatureFromPrevState(modular.Component):
  """Copies a feature from the prev_state.

  During training, the feature must be in behaviour_features.
  """

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               is_training: bool,
               stream_shape: Iterable[int],
               stream_dtype: jnp.dtype = jnp.float32,
               name: Optional[str] = None):
    """Initializes FeatureFromPrevState module.

    Args:
      input_name: The name of the input to use, of shape `stream_shape` and
        dtype `stream_dtype`.
      output_name: The name to give to the output, of shape `stream_shape` and
        dtype `stream_dtype`.
      is_training: A boolean specifying whether this is training or inference.
        During inference, `behaviour_features` must be in the inputs to provide
        the values used by the behaviour model (or the recorded episode for
        offline learning).
      stream_shape: The shape of the input and output.
      stream_dtype: The dtype of the input and output.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._is_training = is_training
    self._stream_spec = specs.Array(stream_shape, stream_dtype)

  @property
  def input_spec(self) -> types.SpecDict:
    """Generates the specifications for the input."""
    spec = types.SpecDict()
    if self._is_training:
      spec["behaviour_features", self._input_name] = self._stream_spec
    return spec

  @property
  def prev_state_spec(self) -> types.SpecDict:
    """Generates the specifications for the previous state."""
    return types.SpecDict({self._input_name: self._stream_spec})

  @property
  def output_spec(self) -> types.SpecDict:
    """Generates the specifications for the output."""
    return types.SpecDict({self._output_name: self._stream_spec})

  @property
  def next_state_spec(self) -> types.SpecDict:
    """Generates the specifications for the next state."""
    return types.SpecDict()

  def _unroll(self,
              inputs: types.StreamDict,
              prev_state: types.StreamDict) -> modular.UnrollOutputType:
    outputs = types.StreamDict()
    prev_state_input = prev_state[self._input_name][jnp.newaxis]
    if self._is_training:
      behaviour_inputs = inputs.get("behaviour_features")[self._input_name]
      outputs = types.StreamDict({
          self._output_name: jnp.concatenate(
              [prev_state_input, behaviour_inputs[:-1]], axis=0)})
    else:
      outputs = types.StreamDict({self._output_name: prev_state_input})
    return outputs, types.StreamDict(), {}


class FeatureToNextState(modular.Component):
  """Copies a feature from the prev_state."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               overlap_len: int,
               stream_shape: Iterable[int],
               stream_dtype: jnp.dtype = jnp.float32,
               name: Optional[str] = None):
    """Initializes FeatureToNextState module.

    Args:
      input_name: The name of the input to use, of shape `stream_shape` and
        dtype `stream_dtype`.
      output_name: The name to give to the output, of shape `stream_shape` and
        dtype `stream_dtype`.
      overlap_len: The number of timesteps overlapping between two trajectories.
        During training, the timestep passed to the next rollout is the one
        immediately before the next rollout, which is not the last one if
        overlap_len is not 0. Note that during inference, overlap_len must be 0
        since the rollout length is 1.
      stream_shape: The shape of the input and output.
      stream_dtype: The dtype of the input and output.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._stream_spec = specs.Array(stream_shape, stream_dtype)
    self._overlap_len = overlap_len
    if overlap_len < 0:
      raise ValueError(f"overlap_len must be non-negative, not {overlap_len}.")

  @property
  def input_spec(self) -> types.SpecDict:
    """Generates the specifications for the input."""
    return types.SpecDict({self._input_name: self._stream_spec})

  @property
  def prev_state_spec(self) -> types.SpecDict:
    """Generates the specifications for the previous state."""
    return types.SpecDict()

  @property
  def output_spec(self) -> types.SpecDict:
    """Generates the specifications for the output."""
    return types.SpecDict()

  @property
  def next_state_spec(self) -> types.SpecDict:
    """Generates the specifications for the next state."""
    return types.SpecDict({self._output_name: self._stream_spec})

  def _unroll(self,
              inputs: types.StreamDict,
              prev_state: types.StreamDict) -> modular.UnrollOutputType:
    x = inputs[self._input_name]
    unroll_len = x.shape[0]
    if self._overlap_len >= unroll_len:
      raise ValueError(f"overlap_len ({self._overlap_len}) is larger than the "
                       f"unroll length ({x.shape[0]}).")
    effective_sequence_length = unroll_len - self._overlap_len - 1
    next_state = types.StreamDict({
        self._output_name: x[effective_sequence_length]})
    return types.StreamDict(), next_state, {}

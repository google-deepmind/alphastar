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

"""Abstract classes for the AlphaStar components.

Components are the basic building blocks of the alphastar architecture.
They are typically assembled in a sequence, using `SequentialComponent`.
Components implement an `unroll` function, which takes inputs from the current
rollout, and prev_state from the previous rollout. It returns outputs for the
current rollout, and an optional state to be passed to the next rollout.
Inputs and outputs are types.StructDict objects (nested dictionary with
chex.Array as leaves).
The `unroll` function takes two arguments:
  `inputs`: All the tensors corresponding to the current rollout. In a
    sequential component, this contains the observations for this rollout, and
    the outputs of all the previous components in the sequence. Since each input
    is a rollout, all tensors have a first dimension with size `unroll_len`.
  `prev_state`: The set of `state` tensors coming from the previous rollout
    (eg. LSTM state). They only contain the state of the timestep immediately
    before the first timestep of the current rollout (if there is no overlap
    between rollouts, that's the last step of the previous rollout), so unlike
    inputs, they do not have a first dimension with size `unroll_len`.
    Note that this contains the states produced by all the components in the
    previous rollout, including components which appear later in the sequence.
And the `unroll` function returns three arguments:
  `outputs`: The set of tensors this component computes for this rollout, they
    will be passed to the next components in the sequence and returned by the
    SequentialComponent `unroll` function.
  `next_state`: The set of tensors to pass to the next rollout components. All
    the states passed to the next rollouts are aggregated into a single
    types.StreamDict object before being passed to the next rollout.
  `log`: A log_utils.Log object (dictionary of dictionary) containing logs. The
    first level contains the name of the logs, the second contains the type of
    reduce function to apply to the logs (see log_utils).

To implement this function, a subclass must implement `_unroll`, since the
`unroll` function performs additional types checks.

By default, the `prev_state` of the very first rollout contains only zeros. It
can be changed by subclassing the `_initial_state` function.

Components have input and output static typing. This is enforced through 4
properties which must be implemented for each module, returning a types.SpecDict
object (nested dictionaries with specs.Array as leaves):
  `input_spec`: The spec of the `inputs` argument of the `unroll` function. This
    must be a subset of what is passed as `inputs`, and the shapes and data
    types must match. Only inputs specified in the `input_spec` will be visible
    in the `_unroll` function.
  `prev_state_spec`: The spec of the `prev_state` argument of the `unroll`
    function. This must be a subset of what is passed as `prev_state`, and the
    shapes and data types must match. Only inputs specified in the
    `prev_state_spec` will be visible in the `_unroll` function.
  `output_spec`: The spec of the `output` returned by the `_unroll` function.
    The leaves must match exactly, and the shapes and types must match as well.
  `next_state_spec`: The spec of the `next_state` returned by the `_unroll`
    function. The leaves must match exactly, and the shapes and types must match
    as well. Note that since `prev_state` contains the states passed by all the
    components from the previous rollout, `next_state_spec` has no reason to
    match `prev_state_spec`. For instance, one component could use the state
    passed by another component, or pass a state without using it (so that other
    components use it).

For convenience, a `BatchedComponent` class can be used instead of the base
`Component`. Instead of `_unroll`, the subclasses must implement a `_forward`
function. The difference is that the `_forward` function operates on a single
timestep instead of a rollout. This has two consequences:
  * These components do not use `prev_state` and `next_state`.
  * The tensors in the `inputs` argument (and returned in `outputs`) do not have
    the first dimension with size `rollout_len`.
"""

import abc
from typing import Callable, Optional, Tuple

from alphastar import types
from alphastar.commons import log_utils
import chex
import haiku as hk
import jax
import jax.numpy as jnp


ForwardOutputType = Tuple[types.StreamDict, log_utils.Log]
UnrollOutputType = Tuple[types.StreamDict, types.StreamDict, log_utils.Log]


class Component(abc.ABC):
  """Basic AlphaStar component (see module docstring)."""

  def __init__(self, name: Optional[str] = None):
    self._name = name or "Component"

  @property
  def name(self) -> str:
    return self._name

  @property
  @abc.abstractmethod
  def input_spec(self) -> types.SpecDict:
    """Returns the spec of the input of this module."""

  @property
  @abc.abstractmethod
  def prev_state_spec(self) -> types.SpecDict:
    """Returns the spec of the prev_state of this module."""

  @property
  @abc.abstractmethod
  def output_spec(self) -> types.SpecDict:
    """Returns the spec of the output of this module."""

  @property
  @abc.abstractmethod
  def next_state_spec(self) -> types.SpecDict:
    """Returns the spec of the next_state of this module."""

  def _initial_state(self) -> types.StreamDict:
    """Initial state of the component.

    If this component returns a next_state in its unroll function, then
    this function provides the initial state. By default, we use zeros,
    but it can be overridden for custom initial state.

    Subclasses should override this function instead of `initial_state`, which
    adds additional checks.

    Returns:
      A Dict containing the state before the first step.
    """
    return jax.tree_map(
        lambda spec: jnp.zeros(shape=spec.shape, dtype=spec.dtype),
        self.next_state_spec)

  def initial_state(self) -> types.StreamDict:
    """Initial state of the component.

    If this component returns a next_state in its unroll function, then
    this function provides the initial state. By default, we use zeros,
    but it can be overridden for custom initial state.

    Returns:
      A Dict containing the state before the first step.
    """
    initial_state = self._initial_state()
    self.next_state_spec.validate(
        initial_state, error_prefix=f"{self.name} initial_state")
    return initial_state

  @abc.abstractmethod
  def _unroll(self,
              inputs: types.StreamDict,
              prev_state: types.StreamDict) -> UnrollOutputType:
    """Computes the output of the module over unroll_len timesteps.

    Call with a unroll_len=1 for a single step.

    Subclasses should override this function instead of `unroll`, which
    adds additional checks.

    Args:
      inputs: A StreamDict containing [unroll_len, ...] tensors.
      prev_state: A StreamDict containing [...] tensors, containing the
        next_state of the last timestep of the previous unroll.

    Returns:
      outputs: A StreamDict containing [unroll_len, ...] tensors.
      next_state: A dict containing [...] tensors representing the
        state to be passed as the first state of the next rollout.
        If overlap_len is 0, this is the last state of this rollout.
        More generally, this is the (unroll_len - overlap_len)-th state.
      logs: A dict containing [unroll_len] tensors to be logged.
    """

  def unroll(self,
             inputs: types.StreamDict,
             prev_state: types.StreamDict) -> UnrollOutputType:
    """Computes the output of the module over unroll_len timesteps.

    Call with a unroll_len=1 for a single step.

    Args:
      inputs: A StreamDict containing [unroll_len, ...] tensors.
      prev_state: A StreamDict containing [...] tensors, containing the
        next_state of the last timestep of the previous unroll.

    Returns:
      outputs: A StreamDict containing [unroll_len, ...] tensors.
      next_state: A dict containing [...] tensors representing the
        state to be passed as the first state of the next rollout.
        If overlap_len is 0, this is the last state of this rollout.
        More generally, this is the (unroll_len - overlap_len)-th state.
      logs: A dict containing [unroll_len] tensors to be logged.
    """
    if inputs:
      try:
        chex.assert_equal_shape(jax.tree_leaves(inputs), dims=0)
      except AssertionError as e:
        raise AssertionError(f"{self.name}: {e}") from e
    self.input_spec.validate(inputs,
                             num_leading_dims_to_ignore=1,
                             error_prefix=f"{self.name} inputs")
    self.prev_state_spec.validate(prev_state,
                                  error_prefix=f"{self.name} prev_state")
    # We hide inputs not specified in input_spec to prevent accidental use.
    inputs = inputs.filter(self.input_spec)
    prev_state = prev_state.filter(self.prev_state_spec)
    with hk.experimental.name_scope(self.name):
      outputs, next_state, logs = self._unroll(inputs, prev_state)
    self.output_spec.validate(outputs,
                              num_leading_dims_to_ignore=1,
                              error_prefix=f"{self.name} outputs")
    self.next_state_spec.validate(next_state,
                                  error_prefix=f"{self.name} next_state")
    return outputs, next_state, logs

ArchitectureBuilder = Callable[[types.InputSpec, types.ActionSpec, bool],
                               Component]


class BatchedComponent(Component):
  """A Component which is not using the unroll dimension.

  This is a helper module to write simpler components.
  Such a component computes a function _forward such that
  unroll(x)[t] = _forward(x[t]) where t=0..unroll_len-1.

  Such a module must be stateless.
  """

  @property
  def prev_state_spec(self) -> types.SpecDict:
    return types.SpecDict()

  @property
  def next_state_spec(self) -> types.SpecDict:
    return types.SpecDict()

  def _unroll(self,
              inputs: types.StreamDict,
              prev_state: types.StreamDict) -> UnrollOutputType:
    del prev_state
    outputs, logs = jax.vmap(self._forward)(inputs)
    return outputs, types.StreamDict(), logs

  @abc.abstractmethod
  def _forward(self, inputs: types.StreamDict) -> ForwardOutputType:
    """Computes the output of this module for each timestep.

    Args:
      inputs: A StreamDict containing [...] tensors.

    Returns:
      outputs: A StreamDict containing [...] tensors.
      logs: A dict containing [...] tensors to be logged.
    """


class SequentialComponent(Component):
  """A component made of a sequence of components.

  Components are added with the `append` method and form a sequence, such
  that the inputs of the i-th component contain of the union of the inputs
  of the SequentialComponent, and the output of the first to the (i-1)-th
  components (if two inputs have the same name, the most recent hides the older
  ones).

  On the other hand, the states of the previous rollout (`prev_state`) are
  agnostic to the order of the components since the states of all of the
  components are aggregated before being passed from the previous rollout (and
  two next states cannot have the same name).

  For instance, given two components `component_a` and `component_b`, and inputs
  `rollout1` and `rollout2`:
  ```
  sequence = SequentialComponent()
  sequence.append(component_a)
  sequence.append(component_b)
  state0 = sequence.initial_state()
  outputs1, state1, log1 = sequence.unroll(rollout1, state0)
  outputs2, state2, log2 = sequence.unroll(rollout2, state1)
  ```
  is equivalent to:
  ```
  def merge(a, b):
    merged = a.copy()
    for k, v in b.items():
      merged[k] = v
    return merged

  state0a = component_a.initial_state()
  state0b = component_b.initial_state()
  state0 = merge(state0a, state0b)

  outputs1a, state1a, log1a = component_a.unroll(rollout1, state0)
  inputs1b = merged(rollout1, outputs1a)
  outputs1b, state1b, log1b = component_b.unroll(inputs1b, state0)
  outputs1 = merge(outputs1a, outputs1b)
  state1 = merge(state1a, state1b)
  log1 = merge(log1a, log1b)

  outputs2a, state2a, log2a = component_a.unroll(rollout2, state1)
  inputs2b = merged(rollout2, outputs2a)
  outputs2b, state2b, log2b = component_b.unroll(inputs2b, state1)
  outputs2 = merge(outputs2a, outputs2b)
  state2 = merge(state2a, state2b)
  log2 = merge(log2a, log2b)
  ```
  """

  def __init__(self,
               name: Optional[str] = None):
    super().__init__(name=name)
    self._components = []
    self._input_spec = types.SpecDict()
    self._output_spec = types.SpecDict()
    self._prev_state_spec = types.SpecDict()
    self._next_state_spec = types.SpecDict()
    self._input_spec_src = types.StrDict()
    self._output_spec_src = types.StrDict()
    self._prev_state_spec_src = types.StrDict()
    self._next_state_spec_src = types.StrDict()

  def append(self, component: Component) -> None:
    """Adds a component to the sequence.

    If performs a spec check on the inputs and outputs:
      * For each input spec `x` in `component.input_spec`, we use the following
        process:
          1) If the name of `x` is in `self.input_spec` or in the output spec of
            any previously appended component, then we check that the shape and
            data type match. If not, a ValueError is raised.
          2) If the name of `x` does is not in the output spec of any previously
            added component, then this input must be specified as an input of
            the `SequentialComponent`, so we add it to `self.input_spec`.
      * For each output spec `x` in `component.output_spec`, we use the
        following process:
          1) If the name appears in `self.input_spec` or in the output spec of
            any previously added component, then we check that the shape and
            data type match. If not, a ValueError is raised.
          2) `x` is added to `self.output_spec`.
      * For each previous state input `x` in `component.prev_state_spec`, we
        check that the spec matches any previous state with the same name
        specified by previously added component. If not, a ValueError is raised.
        We then add `x` to `self.prev_state_spec`.
      * For each next state output `x` in `component.next_state_spec`, we check
        that no previously added component specified a next state with the same
        name. If not, a ValueError is raised. We then add `x` to
        `self.next_state_spec`.

    Args:
      component: The `Component` to append at the end of the sequence of
        components contained by this module.

    Raises:
      ValueError: If any of the spec check specified above fails.
    """
    self._components.append(component)
    # input spec:
    for spec_name, spec in component.input_spec.items():
      if spec_name in self._output_spec:
        if spec != self._output_spec[spec_name]:
          raise ValueError(
              f"Input {spec_name} matches the name of an output of a "
              f"previously added component ({self._output_spec_src[spec_name]})"
              ", but their specs do not match."
              f"Input spec: {spec}. "
              f"Previous output spec: {self._output_spec[spec_name]}.")
      elif spec_name in self._input_spec:
        if spec != self._input_spec[spec_name]:
          raise ValueError(
              f"Input {spec_name} matches the name of an input of a "
              f"previously added component ({self._input_spec_src[spec_name]})"
              ", but their specs do not match."
              f"Input spec: {spec}. "
              f"Previous input spec: {self._input_spec[spec_name]}.")
      else:
        self._input_spec_src[spec_name] = component.name
        self._input_spec[spec_name] = spec
    # prev_state spec:
    for spec_name, spec in component.prev_state_spec.items():
      if spec_name in self._prev_state_spec:
        if spec != self._prev_state_spec[spec_name]:
          raise ValueError(
              f"Previous state {spec_name} matches the name of a previous "
              "state used by a previously added component "
              f"({self._prev_state_spec_src[spec_name]})"
              ", but their specs do not match."
              f"prev_state spec: {spec}. "
              f"Previous prev_state spec: {self._prev_state_spec[spec_name]}.")
      else:
        self._prev_state_spec_src[spec_name] = component.name
        self._prev_state_spec[spec_name] = spec
    # output spec
    for spec_name, spec in component.output_spec.items():
      if spec_name in self._output_spec:
        if spec != self._output_spec[spec_name]:
          raise ValueError(
              f"Output {spec_name} matches the name of an output of a "
              f"previously added component ({self._output_spec_src[spec_name]})"
              ", but their specs do not match."
              f"Output spec: {spec}. "
              f"Previous output spec: {self._output_spec[spec_name]}.")
      elif spec_name in self._input_spec:
        if spec != self._input_spec[spec_name]:
          raise ValueError(
              f"Output {spec_name} matches the name of an input of a "
              f"previously added component ({self._input_spec_src[spec_name]})"
              ", but their specs do not match."
              f"Output spec: {spec}. "
              f"Previous input spec: {self._input_spec[spec_name]}.")
      self._output_spec_src[spec_name] = component.name
      self._output_spec[spec_name] = spec
    # next_state spec:
    for spec_name, spec in component.next_state_spec.items():
      if spec_name in self._next_state_spec:
        raise ValueError(
            f"Next state {spec_name} is used by two components: "
            f"{self._next_state_spec_src[spec_name]} and {component.name}.")
      else:
        self._next_state_spec_src[spec_name] = component.name
        self._next_state_spec[spec_name] = spec

  @property
  def input_spec(self) -> types.SpecDict:
    return self._input_spec

  @property
  def prev_state_spec(self) -> types.SpecDict:
    return self._prev_state_spec

  @property
  def output_spec(self) -> types.SpecDict:
    return self._output_spec

  @property
  def next_state_spec(self) -> types.SpecDict:
    return self._next_state_spec

  def _initial_state(self) -> types.StreamDict:
    initial_state = types.StreamDict()
    initial_state_src = types.NestedDict[str]()
    for component in self._components:
      for state_name, state in component.initial_state().items():
        if state_name in initial_state:
          # This should never happen, since we already check next_state.
          raise ValueError(
              f"Initial state {state_name} is defined by {component.name} but "
              f"was already defined by {initial_state_src[state_name]}.")
        else:
          initial_state_src[state_name] = component.name
          initial_state[state_name] = state
    # Make sure that initial_state matches next_state_spec:
    self.next_state_spec.validate(initial_state,
                                  exact_match=True,
                                  error_prefix=f"{self.name} Initial state")
    return initial_state

  def _unroll(self,
              inputs: types.StreamDict,
              prev_state: types.StreamDict) -> UnrollOutputType:
    inputs = inputs.copy()
    outputs, next_state, logs = types.StreamDict(), types.StreamDict(), {}
    for component in self._components:
      comp_outputs, comp_next_state, comp_logs = component.unroll(
          inputs, prev_state)
      inputs.update(comp_outputs)
      outputs.update(comp_outputs)
      next_state.update(comp_next_state)
      for log_name, log in comp_logs.items():
        logs[f"[{component.name}] {log_name}"] = log
    return outputs, next_state, logs

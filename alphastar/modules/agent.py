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

"""The AlphaStar agent interface."""

import abc
import functools
from typing import Any, Optional

from absl import logging
from alphastar import types
from alphastar.architectures import modular
from alphastar.unplugged.data import util as data_util
import haiku as hk
import jax


class Agent(abc.ABC):
  """An agent interface used for AlphaStar."""

  @abc.abstractmethod
  def initial_state(self, rng_key, batch_size: int):
    """Initial internal state of the agent."""

  @abc.abstractmethod
  def init(self, rng_key, inputs, prev_state):
    """Get the initial parameters."""

  @abc.abstractmethod
  def apply(self, params, rng_key, inputs, prev_state):
    """Forward pass."""

  @property
  @abc.abstractmethod
  def input_spec(self):
    """The spec of the input of the agent."""

  @property
  @abc.abstractmethod
  def state_spec(self):
    """The spec of the prev_state and next_state of the agent."""

  @property
  @abc.abstractmethod
  def output_spec(self):
    """The spec of the output of the agent."""

  @abc.abstractmethod
  def warmup(self, obs: types.StreamDict) -> hk.Params:
    """Warmup the agent haiku modules."""


class AlphaStarAgent(Agent):
  """AlphaStar agent."""

  def __init__(
      self,
      component: modular.Component,
      jit_agent_functions: bool = True,
      jit_device: Optional[Any] = None,
      jit_backend: Optional[str] = None):
    """Initializes an AlphaStar agent.

    Args:
      component : Stateful architecture of the agent as a component.
      jit_agent_functions : Whether to JIT compile agent reset and step
        functions. Usually, do not set it to False as this may trigger
        recompilation of some modules such as hk.scan. If you want un-jitted
        agent functions, set it to False and turn off the recompilation check
        simultaneously.
      jit_device: Device if for JIT.
      jit_backend: Backend used for JIT (cou, tpu etc.)
    """
    self._component = component
    self._component.prev_state_spec.validate(
        self._component.next_state_spec,
        error_prefix='Agent next_state must contain prev state')
    if jit_agent_functions:
      self._agent_function_wrapper = functools.partial(
          jax.jit, device=jit_device, backend=jit_backend)
    else:
      self._agent_function_wrapper = lambda fun: fun

    @hk.transform
    def initial_state_fun(batch_size: int) -> types.StreamDict:
      return jax.vmap(component.initial_state, axis_size=batch_size)()
    self._initial_state_fun = initial_state_fun

    @hk.transform
    def unroll_fun(inputs: types.StreamDict,
                   prev_state: types.StreamDict) -> modular.UnrollOutputType:
      return jax.vmap(component.unroll)(inputs, prev_state)
    self._unroll_fun = unroll_fun

  def initial_state(self,
                    rng_key: jax.random.KeyArray,
                    batch_size: int) -> types.StreamDict:
    """Sets initial state for the agent."""
    init_params = self._initial_state_fun.init(rng_key, batch_size)
    return self._initial_state_fun.apply(init_params, rng_key, batch_size)

  def init(self,
           rng_key: jax.random.KeyArray,
           inputs: types.StreamDict,
           prev_state: types.StreamDict) -> hk.Params:
    """Returns the initial parameters for the agent."""
    inputs = inputs.filter(self.input_spec)
    unroll_init = self._agent_function_wrapper(self._unroll_fun.init)
    return unroll_init(rng_key, inputs, prev_state)

  def apply(self,
            params: hk.Params,
            rng_key: jax.random.KeyArray,
            inputs: types.StreamDict,
            prev_state: types.StreamDict) -> modular.UnrollOutputType:
    """Performs forward step of an agent."""
    inputs = inputs.filter(self.input_spec)
    unroll_apply = self._agent_function_wrapper(self._unroll_fun.apply)
    return unroll_apply(params, rng_key, inputs, prev_state)

  @property
  def input_spec(self) -> types.SpecDict:
    return self._component.input_spec

  @property
  def state_spec(self) -> types.SpecDict:
    return self._component.prev_state_spec

  @property
  def output_spec(self) -> types.SpecDict:
    return self._component.output_spec

  def warmup(self,
             batch_size: int = 1,
             unroll_len: int = 1) -> hk.Params:
    """Warms up an agent by doing a forward pass."""
    obs = data_util.get_dummy_observation(
        self.input_spec, batch_size=batch_size, unroll_len=unroll_len)
    rng = jax.random.PRNGKey(0)
    initial_state_key, params_key, key = jax.random.split(rng, 3)
    state = self.initial_state(initial_state_key, batch_size=batch_size)
    logging.info('Warming up the agent.')
    logging.info('Inputs: %s', jax.tree_map(lambda x: (x.shape, x.dtype), obs))
    logging.info('State: %s', jax.tree_map(lambda x: (x.shape, x.dtype), state))
    params = self.init(params_key, obs, state)
    self.apply(params, key, obs, state)
    return params

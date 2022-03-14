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

"""Utility functions for components."""

from typing import MutableMapping, Optional

from alphastar import types
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
from pysc2.lib import actions as sc2_actions

REPEATED_FUNCTION_TYPES = [sc2_actions.raw_cmd]


def astype(x: chex.Array, dtype: jnp.dtype) -> chex.Array:
  """Cast x if necessary."""
  if x.dtype != dtype:
    return x.astype(dtype)
  else:
    return x


def get_function_list(action_spec: types.ActionSpec
                      ) -> sc2_actions.Functions:
  """Get the list of Function available given this action_spec."""
  num_functions = action_spec['function'].maximum + 1
  if num_functions > len(sc2_actions.RAW_FUNCTIONS):
    raise ValueError(
        f'action_spec contains {num_functions}, which is larger than the '
        f'actual number of functions in sc2 {len(sc2_actions.RAW_FUNCTIONS)}.')
  normal_functions = [
      f for f in sc2_actions.RAW_FUNCTIONS if f.id < num_functions]
  functions = sc2_actions.Functions(normal_functions)
  # Sanity check:
  assert len(functions) == num_functions
  return functions


def get_full_argument_masks(action_spec: types.ActionSpec
                            ) -> MutableMapping[str, chex.Array]:
  """Get the (static) full argument masks.

  For each argument arg, full_argument_mask[arg][function] is the
  mask for the argument given the function argument.

  Args:
    action_spec: The action specification.

  Returns:
    A dict containing a jnp.ndarray of size (num_function,) for each
      action, specifying if the argument is used for each function argument.
  """
  function_list = get_function_list(action_spec)
  full_argument_masks = dict(
      {k: np.zeros((len(function_list),), dtype=bool) for k in action_spec},
      # Function and delay are never masked:
      function=np.ones((len(function_list),), dtype=bool),
      delay=np.ones((len(function_list),), dtype=bool))
  for func in function_list:
    for argument in func.args:
      full_argument_masks[argument.name][func.id] = True
      if func.function_type in REPEATED_FUNCTION_TYPES:
        # repeat is used iff the function type is in REPEATED_FUNCTION_TYPES
        full_argument_masks['repeat'][func.id] = True
  return jax.tree_map(jnp.asarray, full_argument_masks)


def vector_layer_norm(x: chex.Array) -> chex.Array:
  if x.ndim < 1:
    raise ValueError(
        f'Input must have at least one dimension (shape: {x.shape}).')
  layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return astype(layer_norm(astype(x, jnp.float32)), x.dtype)


def units_layer_norm(x: chex.Array) -> chex.Array:
  if x.ndim < 2:
    raise ValueError(
        f'Input must have at least two dimension (shape: {x.shape}).')
  layer_norm = hk.LayerNorm(axis=-1, create_scale=True, create_offset=True)
  return astype(layer_norm(astype(x, jnp.float32)), x.dtype)


def visual_layer_norm(x: chex.Array) -> chex.Array:
  if x.ndim < 3:
    raise ValueError(
        f'Input must have at least three dimension (shape: {x.shape}).')
  layer_norm = hk.LayerNorm(
      axis=[-3, -2, -1], create_scale=True, create_offset=True)
  return astype(layer_norm(astype(x, jnp.float32)), x.dtype)


class VectorResblock(hk.Module):
  """Fully connected residual block."""

  def __init__(self,
               num_layers: int = 2,
               hidden_size: Optional[int] = None,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes VectorResblock module.

    Args:
      num_layers: Number of layers in the residual block.
      hidden_size: Size of the activation vector in the residual block.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._use_layer_norm = use_layer_norm

  def __call__(self, x: chex.Array) -> chex.Array:
    chex.assert_rank(x, 1)
    chex.assert_type(x, jnp.float32)
    shortcut = x
    input_size = x.shape[-1]
    for i in range(self._num_layers):
      if i < self._num_layers - 1:
        output_size = self._hidden_size or input_size
        w_init, b_init = None, None
      else:
        output_size = input_size
        w_init = hk.initializers.RandomNormal(stddev=0.005)
        b_init = hk.initializers.Constant(0.)
      if self._use_layer_norm:
        x = vector_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Linear(output_size=output_size, w_init=w_init, b_init=b_init)(x)
    return x + shortcut


class UnitsResblock(VectorResblock):
  """Fully connected residual block, unit-wise."""

  def __call__(self, x: chex.Array) -> chex.Array:
    chex.assert_rank(x, 2)
    chex.assert_type(x, jnp.float32)
    return jax.vmap(super().__call__)(x)


class VisualResblock(hk.Module):
  """Convolutional (2d) residual block."""

  def __init__(self,
               kernel_size: int,
               num_layers: int = 2,
               hidden_size: Optional[int] = None,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes VisualResblock module.

    Args:
      kernel_size: The size of the convolution kernel.
      num_layers: Number of layers in the residual block.
      hidden_size: Size of the activation vector in the residual block.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._kernel_size = kernel_size
    self._num_layers = num_layers
    self._hidden_size = hidden_size
    self._use_layer_norm = use_layer_norm

  def __call__(self, x: chex.Array) -> chex.Array:
    chex.assert_rank(x, 3)
    chex.assert_type(x, jnp.float32)
    shortcut = x
    input_size = x.shape[-1]
    for i in range(self._num_layers):
      if i < self._num_layers - 1:
        output_size = self._hidden_size or input_size
        w_init, b_init = None, None
      else:
        output_size = input_size
        w_init = hk.initializers.RandomNormal(stddev=0.005)
        b_init = hk.initializers.Constant(0.)
      if self._use_layer_norm:
        x = visual_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Conv2D(output_channels=output_size,
                    kernel_shape=self._kernel_size,
                    w_init=w_init,
                    b_init=b_init)(x)
    return x + shortcut

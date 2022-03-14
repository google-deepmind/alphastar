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

"""Base class for Alphastar losses."""

import abc
from typing import Callable
from typing import Mapping
from typing import Optional
from typing import Tuple

from alphastar import types
from alphastar.commons import log_utils
import chex
import jax
from jax import numpy as jnp

LossOutputType = Tuple[chex.Array, log_utils.Log]


class Loss(abc.ABC):
  """Basic AlphaStar loss function."""

  def __init__(self, name: Optional[str] = "Loss"):
    self._name = name or "Loss"

  @property
  def name(self) -> str:
    return self._name

  @property
  @abc.abstractmethod
  def input_spec(self) -> types.SpecDict:
    """Returns the spec of the input of this loss."""

  @abc.abstractmethod
  def _loss(self, inputs: types.StreamDict) -> LossOutputType:
    """Computes the output of the loss over unroll_len timesteps.

    Args:
      inputs: A StreamDict containing [unroll_len, ...] tensors.

    Returns:
      loss: A float tensor of shape [unroll_len] containing the loss per
        timestep.
      logs: Per-timestep logs (shape [unroll_len]). This is a dict containing,
        for each entry, a dict with reduce functions as keys.
    """

  def loss(self, inputs: types.StreamDict) -> LossOutputType:
    """Computes loss on unbatched unrolled inputs. See _loss too.

    Inputs to this function are expected to be of the shape unroll_len and
    are not expected to be batched further. For batched version of the loss,
    check batched_loss.

    Args:
      inputs: Inputs used for loss computation, a StreamDict with elements of
        shape [unroll_len, ...].

    Returns:
      loss: A float tensor of shape [unroll_len] containing the loss per
        timestep.
      logs: Per-timestep logs (shape [unroll_len]). This is a dict containing,
        for each entry, a dict with reduce functions as keys.

    Raises:
      ValueError: when not all tree leaves are of the shape of unroll length.
    """
    if not inputs:
      raise ValueError("Losses must have at least one input.")
    unroll_len = jax.tree_leaves(inputs)[0].shape[0]
    try:
      chex.assert_tree_shape_prefix(inputs, (unroll_len,))
    except AssertionError as e:
      raise ValueError(
          f"All inputs should have same size as unroll length ({unroll_len}) "
          f"-- {self.name}: {e}") from e
    self.input_spec.validate(inputs,
                             num_leading_dims_to_ignore=1,
                             error_prefix=f"{self.name} inputs")
    # We hide inputs not specified in input_spec to prevent accidental use.
    inputs = inputs.filter(self.input_spec)
    outputs, logs = self._loss(inputs)
    chex.assert_shape(outputs, [unroll_len])
    chex.assert_type(outputs, jnp.float32)
    for k, v in logs.items():
      if not isinstance(v, Mapping):
        raise ValueError("Logs must be depth 2 nested dicts, "
                         f"but log {k} has type {type(v)}.")
      for inner_k, inner_v in v.items():
        if not isinstance(inner_v, chex.Array):
          raise ValueError("Logs must be depth 2 nested dicts, "
                           f"but log {k}/{inner_k} has type {type(inner_v)}.")
        if inner_v.shape != (unroll_len,):
          raise ValueError(
              f"Logs must have shape [unroll_len] ({unroll_len},), "
              f" but {k}/{inner_k} has shape {inner_v.shape}.")
    return outputs, logs

  def batched_loss(self, inputs: types.StreamDict) -> LossOutputType:
    """Computes loss on inputs with a batch dimension.

    Unless the loss explicitely needs access to the batch dimension, use
    _loss instead, which applies a vmap to hide the batch dimension.

    Args:
      inputs: Inputs used for loss computation, a StreamDict with elements of
        shape [batch_size, unroll_len, ...].

    Returns:
      loss: A float tensor of shape [batch_size, unroll_len] containing the loss
        per timestep.
      logs: Per-timestep logs (shape [batch_size, unroll_len]). This is a dict
        containing, for each entry, a dict with reduce functions as keys.

    Raises:
      ValueError: when not all tree leaves are of the shape of
        (batch size, unroll length).
    """
    batch_size, unroll_len = jax.tree_leaves(inputs)[0].shape[:2]
    try:
      chex.assert_tree_shape_prefix(inputs, (batch_size, unroll_len))
    except AssertionError as e:
      raise ValueError(
          "All inputs should have same size as (batch_size, unroll length) "
          f"({batch_size}, {unroll_len}) "
          f"-- {self.name}: {e}") from e

    outputs, logs = jax.vmap(self.loss)(inputs)
    chex.assert_shape(outputs, [batch_size, unroll_len])
    return outputs, logs

LossBuilder = Callable[[types.ActionSpec], Loss]

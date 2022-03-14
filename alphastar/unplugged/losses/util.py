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

"""Utility functions for the losses."""

from typing import Dict

from alphastar import types
from alphastar.commons import log_utils
import chex
import dm_env
import jax
import jax.numpy as jnp
import numpy as np


def get_global_loss_masks(step_type: chex.Array,
                          argument_masks: types.StreamDict,
                          num_first_steps_to_ignore: int,
                          num_last_steps_to_ignore: int,
                          ) -> types.StreamDict:
  """Gets the global loss masks (for each argument).

  The mask is 0 if at least one of these conditions is true:
  * The argument is masked
  OR
  * The step is an episode terminal state (since the action associated with
    this observation is a dummy, and the next state will be always an initial
    state).
  OR
  * The step is one of the first `num_first_steps_to_ignore` steps of the
    rollout (this can be used for burn-in).
  OR
  * The step is one of the first `num_last_steps_to_ignore` steps of the
    rollout (this can be used for bootstrapping).

  Args:
    step_type: Step type.
    argument_masks: A StreamDict containing the argument masks, unit_tags can be
      a single scalar mask (instad of one per unit_tags argument).
    num_first_steps_to_ignore: The number of steps to mask at the beginning of
      every trajectory.
    num_last_steps_to_ignore: The number of steps to mask at the end of
      every trajectory.

  Returns:
    A StreamDict containing the loss masks for each of the arguments.
  """
  if (num_first_steps_to_ignore + num_last_steps_to_ignore >=
      step_type.shape[0]):
    raise ValueError(
        "Total number of steps to ignore ("
        f"{num_first_steps_to_ignore, num_last_steps_to_ignore}) "
        f"should be less than the total timesteps {step_type.shape[0]}")
  chex.assert_rank(step_type, 1)
  # Step type and argument masks don't have identical tree structure but are
  # expected to have the same shape at their leaves.
  chex.assert_equal_shape(jax.tree_leaves([step_type, argument_masks]))
  terminal_state_mask = jnp.not_equal(step_type, int(dm_env.StepType.LAST))
  trajectory_mask = np.zeros(shape=(step_type.shape[0],), dtype=np.bool_)
  trajectory_mask[num_first_steps_to_ignore:step_type.shape[0] -
                  num_last_steps_to_ignore] = True
  trajectory_mask = jnp.asarray(trajectory_mask)
  global_mask = jnp.logical_and(terminal_state_mask, trajectory_mask)
  return jax.tree_map(lambda x: jnp.logical_and(x, global_mask), argument_masks)


def get_masked_log(data: chex.Array,
                   mask: chex.Array) -> Dict[log_utils.ReduceType, chex.Array]:
  """Gets the masked value and count of data to log."""
  masked_data = jnp.where(mask, data, 0)
  return {
      log_utils.ReduceType.MEAN: masked_data,
      log_utils.ReduceType.SUM: masked_data,
      log_utils.ReduceType.NUM: mask.astype(jnp.int32)}

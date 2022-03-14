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

"""Supervised (Behaviour Cloning) losses."""

import functools
from typing import Dict, Mapping, Tuple

from alphastar import types
from alphastar.architectures import util as arch_util
from alphastar.commons import log_utils
from alphastar.unplugged.losses import loss_base
from alphastar.unplugged.losses import util as loss_util
import chex
from dm_env import specs
import jax
import jax.numpy as jnp


class Supervised(loss_base.Loss):
  """Supervised loss.

  Logits are assumed to be already masked, ie. if a mask is 0, we assume that
  the corresponding logit is -infinity.
  """

  def __init__(self,
               action_spec: types.ActionSpec,
               weights: Mapping[types.ArgumentName, float],
               burnin_len: int = 0,
               overlap_len: int = 0,
               name: str = 'Supervised'):
    """Initializes the module.

    Args:
      action_spec: The action specification.
      weights: The weight of the cross-entropy loss for each part of the loss.
        Its structure must match the action spec.
      burnin_len: The number of steps used for burn-in. The loss is not applied
        to these steps.
      overlap_len: The number of steps at the end of the trajectory where we the
        loss should not be applied.
      name: Name of the module.
    """
    super().__init__(name=name)
    self._burnin_len = burnin_len
    self._overlap_len = overlap_len
    self._action_spec = action_spec
    self._weights = weights
    if set(weights.keys()) != set(action_spec.keys()):
      raise ValueError('Weights keys do not match action spec keys. Got '
                       f'{set(weights.keys())} and {set(action_spec.keys())}.')

  @property
  def input_spec(self) -> types.SpecDict:
    """Gets the specs for inputs to supervised loss."""
    spec = types.SpecDict()
    spec['step_type'] = specs.Array((), jnp.int32)
    for arg, arg_spec in self._action_spec.items():
      spec['argument_masks', arg] = specs.Array((), jnp.bool_)
      num_logits = arg_spec.maximum + 1
      if arg == arch_util.Argument.UNIT_TAGS:
        num_unit_tags = arg_spec.shape[0]
        spec['logits', arg] = specs.Array(
            (num_unit_tags, num_logits), jnp.float32)
        spec['masks', arg] = specs.Array((num_unit_tags, num_logits), jnp.bool_)
        spec['action', arg] = specs.Array((num_unit_tags,), jnp.int32)
      else:
        spec['logits', arg] = specs.Array((num_logits,), jnp.float32)
        spec['masks', arg] = specs.Array((num_logits,), jnp.bool_)
        spec['action', arg] = specs.Array((), jnp.int32)
    return spec

  def _single_arg_loss(self,
                       logits: chex.Array,
                       action: chex.Array,
                       mask: chex.Array,
                       global_mask: chex.Array,
                       weight: float,
                       ) -> Tuple[chex.Array, Dict[str, chex.Array]]:
    """Returns the loss for a single argument of the action."""

    chex.assert_rank([action, global_mask, logits, mask], [0, 0, 1, 1])
    chex.assert_equal_shape([logits, mask])
    xentropy = -jax.nn.log_softmax(logits)[action]

    # correct_argmax is the MAP of the logits, used for metrics (logging) only
    correct_argmax = jnp.equal(jnp.argmax(logits), action).astype(jnp.float32)
    # Masking
    # If the target action is masked, this means the data is not consistent.
    # We set the loss to zero in this case to avoid divergence.
    target_mask = mask[action]
    xentropy_mask = jnp.logical_and(target_mask, global_mask)
    xentropy = jnp.where(xentropy_mask, xentropy, 0)

    loss = weight * xentropy

    log = {}
    log['xentropy'] = loss_util.get_masked_log(xentropy, global_mask)
    log['loss'] = loss_util.get_masked_log(loss, global_mask)
    log['argmax_accuracy'] = loss_util.get_masked_log(correct_argmax,
                                                      global_mask)
    log['unmasked_ratio'] = loss_util.get_masked_log(global_mask,
                                                     jnp.ones_like(global_mask))
    log['wrong_mask_ratio'] = loss_util.get_masked_log(1. - target_mask,
                                                       global_mask)
    return loss, log

  def _loss(self, inputs: types.StreamDict) -> loss_base.LossOutputType:
    """The loss function."""
    global_masks = loss_util.get_global_loss_masks(
        step_type=inputs['step_type'],
        argument_masks=inputs.get('argument_masks'),
        num_first_steps_to_ignore=self._burnin_len,
        num_last_steps_to_ignore=self._overlap_len)

    # Compute all the losses
    losses, logs = [], {}
    # Reduce functions used for unit_tags, reducing over axis 1 (see below):
    unit_tags_reduce_fns = {k: functools.partial(fn, axis=1)
                            for k, fn in log_utils.REDUCE_FUNCTIONS.items()}
    unit_tags_reduce_fns[log_utils.ReduceType.NON_REDUCED] = lambda x: x[:, 0]
    for arg in self._action_spec:
      single_arg_loss = jax.vmap(self._single_arg_loss, [0, 0, 0, 0, None])
      if arg == arch_util.Argument.UNIT_TAGS:
        # Unlike all other arguments, of size [unroll_len, ...], unit_tags have
        # size [unroll_len, max_num_selected_units, ...] representing
        # max_num_selected_units independent actions. We use a vmap to compute
        # the loss independently over this axis:
        single_arg_loss = jax.vmap(
            single_arg_loss, [1, 1, 1, None, None], (1, 1))
      arg_loss, arg_log = single_arg_loss(
          inputs['logits', arg], inputs['action', arg],
          inputs['masks', arg], global_masks[arg], self._weights[arg])
      if arg == arch_util.Argument.UNIT_TAGS:
        # We now reduce the unit_tags loss and logs over axis 1, so that they
        # have the same shape as other arguments:
        arg_loss = jnp.mean(arg_loss, axis=1)
        arg_log = log_utils.reduce_logs(arg_log, unit_tags_reduce_fns)
      losses.append(arg_loss)
      for log_name, log in arg_log.items():
        logs[f'[{self.name}] {arg}_{log_name}'] = log
    total_loss = sum(losses)
    logs[f'[{self.name}] loss'] = {log_utils.ReduceType.MEAN: total_loss}

    return total_loss, logs

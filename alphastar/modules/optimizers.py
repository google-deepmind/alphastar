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

"""Optimizers for the learner."""

import enum
import functools
from typing import Any, Callable, List, Optional, Tuple

from absl import logging
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax


class LearningRateScheduleType(str, enum.Enum):
  STAIRCASE = 'staircase'
  COSINE = 'cosine'


class WeightDecayFilterType(str, enum.Enum):
  LAYER_NORM = 'layer_norm'
  BIAS = 'bias'


WeightDecayFilter = Callable[[optax.Params], Any]


# Schedules:
def get_staircase_schedule(
    num_updates_before_decay: int,
    drop_factor: float,
    num_decays_cutoff: int = 9) -> optax.Schedule:
  """Gets a staircase style schedule for learning rate."""
  if num_updates_before_decay * num_decays_cutoff > np.iinfo(np.int32).max:
    raise ValueError('num_updates_before_decay is too large to fit into int32. '
                     'Decrease it or increase batch_size and/or unroll_len.')
  return optax.piecewise_constant_schedule(
      init_value=1.0,
      boundaries_and_scales={
          k * num_updates_before_decay: drop_factor
          for k in range(1, num_decays_cutoff+1)})


def get_cosine_schedule(
    num_updates_before_decay: int,
    total_num_training_updates: int) -> optax.Schedule:
  """Gets a cosine decay schedule for learning rate."""
  num_decayed_updates = total_num_training_updates - num_updates_before_decay
  def schedule(count):
    offset_count = jnp.maximum(count - num_updates_before_decay, 0)
    return optax.cosine_decay_schedule(
        init_value=1, decay_steps=num_decayed_updates)(offset_count)
  return schedule


def add_warmup_to_schedule(
    num_warmup_updates: int,
    wrapped_schedule: optax.Schedule) -> optax.Schedule:
  """Wrapper module to add warmup to any schedule."""
  def schedule(count):
    factor = jnp.minimum(count / num_warmup_updates, 1.)
    return wrapped_schedule(count) * factor
  return schedule


# Weight decay:
def layer_norm_weight_filter(params):
  def f(module_name, name, value):
    del name, value
    return 'layer_norm' not in module_name
  return hk.data_structures.map(f, params)


def bias_weight_filter(params):
  def f(module_name, name, value):
    del module_name, value
    return name != 'b'
  return hk.data_structures.map(f, params)


weight_decay_filters = {
    WeightDecayFilterType.LAYER_NORM: layer_norm_weight_filter,
    WeightDecayFilterType.BIAS: bias_weight_filter,
}


def _logging_fn(opt_state, lr_schedule, scale_index: int, learning_rate: float):
  """A logging function that extracts logs fom optimizer state.

  Args:
    opt_state: Optimizer state.
    lr_schedule: Learning rate schedule function.
    scale_index: Index of learning rate scale in optimizer state.
    learning_rate: Learning Rate fed into optimizer.

  Returns:
    Logs from the optimizer state.
  """
  log = {}
  learning_rate_scale = jnp.squeeze(lr_schedule(opt_state[scale_index].count))
  log['learning_rate_scale'] = learning_rate_scale
  log['learning_rate'] = learning_rate * learning_rate_scale
  return log


def get_optimizer(
    num_frames_per_learner_update: int,
    extra_weight_decay_mask_fn: Optional[WeightDecayFilter],
    total_num_training_frames: int,
    weight_decay_filter_out: List[str],
    learning_rate: float,
    learning_rate_schedule_type: LearningRateScheduleType,
    lr_frames_before_decay: float,
    lr_num_warmup_frames: float,
    adam_b1: float,
    adam_b2: float,
    adam_eps: float,
    use_adamw: bool,
    weight_decay: float,
    staircase_lr_drop_factor: float,
    before_adam_gradient_clipping_norm: Optional[float] = None,
    after_adam_gradient_clipping_norm: Optional[float] = None,
    ) -> Tuple[optax.GradientTransformation, Any]:
  """Build the optimizer from the flags.

  Args:
    num_frames_per_learner_update: The number of frames processed per learner
      update.
    extra_weight_decay_mask_fn: A function which takes params and returns a tree
      with boolean leaves stating whether we should apply weight decay to this
      weight vector.
    total_num_training_frames : Total number of training frames
    weight_decay_filter_out: Specify which parameters are ignored by weight
      decay. Must be part of
      {"|".join(list(optimizers.WeightDecayFilterType))}].'
    learning_rate: Initial learning rate
    learning_rate_schedule_type: Type of learning rate schedule.
    lr_frames_before_decay: Number of training frames before the learning rate
      starts being reduced
    lr_num_warmup_frames: Number of steps for learning rate warmup
    adam_b1: Adam b1 parameter
    adam_b2: Adam b2 parameter
    adam_eps: Adam epsilon parameter
    use_adamw: Whether to use AdamW. If not, weight decay is applied before
      Adam
    weight_decay: Co-efficient of weight decay,
    staircase_lr_drop_factor: Multiply the learning rate by this when decaying
    before_adam_gradient_clipping_norm: Global gradient norm for clipping
      before Adam.
    after_adam_gradient_clipping_norm: Global gradient norm for clipping after
      Adam

  Returns:
    A tuple containing the optax optimizer and a logging function
      which take the optimizer state as an input and return dict to log.
  """
  optimizers = []

  if weight_decay:
    mask_fns = [weight_decay_filters[x] for x in weight_decay_filter_out]
    if extra_weight_decay_mask_fn:
      mask_fns.append(extra_weight_decay_mask_fn)
    def mask_fn(params):
      all_masks = [f(params) for f in mask_fns]
      if all_masks:
        output = jax.tree_map(lambda *masks: all(masks), *all_masks)
      else:
        output = jax.tree_map(lambda _: True, params)
      logging.info('Using weight decay filter:\n%s', output)
      return output
    weight_decay = optax.masked(
        inner=optax.additive_weight_decay(weight_decay),
        mask=mask_fn)
  else:
    weight_decay = None

  if weight_decay and not use_adamw:
    optimizers.append(weight_decay)
  if before_adam_gradient_clipping_norm:
    optimizers.append(optax.clip_by_global_norm(
        before_adam_gradient_clipping_norm))
  optimizers.append(optax.scale_by_adam(
      b1=adam_b1, b2=adam_b2, eps=adam_eps))
  if weight_decay and use_adamw:
    optimizers.append(weight_decay)
  if after_adam_gradient_clipping_norm:
    optimizers.append(optax.clip_by_global_norm(
        after_adam_gradient_clipping_norm))

  num_updates_before_decay = int(
      lr_frames_before_decay / num_frames_per_learner_update)
  total_num_training_updates = int(
      total_num_training_frames / num_frames_per_learner_update)
  if learning_rate_schedule_type == LearningRateScheduleType.STAIRCASE:
    lr_schedule = get_staircase_schedule(
        num_updates_before_decay=num_updates_before_decay,
        drop_factor=staircase_lr_drop_factor)
  elif learning_rate_schedule_type == LearningRateScheduleType.COSINE:
    lr_schedule = get_cosine_schedule(
        num_updates_before_decay=num_updates_before_decay,
        total_num_training_updates=total_num_training_updates)
  else:
    raise ValueError(f'Unknown schedule {learning_rate_schedule_type}.')

  if lr_num_warmup_frames:
    num_warmup_updates = int(
        lr_num_warmup_frames / num_frames_per_learner_update)
    lr_schedule = add_warmup_to_schedule(num_warmup_updates, lr_schedule)

  logging_fn = functools.partial(
      _logging_fn, lr_schedule=lr_schedule, scale_index=len(optimizers),
      learning_rate=learning_rate)
  optimizers.append(optax.scale_by_schedule(lr_schedule))
  optimizers.append(optax.scale(-learning_rate))

  return optax.chain(*optimizers), logging_fn

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

"""ACME based learner for Alphastar.
"""
# pylint: disable=logging-fstring-interpolation

import time
from typing import Any, Mapping, Optional, Sequence, Tuple

from absl import logging
import acme
from acme.jax import utils
from acme.utils import counting
from acme.utils import loggers
from alphastar.architectures import modular
from alphastar.commons import log_utils
from alphastar.commons import metrics
from alphastar.modules import agent as agent_lib
from alphastar.modules import common
from alphastar.unplugged import losses
from alphastar.unplugged.data import data_source_base
from alphastar.unplugged.data import util as data_util
import chex
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np
import optax
import reverb
import tree

_PMAP_AXIS_NAME = 'data'


@chex.dataclass
class TrainingState:
  """Training state consists of network parameters and optimiser state."""
  params: hk.Params
  opt_state: optax.OptState
  net_state: Any
  step: int
  rng: chex.PRNGKey


class SupervisedLearner(acme.Learner):
  """Supervised Learner module for newer style architectures (eg. v3)."""

  def __init__(self,
               architecture_builder: modular.ArchitectureBuilder,
               loss_builder: losses.LossBuilder,
               batch_size: int,
               unroll_len: int,
               overlap_len: int,
               frames_per_step: int,
               data_source: data_source_base.DataSource,
               optimizer: optax.GradientTransformation,
               optimizer_logs_fn,
               rng_key: chex.PRNGKey,
               counter: Optional[counting.Counter] = None,
               logger: Optional[loggers.Logger] = None,
               devices: Optional[Sequence[jax.xla.Device]] = None,
               increment_counts: bool = True,
               reduce_metrics_all_devices: bool = False,
               log_every_n_seconds: int = 60,
               log_to_csv: bool = True):
    """Initializes a supervised learner.

    Args:
      architecture_builder : A builder that constructs the agent architecture.
      loss_builder : A builder function that constructs the training loss.
      batch_size : Training batch size used per host.
      unroll_len : Unroll length (sequence length) for the inputs.
      overlap_len : Overlap length between successful sequences.
      frames_per_step : Number of frames used per training step.
      data_source : Data source that is used as data iterator for training.
      optimizer: An optax optimizer module.
      optimizer_logs_fn : Helper function that logs optimizer statistics.
      rng_key: A JAX random number generator.
      counter: An ACME counter object that keeps counts of different ordinal
        statistcs in training.
      logger: An ACME logger object that logs training metrics.
      devices: XLA devices for the learner model.
      increment_counts: Boolean to decide if the learner needs to increment
        counts. This can be True for a primary learner and False for other
        learners in multi-host training.
      reduce_metrics_all_devices: Boolean to decide if metrics needed to be
        reduced across all hosts in multi-host training.
      log_every_n_seconds: Interval between logs in seconds.
      log_to_csv: Boolean to decide if logging to CSV is necessary. By default,
        training stats are logged to the terminal.
    """

    local_devices = jax.local_devices()
    devices = devices or jax.devices()
    local_devices = [d for d in devices if d in local_devices]
    logging.info(f'In total, there are {devices} devices and '
                 f'{local_devices} local devices. \n Devices are '
                 f' : {devices} and local devices are {local_devices}')

    # Error checks.
    if batch_size % len(local_devices) != 0:
      raise ValueError(
          f'Batch size ({batch_size}) must always be a multiple of number of '
          f'training devices on each host ({local_devices}).')

    # Setup and initialization.
    action_spec = data_source.action_spec
    input_spec = data_source.input_spec
    agent = agent_lib.AlphaStarAgent(
        architecture_builder(input_spec, action_spec, True))
    training_loss = loss_builder(action_spec)

    def _loss(params, state, key, data):
      if 'prev_features' in data:
        logging.log_first_n(logging.INFO, 'Using prev_features', n=1)
        state.update(jax.tree_map(lambda x: x[:, 0], data.get('prev_features')))
        del data['prev_features']
      outputs, next_state, _ = agent.apply(params, key, data, state)
      loss_inputs = data.copy()
      loss_inputs.update(outputs)
      loss, logs = training_loss.batched_loss(loss_inputs)
      mean_loss = jnp.mean(loss, axis=[0, 1])
      reduced_logs = log_utils.reduce_logs(logs)
      return mean_loss, (reduced_logs, next_state)

    def update_step(
        state: TrainingState, sample: reverb.ReplaySample
    ) -> Tuple[TrainingState, Mapping[str, jnp.ndarray]]:
      """Computes an SGD step, returning new state and metrics for logging."""

      # Compute gradients.
      grad_fn = jax.value_and_grad(_loss, has_aux=True)
      loss_key, new_key = jax.random.split(state.rng)
      (loss_value,
       (logs, net_state)), gradients = grad_fn(state.params, state.net_state,
                                               loss_key, sample)

      # Average gradients over pmap replicas before optimizer update.
      gradients = jax.lax.pmean(gradients, _PMAP_AXIS_NAME)

      # Apply updates.
      updates, new_opt_state = optimizer.update(gradients, state.opt_state,
                                                state.params)
      new_params = optax.apply_updates(state.params, updates)

      if reduce_metrics_all_devices:
        loss_value = jax.lax.pmean(loss_value, _PMAP_AXIS_NAME)
        logs = metrics.reduce_metrics(logs, axis_name=_PMAP_AXIS_NAME)

      training_metrics = {
          'loss': loss_value,
          'gradient_norm': optax.global_norm(gradients),
          'param_norm': optax.global_norm(new_params),
          'param_updates_norm': optax.global_norm(updates),
      }

      training_metrics.update(common.flatten_metrics(logs))
      training_metrics.update(
          common.flatten_metrics(optimizer_logs_fn(new_opt_state)))
      new_steps = state.step + 1

      new_state = TrainingState(
          params=new_params,
          opt_state=new_opt_state,
          net_state=net_state,
          step=new_steps,
          rng=new_key,
      )

      return new_state, training_metrics

    def make_initial_state(key: jnp.ndarray) -> TrainingState:
      """Initialises the training state (parameters and optimiser state)."""
      key, new_key = jax.random.split(key)
      per_local_device_batch_size = int(batch_size / len(local_devices))
      dummy_obs = data_util.get_dummy_observation(
          input_spec=data_source.input_spec,
          batch_size=per_local_device_batch_size,
          unroll_len=unroll_len)
      initial_state_key, params_key = jax.random.split(key)
      initial_state = agent.initial_state(
          initial_state_key, per_local_device_batch_size)
      params = agent.init(params_key, dummy_obs, initial_state)

      params_log_lines = ['All params:']
      tree.map_structure_with_path(
          lambda path, v: params_log_lines.append(f'{path} {v.shape}'), params)
      logging.info('\n'.join(params_log_lines))

      initial_opt_state = optimizer.init(params)
      return TrainingState(
          params=params,
          net_state=initial_state,
          opt_state=initial_opt_state,
          step=0,
          rng=new_key)

    # Initialize state.
    rng_key, init_rng = jax.random.split(rng_key)
    state = make_initial_state(init_rng)

    self._local_devices = local_devices
    self._frames_per_step = frames_per_step
    self._state = utils.replicate_in_all_devices(state, local_devices)
    self._prefetched_data_iterator = data_source.get_generator()
    self._update_step = jax.pmap(
        update_step, axis_name=_PMAP_AXIS_NAME, devices=devices)

    # Set up logging/counting.
    self._counter = counter
    self._logger = logger or common.make_default_logger(
        'learner', time_delta=log_every_n_seconds, log_to_csv=log_to_csv)
    self._increment_counts = increment_counts

  def _preprocess(self, data):
    """Reshapes input so that it can be distributed across multiple cores."""

    def add_core_dimension(x):
      num_devices = len(self._local_devices)
      if x.shape[0] % num_devices != 0:
        raise ValueError(f'The batch size must be a multiple of the number of'
                         f' devices. Got batch size = {x.shape[0]} and number'
                         f' of devices = {num_devices}.')
      prefix = (num_devices, x.shape[0] // num_devices)
      return np.reshape(x, prefix + x.shape[1:])

    multi_inputs = jax.tree_map(add_core_dimension, data)
    return multi_inputs

  def step(self):
    """Does a step of SGD and logs the results."""

    # Do a batch of SGD.
    start = time.time()
    samples = self._preprocess(next(self._prefetched_data_iterator))

    self._state, results = self._update_step(self._state, samples)

    # Take results from first replica.
    results = utils.get_from_first_device(results, as_numpy=False)

    if self._counter:
      if self._increment_counts:
        counts = self._counter.increment(
            steps=1,
            num_frames=self._frames_per_step,
            time_elapsed=time.time() - start)
      else:
        counts = self._counter.get_counts()
      if 'learner_steps' in counts:
        results['steps_per_second'] = counts['learner_steps'] / counts[
            'learner_time_elapsed']
        results['frames_per_second'] = counts['learner_num_frames'] / counts[
            'learner_time_elapsed']
    else:
      counts = {}

    # Snapshot and attempt to write logs. Logger already throttles the logging.
    self._logger.write({**results, **counts})

  def get_variables(self, names: Sequence[str]) -> Sequence[hk.Params]:
    # Return first replica of parameters.
    return [utils.get_from_first_device(self._state.params, as_numpy=False)]

  def save(self) -> TrainingState:
    # Serialize only the first replica of parameters and optimizer state.
    return jax.tree_map(utils.get_from_first_device, self._state)

  def restore(self, state: TrainingState):
    self._state = utils.replicate_in_all_devices(state, self._local_devices)

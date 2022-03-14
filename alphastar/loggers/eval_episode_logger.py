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

"""Logger for logging evaluation episodes."""

import collections as py_collections
import time
from typing import Any, Callable, Dict, Optional

from acme.utils import loggers
from alphastar import types
from alphastar.commons import log_utils
from alphastar.commons import metrics
from alphastar.loggers import episode_logger as episode_logger_lib
from alphastar.modules import common as acme_common
import dm_env
import jax
import numpy as np


class EvalEpisodeLogger(episode_logger_lib.EpisodeLogger):
  """A logger for logging episodes during evaluation."""

  def __init__(
      self,
      log_name: str,
      log_to_csv: bool,
      home_player_idx: int = 0,
      agent_scope=None,
      logger: Optional[loggers.Logger] = None,
      print_fn: Optional[Callable[[str], None]] = None,
  ):
    """Initializes the evaluation episode logger.

    Args:
      log_name: Name of the log file.
      log_to_csv: Boolean to decide if logging to csv is needed.
      home_player_idx: Index for the home player.
      agent_scope: Logger used to write agent outputs and observations
        beyond what the standard logging does.
      logger: An ACME logger object used for logging.
      print_fn: The default print function used for in-process logging.
    """

    super().__init__()
    self._home_player_idx = home_player_idx
    self._logger = logger
    self._log_name = log_name
    self._log_to_csv = log_to_csv
    self._print_fn = print_fn or print
    if self._logger is None:
      self.make_default_logger()
    self._new_episode()
    self._agent_scope = agent_scope

  def make_default_logger(
      self, print_fn: Optional[Callable[[str], None]] = None,):
    self._logger = acme_common.make_default_logger(
        self._log_name, log_to_csv=self._log_to_csv, print_fn=self._print_fn)

  def _new_episode(self):
    self._start_time = time.time()
    self._num_home_frames = 0
    self._home_player_logs = py_collections.defaultdict(lambda: [])

  def register_step(self,
                    player: int,
                    step_type: dm_env.StepType,
                    prev_reward: float,
                    observation: types.StreamDict,
                    agent_output: Optional[types.StreamDict],
                    log: Optional[log_utils.Log],
                    static_log: Dict[str, Any]):
    if player == self._home_player_idx:
      self._num_home_frames += 1
      home_log = log or {}
      for k, v in metrics.flatten_metrics(home_log).items():
        self._home_player_logs[k].append(np.mean(v))

      if self._num_home_frames % 50 == 1:
        self._print_fn(
            'Running episode, frame=%s, game_loop=%s, static_log=%s, log=%s.' %
            (self._num_home_frames, observation['game_loop'], static_log, log))

      if step_type == dm_env.StepType.LAST:
        home_obs = observation
        episode_length = np.squeeze(home_obs['game_loop'])
        episode_length_minutes = episode_length / 22.4 / 60
        avg_home_player_logs = {
            k: np.mean(v) for k, v in self._home_player_logs.items()}
        eval_time_seconds = time.time() - self._start_time
        log_data = dict(
            outcome=prev_reward,
            num_frames_per_episode=self._num_home_frames,
            episode_length_minutes=episode_length_minutes,
            apm=self._num_home_frames / episode_length_minutes,
            episode_eval_time_seconds=eval_time_seconds,
            eval_frames_per_second=self._num_home_frames / eval_time_seconds,
            **static_log,
            **avg_home_player_logs)
        self._logger.write(acme_common.flatten_metrics(log_data))
        self._new_episode()

    if self._agent_scope and agent_output:
      obs_without_zerodimarrays = jax.tree_map(
          lambda x: x.item() if x.ndim == 0 else x, observation)
      self._agent_scope.write(
          output=agent_output, observation=obs_without_zerodimarrays.asdict())

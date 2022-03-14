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

"""Abstraction for data logged by actors / evaluators."""

import abc
from typing import Any, Dict, Optional, Union

from alphastar import types
from alphastar.commons import log_utils
import dm_env

Number = Union[int, float]


class EpisodeLogger(abc.ABC):

  @abc.abstractmethod
  def register_step(self,
                    player: int,
                    step_type: dm_env.StepType,
                    prev_reward: float,
                    observation: types.StreamDict,
                    agent_output: Optional[types.StreamDict],
                    log: Optional[log_utils.Log],
                    static_log: Dict[str, Any]):
    """Registers a new step.

    Args:
      player: The zero-based index of the player registering this step.
      step_type: The step type.
      prev_reward: The reward received before the observation by the player.
      observation: The observation for the player.
      agent_output: The output produced by the agent at this time step, or None
        if there is no output (last step, or bot/competitor).
      log: What the player needs to log at this step, or None if nothing to log.
        Note that because these logs are typically averaged over episodes, they
        can only be numeric types.
      static_log: Static log for the player. Static log is expected not to
        change during the episode, but can be of any type.
    """

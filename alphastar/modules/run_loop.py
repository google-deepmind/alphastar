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

"""Episode run loop utilities."""

from typing import Any, Dict, Optional

from alphastar import types
from alphastar.loggers import episode_logger as episode_logger_lib
from alphastar.modules import evaluator_base
import dm_env
import jax
import numpy as np


def play_episode(
    env: dm_env.Environment,
    agent: evaluator_base.Evaluator,
    player: int,
    episode_logger: episode_logger_lib.EpisodeLogger,
    static_log: Dict[str, Any],
    max_num_steps: Optional[int] = None):
  """Plays out an episode against an environment using the specified evaluator.

  Args:
    env: dm_env that will accept PySC2 actions and return PySC2 observations.
    agent: An agent evaluator.
    player: Index of the player in the game (0 or 1).
    episode_logger: Logger for step output.
    static_log: Static log output for the episode logger.
    max_num_steps: Maximum number of steps befpre termination.

  Returns:
    Replay bytes.

  Raises:
    RuntimeError: if the agent output has a wrong format.
  """
  timestep = env.step({})
  step_counter = 0
  while timestep.step_type != dm_env.StepType.LAST:
    first_step = timestep.step_type == dm_env.StepType.FIRST
    agent_output, logs = agent.step(timestep)
    episode_logger.register_step(
        player=player,
        step_type=timestep.step_type,
        prev_reward=timestep.reward if not first_step else 0.0,
        observation=timestep.observation,
        agent_output=agent_output,
        log=logs,
        static_log=static_log)
    # Can this be done by the evaluator(s)?
    if 'action' not in agent_output:
      raise RuntimeError(
          'The agent must provide an "action" key in its output.')
    try:
      timestep = env.step(jax.tree_map(np.squeeze, agent_output.get('action')))
    except TypeError as e:
      raise RuntimeError(
          f'Action issue in stepping : {agent_output["action"]}') from e
    step_counter += 1
    if max_num_steps and step_counter >= max_num_steps:
      break

  # Register the final step.
  episode_logger.register_step(
      player=player,
      step_type=timestep.step_type,
      prev_reward=timestep.reward,
      observation=types.StreamDict(timestep.observation),
      agent_output=None,
      log=None,
      static_log=static_log)

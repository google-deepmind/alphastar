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

"""Evaluator abstraction, for performing inference."""

import abc
import enum
from typing import Tuple

from alphastar import types
from alphastar.commons import log_utils
import dm_env


class EvaluatorType(enum.IntEnum):
  """Defines different types of evaluation."""
  CHECKPOINT = 0
  RANDOM_PARAMS = 1


class Evaluator(abc.ABC):
  """Evaluator abstraction."""

  @abc.abstractmethod
  def reset(self) -> types.StreamDict:
    """Resets the evaluator in preparation for a new episode.

    Returns:
      Dict of data reflecting the current state of the evaluator.
    """

  @abc.abstractmethod
  def step(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[types.StreamDict, log_utils.Log]:
    """Steps the evaluator.

    Args:
      timestep: The latest environment step.

    Returns:
      (agent output (must have an `action` attribute), output_logs).
    """

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

"""Base data source class for AlphaStar data."""

import abc
import enum
from alphastar import types
from typing import Generator, Optional, Tuple

from alphastar import collections
from pysc2.env import enums as sc2_enums


class DataSplit(str, enum.Enum):
  TRAIN = "train"
  TEST = "test"
  DEBUG = "debug"


class DataSource(abc.ABC):
  """Abstract class for data sources."""

  def __init__(self,
               unroll_len: int,
               overlap_len: int,
               batch_size: int,
               home_race: Optional[sc2_enums.Race],
               away_race: Optional[sc2_enums.Race]):
    self._unroll_len = unroll_len
    self._overlap_len = overlap_len
    if self._overlap_len >= self._unroll_len:
      raise ValueError("Rollout length must be larger than overlap.")
    self._batch_size = batch_size
    if home_race == sc2_enums.Race.random or away_race == sc2_enums.Race.random:
      # We raise an error here as using random can either mean only the random
      # race, or any race, depending on the parts of the code.
      raise ValueError("Filtering random race is not supported. "
                       "Use None to disable filtering.")
    self._home_race = home_race
    self._away_race = away_race

  @property
  @abc.abstractmethod
  def env_spec(self) -> Tuple[types.ObsSpec, types.ActionSpec]:
    """The environment spec."""

  @property
  @abc.abstractmethod
  def input_spec(self) -> types.SpecDict:
    """The full spec of the input to the agent."""

  @property
  def obs_spec(self) -> types.ObsSpec:
    return self.env_spec[0]

  @property
  def action_spec(self) -> types.ActionSpec:
    return self.env_spec[1]

  @abc.abstractmethod
  def get_generator(self) -> Generator[collections.Struct, None, None]:
    """Returns a data generator."""

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

"""Repository of Competitors and associated utilities."""

import enum
import itertools
from typing import Optional, Sequence

import numpy as np

from s2clientprotocol import common_pb2
from s2clientprotocol import sc2api_pb2


class CompetitorType(enum.Enum):
  SHELL = 1,   # A competitor built with agent shell with its own trained model.
  PROTO = 2,   # A competitor built from a proto file of agent properties.
  BOT = 3,     # A competitor that is a built in bot.
  BINARY = 4   # A competitor binary which can join hosted games.


class BuiltInAI:
  """The built-in AI of a particular difficulty level."""

  def __init__(self,
               difficulty: str,
               home_races: Sequence[str] = ("protoss", "terran", "zerg",
                                            "random"),
               competitor_race: Optional[str] = None,
               sample_race: bool = True):
    """Initializes a built-in bot.

    Args:
      difficulty: The level of difficulty of the bot.
      home_races: Set of home races that bot can play with.
      competitor_race: Race for the competitor.
      sample_race : Boolen to decide if we need to sample the race for the
        competitor

    Raises:
      ValueError if competitor_race is not set and sample_race is False.
    """

    self._difficulty = difficulty
    self._home_races = home_races
    if competitor_race:
      self._competitor_race = competitor_race
    elif not sample_race and len(home_races) > 1:
      raise ValueError("Race is not sampled, but competitor_race "
                       "for BuiltInAI is not set!")
    else:
      self._competitor_race = np.random.choice(home_races)

  @property
  def type(self):
    return CompetitorType.BOT

  @property
  def difficulty(self):
    return self._difficulty

  def intersect_home_races(self, opponent_away_races, ignore_if_not_compatible):
    home_races = [x for x in self._home_races if x in opponent_away_races]
    if home_races or not ignore_if_not_compatible:
      self._home_races = home_races

  def set_competitor_race(self, competitor_race):
    self._competitor_race = competitor_race

  @property
  def home_races(self):
    return self._home_races

  @property
  def away_races(self):
    return ["protoss", "terran", "zerg", "random"]

  @property
  def competitor_race(self):
    return self._competitor_race

_BUILT_IN_BOTS = [
    "very_easy", "easy", "medium", "medium_hard", "hard", "harder",
    "very_hard", "cheat_vision", "cheat_money", "cheat_insane"
]

_BOT_RACES = ["terran", "zerg", "protoss"]

BUILT_IN_COMPETITORS = {k: BuiltInAI(k) for k in _BUILT_IN_BOTS}
BUILT_IN_COMPETITORS.update({
    "{}_{}".format(bot, race): BuiltInAI(bot, competitor_race=race)
    for (bot, race) in itertools.product(_BUILT_IN_BOTS, _BOT_RACES)})


def race_string_to_enum(race):
  return common_pb2.Race.Value(race.capitalize())


def difficulty_string_to_enum(difficulty):
  return sc2api_pb2.Difficulty.Value(
      "".join(s.capitalize() for s in difficulty.split("_")))


def is_built_in_bot(difficulty):
  return difficulty in BUILT_IN_COMPETITORS

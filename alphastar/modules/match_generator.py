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

"""Generates match setups, given race and map constraints and an opponent."""

import collections
import random
from typing import Sequence

from absl import logging
from alphastar.commons import competitors
from pysc2.env import sc2_env

Match = collections.namedtuple('Match', ['home_race', 'away_race', 'map_name'])


class MatchGenerator:
  """Generates match setups, given race and map constraints and an opponent."""

  def __init__(self, home_races: Sequence[sc2_env.Race],
               away_races: Sequence[sc2_env.Race], map_names: Sequence[str]):
    """Initializer.

    Args:
      home_races: Races which the home (first) player may select.
      away_races: Races which the away (second) player may select.
      map_names: Maps which may be selected from.
    """
    if not home_races:
      raise ValueError('home_races must be non-empty')
    if not away_races:
      raise ValueError('away_races must be non-empty')
    if not map_names:
      raise ValueError('map_names must be non-empty')

    self._home_races = home_races
    self._away_races = away_races
    self._map_names = map_names

  def generate(self, opponent_name: str) -> Match:
    """Returns a random Match, taking into account opponent constraints.

    Args:
      opponent_name: Built-in bot difficulty name, else a competitor name.
    """
    if competitors.is_built_in_bot(opponent_name):
      home_races_available = self._home_races
      away_races_available = self._away_races
      map_names_available = self._map_names
    else:
      raise ValueError('Only games against built-in bots are supported at '
                       'the moment.')

    home_race = sc2_env.Race(random.choice(home_races_available))
    away_race = sc2_env.Race(random.choice(away_races_available))
    map_name = random.choice(map_names_available)
    logging.info('Match setup: Agent as %s, %s as %s - on %s.', home_race.name,
                 opponent_name, away_race.name, map_name)
    return Match(home_race, away_race, map_name)

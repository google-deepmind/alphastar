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

"""Utility functions for agents."""

import math
from alphastar import types


def get_world_size(action_spec: types.ActionSpec) -> int:
  """Gets number of horizontal and vertical pixels of the (square) world map."""
  world_size = int(math.sqrt(action_spec["world"].maximum + 1))
  assert world_size * world_size == action_spec["world"].maximum + 1
  return world_size


class Argument:
  """List of starcraft action argument names."""
  FUNCTION = "function"
  DELAY = "delay"
  QUEUED = "queued"
  REPEAT = "repeat"
  UNIT_TAGS = "unit_tags"
  TARGET_UNIT_TAG = "target_unit_tag"
  WORLD = "world"


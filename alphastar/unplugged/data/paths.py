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

"""AlphaStar data source paths."""

import os
from typing import Optional

ALL_REPLAY_VERSIONS = ('4.8.2', '4.8.3', '4.8.4', '4.8.6', '4.9.0', '4.9.1',
                       '4.9.2')
DEFAULT_BASE_PATH = ''

DATASET_PATHS = {
    # Open source datasets currently don't exist.
}
# pylint: enable=line-too-long


def get_dataset_pattern(replay_versions,
                        data_split: str,
                        player_min_mmr: int,
                        base_path: str = DEFAULT_BASE_PATH) -> Optional[str]:
  """Gets the dataset file pattern from replay versions."""
  if not base_path:
    raise ValueError(f'Base path ({base_path}) for data cannot be None.')
  pattern_key = (replay_versions, data_split, player_min_mmr)
  pattern = DATASET_PATHS.get(pattern_key, None)
  if not pattern:
    return None
  return os.path.join(base_path, pattern)

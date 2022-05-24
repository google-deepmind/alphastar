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

"""Utility functions for AlphaStar dataset paths."""

import importlib.util
import os
from typing import Mapping, Optional, Tuple

_RelativePaths = Mapping[Tuple[Tuple[str, ...], str, int], str]


def _read_config(config_path: str) -> Tuple[str, _RelativePaths]:
  """Dynamically imports dataset paths config file and extracts key info."""
  spec = importlib.util.spec_from_file_location('_paths', config_path)
  if spec is None:
    raise ValueError(
        f'No module loader found for {config_path!r}. '
        'This probably means that the file has an invalid extension. '
        'The configuration file is expected to be a Python module.')
  paths_module = importlib.util.module_from_spec(spec)
  try:
    spec.loader.exec_module(paths_module)
  except FileNotFoundError as e:
    raise ValueError(f'File {config_path} not found.') from e
  return paths_module.BASE_PATH, paths_module.RELATIVE_PATHS


def get_dataset_pattern(
    replay_versions: Tuple[str, ...],
    data_split: str,
    player_min_mmr: int,
    dataset_paths_fname: str,
) -> Optional[str]:
  """Gets the dataset file pattern from replay versions."""
  if dataset_paths_fname is None:
    raise ValueError(f'Dataset paths file is trivial : {dataset_paths_fname} .')
  base_path, relative_paths = _read_config(dataset_paths_fname)
  if not base_path:
    raise ValueError(f'Base path ({base_path}) for data cannot be None.')
  pattern_key = (replay_versions, data_split, player_min_mmr)
  pattern = relative_paths.get(pattern_key, None)
  if not pattern:
    return None
  return os.path.join(base_path, pattern)

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

"""Data source for Riegeli-based datasets used in Alphastar offline training."""

import functools
from typing import Any, Generator, Mapping, Optional, Sequence, Tuple

from alphastar import types
from alphastar.unplugged.data import data_source_base
from alphastar.unplugged.data import util as data_utils
from alphastar.unplugged.data import paths
from pysc2.env import converted_env
from pysc2.env import enums as sc2_enums
from pysc2.env.converter.proto import converter_pb2
import tensorflow as tf


def make_episodes_dataset(
    dataset_pattern,
    features,
    training=True,
):
  """Makes episodes dataset from files.

  Args:
    dataset_pattern: File path pattern for the dataset.
    features: TF Example features in the dataset.
    training: Boolean to indicate whether in training mode.

  Returns:
    TF Dataset for the episodes data.
  """

  files_ds = tf.data.Dataset.list_files(dataset_pattern, shuffle=training)

  ds = files_ds.interleave(
      tf.data.TFRecordDataset, num_parallel_calls=tf.data.AUTOTUNE)

  serializer = data_utils.TFExampleCoder(features=features, compress=True)
  ds = ds.map(serializer.decode, num_parallel_calls=tf.data.AUTOTUNE)

  ds = ds.repeat()
  return ds


def make_unrolls_dataset(episode_ds: tf.data.Dataset, batch_size: int,
                         unroll_len: int, shuffle_buffer_size: int):
  """Makes unrolls dataset from episodes.

  Args:
    episode_ds: TF Dataset of episodes.
    batch_size: Batch size used in the dataset.
    unroll_len: Length of unroll for each sequence in the batch.
    shuffle_buffer_size : Size of the shuffle buffer used for the unrolls.

  Returns:
    TF Dataset object with unrolled batched sequences.
  """

  ds = episode_ds.flat_map(tf.data.Dataset.from_tensor_slices)
  ds = ds.batch(unroll_len)
  ds = ds.shuffle(
      buffer_size=shuffle_buffer_size, reshuffle_each_iteration=True)
  ds = ds.batch(batch_size)
  ds = ds.prefetch(buffer_size=tf.data.AUTOTUNE)
  return ds


class OfflineTFRecordDataSource(data_source_base.DataSource):
  """A data source that runs the environment to read replays."""

  def __init__(self,
               unroll_len: int,
               overlap_len: int,
               batch_size: int,
               data_split: data_source_base.DataSplit,
               converter_settings: converter_pb2.ConverterSettings,
               replay_versions: Sequence[str],
               player_min_mmr: int,
               home_race: Optional[sc2_enums.Race],
               away_race: Optional[sc2_enums.Race],
               use_prev_features: bool,
               max_times_sampled: int = 1,
               shuffle_buffer_size: int = 1024,
               extra_replay_filters: Optional[Mapping[str, Any]] = None):
    """Initializes the offline TF Record data source.

    Args:
      unroll_len: Unroll length (sequence length) for the inputs.
      overlap_len: Overlap length between successful sequences.
      batch_size: Batch size for the data.
      data_split: Data split (train or test data)
      converter_settings:  Settings used for the converter that transforms the
        observations and actions.
      replay_versions: Sequence of replay files that are used as part of the
        dataset.
      player_min_mmr: Minimum MMR for the games to be a part of the dataset.
      home_race: Race of the player.
      away_race: Race of the opponent.
      use_prev_features: Whether to add the prev_feature field to the agent
        inputs. They contain the actions taken at the step immediately before
        the first time step of the rollout.
      max_times_sampled: Maximum number of times to sample.
      shuffle_buffer_size: Size of the shuffle buffer for the dataset.
      extra_replay_filters: Map of further filters to be applied on the replay
        data.
    """
    super().__init__(unroll_len=unroll_len,
                     overlap_len=overlap_len,
                     batch_size=batch_size,
                     home_race=home_race,
                     away_race=away_race)
    self._data_split = data_split
    self._converter_settings = converter_settings
    self._player_min_mmr = player_min_mmr
    self._shuffle_buffer_size = shuffle_buffer_size
    self._use_prev_features = use_prev_features

    self._obs_spec, self._action_spec = converted_env.get_environment_spec(
        self._converter_settings)

    if extra_replay_filters:
      # TODO(b/208419046): Add filtering support.
      raise ValueError('Filtering is not supported yet.')

    self._replay_versions = tuple(replay_versions)

    dataset_pattern = paths.get_dataset_pattern(
        self._replay_versions, self._data_split, self._player_min_mmr)
    if dataset_pattern is None:
      raise ValueError(
          f'Dataset not found matching '
          f'replay_versions: {self._replay_versions}, '
          f'data_split: {self._data_split}, '
          f'player_min_mmr: {self._player_min_mmr}'
      )
    self._dataset_pattern = dataset_pattern

  def get_generator(self) -> Generator[types.StreamDict, None, None]:
    """Get generator which yields training batches for the learner."""

    features = data_utils.get_dataset_specs(self._obs_spec)

    episodes_ds = make_episodes_dataset(self._dataset_pattern, features)
    unrolls_ds = make_unrolls_dataset(
        episodes_ds,
        self._batch_size,
        self._unroll_len + (1 if self._use_prev_features else 0),
        shuffle_buffer_size=self._shuffle_buffer_size)
    ds = unrolls_ds.prefetch(1)
    yield from map(functools.partial(data_utils.as_learner_input,
                                     use_prev_features=self._use_prev_features),
                   data_utils.iterate(ds))

  @property
  def env_spec(self) -> Tuple[types.ObsSpec, types.ActionSpec]:
    _, obs_spec = data_utils.split_behaviour_actions(self._obs_spec)
    obs_spec = types.SpecDict(obs_spec)
    action_spec = types.SpecDict(self._action_spec)
    return obs_spec, action_spec

  @property
  def input_spec(self) -> types.SpecDict:
    behaviour_action_spec, obs_spec = data_utils.split_behaviour_actions(
        self._obs_spec)
    behaviour_features_spec = types.SpecDict(dict(action=behaviour_action_spec))
    if self._use_prev_features:
      prev_features_spec = behaviour_features_spec
    else:
      prev_features_spec = None
    return data_utils.get_input_spec(
        obs_spec,
        behaviour_features_spec=behaviour_features_spec,
        prev_features_spec=prev_features_spec)


class DummyDataSource(data_source_base.DataSource):
  """Data source that outputs dummy observations all the time (for testing)."""

  def __init__(
      self,
      unroll_len: int,
      batch_size: int,
      converter_settings: converter_pb2.ConverterSettings,
      **unused_kwargs):
    """Initializes the dummy data source.

    Args:
      unroll_len: Unroll length (sequence length) for the inputs.
      batch_size: Batch size for the data.
      converter_settings:  Settings used for the converter that transforms the
        observations and actions.
    """
    self._unroll_len = unroll_len
    self._batch_size = batch_size
    self._obs_spec, self._action_spec = converted_env.get_environment_spec(
        converter_settings)

  def get_generator(self) -> Generator[types.StreamDict, None, None]:
    dummy_obs = data_utils.get_dummy_observation(
        self.input_spec,
        batch_size=self._batch_size,
        unroll_len=self._unroll_len)
    while True:
      yield dummy_obs

  @property
  def env_spec(self) -> Tuple[types.ObsSpec, types.ActionSpec]:
    _, obs_spec = data_utils.split_behaviour_actions(self._obs_spec)
    obs_spec = types.SpecDict(obs_spec)
    action_spec = types.SpecDict(self._action_spec)
    return obs_spec, action_spec

  @property
  def input_spec(self) -> types.SpecDict:
    behaviour_action_spec, obs_spec = data_utils.split_behaviour_actions(
        self._obs_spec)
    behaviour_features_spec = types.SpecDict(dict(action=behaviour_action_spec))
    return data_utils.get_input_spec(obs_spec, behaviour_features_spec)

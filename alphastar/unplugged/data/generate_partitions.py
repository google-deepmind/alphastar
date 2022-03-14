# Copyright 2022 DeepMind Technologies Limited.
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

"""Generates replay dataset partition files.

Given a directory containing .SC2Replay files and a directory which contains (or
will contain) converted .tfrecord files, determines which replays have yet to be
converted and partitions them, with each partition containing roughly the same
total replay file bytes.

Partitions are represented as text files containing newline-separated replay
hashes. The intention is that generate_dataset.py is instantiated multiple
times, with each instance being passed a separate partition to process.
"""

import collections
import os
from typing import Iterable, List, Sequence

from absl import app
from absl import flags
from absl import logging
import tensorflow as tf

FLAGS = flags.FLAGS

flags.DEFINE_string(
    'sc2_replay_path', None, 'Path to .SC2Replay files.', required=True)
flags.DEFINE_string(
    'converted_path', None, 'Path to .tfrecord files.', required=True)
flags.DEFINE_integer(
    'num_partitions', None, 'Number of partitions to create', required=True)
flags.DEFINE_string(
    'partition_path', None, 'Path to write partition files to.', required=True)


def _enumerate_replays(sc2_replay_path: str,
                       converted_path: str) -> Iterable[str]:
  """Determines which replays remain to be converted.

  Args:
    sc2_replay_path: Source directory containing .SC2Replay files to convert.
    converted_path: Destination directory, possibly containing converted
      versions of replays that have already been processed.

  Returns:
    An iterable of replay hashes that still require conversion.
  """
  src = set(
      f.replace('.SC2Replay', '') for f in tf.io.gfile.listdir(sc2_replay_path))
  if tf.io.gfile.isdir(converted_path):
    counts = collections.Counter(
        f.replace('.tfrecord', '')[:-2]  # -2 strips off player id.
        for f in tf.io.gfile.listdir(converted_path))
    # We expect all replays to be 2 player. Any that don't have a .tfrecord
    # file for each player are marked for conversion.
    dst = set(k for k, v in counts.items() if v == 2)
  else:
    logging.warning('converted_path %s does not exist.', converted_path)
    dst = set()
  return sorted(src - dst)


def _partition_replays_by_size(
    sc2_replay_path: str,
    replay_hashes: List[str],
    num_partitions: int) -> Sequence[List[str]]:
  """Partitions a set of replays, taking into account file sizes.

  Args:
    sc2_replay_path: Source directory containing .SC2Replay files.
    replay_hashes: List of hashes of replays to partition.
    num_partitions: How many partitions to create.

  Yields:
    A sequence of lists of hashes, the original list partitioned such that each
      is of roughly equal total file size.
  """
  sizes = []
  total_size = 0
  for h in replay_hashes:
    name = os.path.join(sc2_replay_path, f'{h}.SC2Replay')
    size = tf.io.gfile.stat(name).length
    sizes.append(size)
  total_size = sum(sizes)

  target_size = total_size // num_partitions
  cumulative_size = 0
  partition = 1
  hashes = []
  for h, size in zip(replay_hashes, sizes):
    hashes.append(h)
    cumulative_size += size
    if cumulative_size // partition >= target_size:
      yield hashes
      hashes.clear()
      partition += 1

  if hashes:
    yield hashes


def _write_partition_files(partitions: Sequence[List[str]],
                           partition_path: str) -> Sequence[str]:
  """Writes a text file for each partition, of newline-separated strings.

  Args:
    partitions: A sequence of string lists.
    partition_path: Directory to write partition files to.

  Yields:
    Sequence of partition filenames written.
  """
  if not tf.io.gfile.isdir(partition_path):
    tf.io.gfile.makedirs(partition_path)
  for i, partition in enumerate(partitions):
    name = os.path.join(partition_path, f'partition_{i}')
    with tf.io.gfile.GFile(name, 'w') as f:
      f.write('\n'.join(map(str, partition)))
    yield name


def main(argv: Sequence[str]) -> None:
  del argv
  for p in _write_partition_files(
      partitions=_partition_replays_by_size(
          sc2_replay_path=FLAGS.sc2_replay_path,
          replay_hashes=_enumerate_replays(
              sc2_replay_path=FLAGS.sc2_replay_path,
              converted_path=FLAGS.converted_path),
          num_partitions=FLAGS.num_partitions),
      partition_path=FLAGS.partition_path):
    logging.info('Created %s.', p)


if __name__ == '__main__':
  app.run(main)

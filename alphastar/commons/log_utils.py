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

"""Logging utility functions."""

import enum
from typing import Callable, Dict, Mapping, Optional

from alphastar.collections import Struct
import chex
import jax
import jax.numpy as jnp


class ReduceType(str, enum.Enum):
  MEAN = "mean"
  MIN = "min"
  MAX = "max"
  NUM = "num"
  SUM = "sum"
  NON_REDUCED = "non_reduced"


# TODO(b/208619590): move Log to types
Log = Dict[str, Dict[ReduceType, chex.Array]]
ReduceFn = Callable[[chex.Array], chex.Array]


REDUCE_FUNCTIONS = Struct({
    ReduceType.MEAN: jnp.nanmean,
    ReduceType.MIN: jnp.nanmin,
    ReduceType.MAX: jnp.nanmax,
    ReduceType.NUM: jnp.nansum,
    ReduceType.SUM: jnp.nansum,
    ReduceType.NON_REDUCED: lambda x: jnp.reshape(x, -1)[0],
})


def reduce_logs(logs: Log,
                reduce_fns: Mapping[ReduceType, ReduceFn] = REDUCE_FUNCTIONS
                ) -> Log:
  """Reduce the logs using the provided set of reduce functions."""
  check_logs_rank(logs)
  reduced_log = {}
  for name, stats in logs.items():
    reduced_log[name] = {}
    for stat, value in stats.items():
      reduced_log[name][stat] = reduce_fns[stat](value)
    # If possible, recompute the mean as total_sum/total_num,
    # it's less biased and has less nans (because we use masking).
    if ReduceType.NUM in stats and ReduceType.SUM in stats:
      stat_num = reduced_log[name][ReduceType.NUM]
      stat_sum = reduced_log[name][ReduceType.SUM]
      reduced_log[name][ReduceType.MEAN] = jnp.where(
          stat_num, stat_sum / stat_num, jnp.nan)
  check_logs_rank(reduced_log, None)
  return reduced_log


def check_logs_rank(logs: Log, expected_rank: Optional[int] = None) -> None:
  """Checks that the logs have the correct rank.

  Logs are expected to be nested dicts representing module/log/metric.

  Args:
    logs: The log nested dictionary.
    expected_rank: The expected rank of each log element. Use None if unknown.
  """
  if logs:
    flat_logs = jax.tree_leaves(logs)
    chex.assert_equal_shape(flat_logs)
    if expected_rank is not None:
      chex.assert_rank(flat_logs, expected_rank)

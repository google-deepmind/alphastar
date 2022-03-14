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

"""Metrics related functions."""

import functools
from typing import Callable, Dict, List, Optional, Tuple

from alphastar.commons import log_utils
import chex
import jax
import jax.numpy as jnp


def pnanreduce(x: jnp.DeviceArray,
               reduce_fn: Callable[..., jnp.DeviceArray],
               axis_name: Optional[str] = None,
               axis_index_groups: Optional[List[List[int]]] = None
               ) -> Tuple[jnp.DeviceArray, jnp.DeviceArray]:
  """Gets the number of non-nan elements reduce them with reduce_fn.

  Args:
    x : Input array
    reduce_fn : Function that is used to reduce the inputs.
    axis_name: Object used to name a pmapped axis
    axis_index_groups: Optional list of lists containing axis indices to do
      the p-reduce operations over.

  Returns:
    Number of non-nan elements, non-nan elements reduced.
  """
  valid_mask = jnp.logical_not(jnp.isnan(x))
  valid_x = jnp.where(valid_mask, x, jnp.zeros_like(x))
  valid_num = jax.lax.psum(
      valid_mask, axis_name=axis_name, axis_index_groups=axis_index_groups)
  valid_reduced = reduce_fn(
      valid_x, axis_name=axis_name, axis_index_groups=axis_index_groups)
  return valid_num, valid_reduced


def pnanmean(x: jnp.DeviceArray,
             axis_name: Optional[str] = None,
             axis_index_groups: Optional[List[List[int]]] = None
             ) -> jnp.DeviceArray:
  """Takes mean over non-nan elements.

  Args:
    x : Input array
    axis_name: Object used to name a pmapped axis
    axis_index_groups: Optional list of lists containing axis indices to do
      the p-reduce operations over.

  Returns:
    Mean of non-nan elements reduced.
  """
  valid_num, valid_sum = pnanreduce(
      x, jax.lax.psum, axis_name, axis_index_groups)
  return jnp.where(valid_num, valid_sum / valid_num, jnp.nan)


def pnantake(x: jnp.DeviceArray,
             reduce_fn: Callable[..., jnp.DeviceArray],
             axis_name: Optional[str] = None,
             axis_index_groups: Optional[List[List[int]]] = None
             ) -> jnp.DeviceArray:
  """Takes a reduction over non-nan elements.

  Args:
    x : Input array
    reduce_fn : Reduction operation to be applied over elements.
    axis_name: Object used to name a pmapped axis
    axis_index_groups: Optional list of lists containing axis indices to do
      the p-reduce operations over.

  Returns:
    Result of reduction operation on non-nan elements.
  """
  valid_num, valid_y = pnanreduce(x, reduce_fn, axis_name, axis_index_groups)
  return jnp.where(valid_num, valid_y, jnp.nan)


P_REDUCE_FUNCTIONS = {
    log_utils.ReduceType.MEAN: pnanmean,
    log_utils.ReduceType.MIN: functools.partial(
        pnantake, reduce_fn=jax.lax.pmin),
    log_utils.ReduceType.MAX: functools.partial(
        pnantake, reduce_fn=jax.lax.pmax),
    log_utils.ReduceType.NUM: functools.partial(
        pnantake, reduce_fn=jax.lax.psum),
    log_utils.ReduceType.SUM: functools.partial(
        pnantake, reduce_fn=jax.lax.psum),
    log_utils.ReduceType.NON_REDUCED: lambda x: x}


def reduce_metrics(
    metrics: log_utils.Log,
    axis_name: Optional[str] = None,
    axis_index_groups: Optional[List[List[int]]] = None,
    local: bool = False,
    ) -> log_utils.Log:
  """Reduce metrics across devices.

  Args:
    metrics : A log object that contains metrics collected.
    axis_name: Object used to name a pmapped axis
    axis_index_groups: Optional list of lists containing axis indices to do
      the p-reduce operations over.
    local: Boolean that says whether metrics are run on a local device or a
      p-mapped set of devices.

  Returns:
    A log object with reduced metrics.

  """
  if axis_name is not None and local:
    raise ValueError('Cannot specify both axis_name and local.')
  elif axis_name is None and axis_index_groups is not None:
    raise ValueError('Must provide axis name when using axis index groups.')

  if local:
    reduce_fns = log_utils.REDUCE_FUNCTIONS
  else:
    reduce_fns = {
        k: functools.partial(
            fn, axis_name=axis_name, axis_index_groups=axis_index_groups)
        for k, fn in P_REDUCE_FUNCTIONS.items()}
  return log_utils.reduce_logs(metrics, reduce_fns)


def flatten_metrics(metrics: log_utils.Log) -> Dict[str, chex.Array]:
  """Flattens metrics to a single level of nesting."""
  output = {}
  for k, v in metrics.items():
    if isinstance(v, dict):
      for k2, v2 in flatten_metrics(v).items():
        output[f'{k}_{k2}'] = v2
    else:
      output[k] = v
  return output

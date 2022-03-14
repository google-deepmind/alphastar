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

"""Functions for using collections in JAX."""

from absl import logging
from alphastar import collections
import jax


def register_struct():
  """Register `structure.Struct` so pytree/tracer knows how to handle it."""
  try:
    jax.tree_util.register_pytree_node(
        collections.Struct,
        flatten_func=lambda s: (tuple(s.values()), tuple(s.keys())),
        unflatten_func=lambda k, xs: collections.Struct(zip(k, xs)))
  except ValueError:
    logging.info('Struct is already registered as JAX PyTree Node.')

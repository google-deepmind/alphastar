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

"""Utility code related to JAX."""

import contextlib

import jax

_PREV_JAX_CONFIG = None


def disable_jax_optimizations():
  global _PREV_JAX_CONFIG
  _PREV_JAX_CONFIG = jax.config.values.copy()
  jax.config.update('jax_disable_most_optimizations', True)


def restore_jax_config():
  if _PREV_JAX_CONFIG:
    jax.config.values.update(**_PREV_JAX_CONFIG)


def _disabled_backend_compile(*args, **kwargs):
  raise RuntimeError('Attempt to compile a JAX program to XLA, but '
                     'compilation is not allowed in this scope. Typically '
                     'this is due to changes in input shapes or types, e.g. '
                     'specs used to generate dummy data not agreeing with the '
                     'actual data. Other cases could be if the program '
                     'contains modules that are always compiled at every call '
                     'if they are not jitted -- for example hk.scan')


# pylint: disable=protected-access
# pylint: disable=attribute-error
def jax_compilation_is_disabled():
  return jax._src.dispatch.backend_compile is _disabled_backend_compile


@contextlib.contextmanager
def no_jax_compilation_allowed():
  """Prevents JAX compilation in the scope of the context."""
  previous_backend_compile = jax._src.dispatch.backend_compile
  jax._src.dispatch.backend_compile = _disabled_backend_compile
  try:
    yield
  finally:
    # Make sure nobody else has patched the same thing in the mean time.
    assert jax_compilation_is_disabled()
    jax._src.dispatch.backend_compile = previous_backend_compile
# pylint: enable=protected-access
# pylint: enable=attribute-error

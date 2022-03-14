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

"""Alphastar architectures."""

import functools
import itertools
import types

from alphastar.architectures import modular
from alphastar.architectures.dummy import dummy
from alphastar.architectures.standard import standard
from alphastar.architectures.standard.configs import full as full_config
from alphastar.architectures.standard.configs import lite as lite_config


ARCHITECTURES = types.MappingProxyType(
    dict(
        alphastar=dict(
            dummy=dict(getter=dummy.get_alphastar_dummy, config=None),
            lite=dict(
                getter=standard.get_alphastar_standard,
                config=lite_config.get_config()),
            full=dict(
                getter=standard.get_alphastar_standard,
                config=full_config.get_config()),
        )))

ARCHITECTURE_NAMES = tuple(
    itertools.chain.from_iterable(
        [[f'{k}.{name}' for name in v] for k, v in ARCHITECTURES.items()]))


def _check_architecture_supported(architecture_name: str):
  """Checks if architecture name is valid."""
  if architecture_name not in ARCHITECTURE_NAMES:
    raise ValueError(f'Unknown architecture {architecture_name}. Architecture '
                     f'name must be in {ARCHITECTURE_NAMES}.')


def get_architecture(architecture_name: str) -> modular.ArchitectureBuilder:
  """Gets an architecture to build based on the architecture name."""
  _check_architecture_supported(architecture_name)
  base, name = architecture_name.split('.', maxsplit=1)
  getter, config = ARCHITECTURES[base][name]['getter'], ARCHITECTURES[base][
      name]['config']
  if config:
    getter = functools.partial(getter, config=config)
  return getter


def is_transformer_arch(arch_str):
  """Checks if an architecture has a Transformer module."""
  return 'transformer' in arch_str


def get_default_config(architecture_name: str):
  """Gets the default architecture for standard architecture configs."""
  _check_architecture_supported(architecture_name)
  base, name = architecture_name.split('.', maxsplit=1)
  return ARCHITECTURES[base][name]['config']

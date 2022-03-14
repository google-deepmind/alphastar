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

"""Immutable dict with sorted keys, access-as-attribute & pretty printing."""

import bisect
from collections import abc
from typing import Mapping, TypeVar


Value = TypeVar("Value")


class Struct(dict, Mapping[str, Value]):
  """Immutable dict with sorted keys, access-as-attribute & pretty printing.

  Inherits from dict explicitly for tensor flow encoding (which uses isinstance
    directly...) and py tree flattening (which is in C++ and ignores method
    overrides).

  Note that the immutability of the dict is valid only when the dict is
  non-empty.

  Note that to override existing fields one can call:
    Struct(existing_struct, existing_1=new_value, existing_2=new_value_2).

  To create new fields, checking that they don't duplicate existing fields:
    Struct(new_1=new_value, new_2=new_value_2, **existing_struct).
  """

  def __init__(self, *args, **kwargs):
    super().__init__(*(_ignore_none(args) if args else []), **kwargs)

    self._sorted_keys = sorted(super(Struct, self).keys())

    # Note that if our dict is empty we set immutable to false. This is
    # a workaround for the fact that tensorflow/python/client/session.py's
    # _DictFetchMapper constructs an empty dict and assigns rather than
    # constructing from a list of key-values. In that case users of this
    # class lose protection against mutation via __setitem__.
    self._immutable = bool(self._sorted_keys)

  def __getattr__(self, name):
    """Permits item-as-attribute lookup."""
    try:
      return self[name]
    except Exception:
      raise AttributeError(name)

  def __iter__(self):
    """Iterates over the sorted keys list."""
    return self._sorted_keys.__iter__()

  def __repr__(self):
    """Pretty-prints the Struct's contents."""
    ret = ["Struct("]
    for k in self:
      v_repr = "\n  ".join(repr(self[k]).split("\n"))
      ret.append("  {}={},".format(k, v_repr))
    ret.append(")")
    return "\n".join(ret)

  def __setattr__(self, k, v):
    """Allows only the setting of our private attributes."""
    if k in ["_sorted_keys", "_immutable"]:
      super(Struct, self).__setattr__(k, v)
    else:
      self._defined_as_read_only()

  def __setitem__(self, k, v):
    """Allows setting items only if the Struct is mutable (see init)."""
    if not self._immutable:
      super(Struct, self).__setitem__(k, v)
      bisect.insort(self._sorted_keys, k)
    else:
      self._defined_as_read_only()

  def __hash__(self):
    """Protects against mutability, *but only in this object*."""
    if not self._immutable:
      raise TypeError("Can only hash immutable Structs")
    return hash(tuple(self.items()))

  def __reduce__(self):
    """Explicit pickling support is required due to the class' evolution.

    Note that code exists which expects the 'None' here (which is superfluous
    now). For the time being, meet those expectations.

    Returns:
      Reduced representation of this Struct instance.
    """
    return self.__class__, (None, list(self.items()),)

  def keys(self):
    """Uses the sorted key list."""
    return self._sorted_keys

  def items(self):
    """Returns a view onto items, sorted by key."""
    return SortedItemsView(self)

  def values(self):
    """Returns a view onto values, sorted by key."""
    return SortedValuesView(self)

  def _defined_as_read_only(self, *args, **kwargs):
    raise RuntimeError("Modifications to Struct instances are not permitted")

  __delitem__ = _defined_as_read_only
  pop = _defined_as_read_only
  popitem = _defined_as_read_only
  clear = _defined_as_read_only
  update = _defined_as_read_only
  setdefault = _defined_as_read_only

  def _not_supported(self, *args, **kwargs):
    raise RuntimeError("Method unsupported by Struct")

  __reversed__ = _not_supported


class SortedItemsView(abc.ItemsView, abc.Sequence):

  def __init__(self, struct: Struct):
    self._struct = struct
    super().__init__(struct)

  def __getitem__(self, i: int):
    key = self._struct._sorted_keys[i]
    return key, self._struct[key]


class SortedValuesView(abc.ValuesView, abc.Sequence):

  def __init__(self, struct: Struct):
    self._struct = struct
    super().__init__(struct)

  def __getitem__(self, i: int):
    key = self._struct._sorted_keys[i]
    return self._struct[key]


def _ignore_none(args):
  """Ignore superfluous None args when reconstructing from copies."""
  if args and args[0] is None:
    args = args[1:]
  return args

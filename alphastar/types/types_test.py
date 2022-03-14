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

"""Tests for types."""

import copy

from absl.testing import absltest
from absl.testing import parameterized
from alphastar import types
import chex
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np
import tree


class EvaluatorTest(parameterized.TestCase):
  """Basic tests for types."""

  def test_nested_dict_access(self):
    x = types.NestedDict()
    self.assertEmpty(x)
    x['a'] = 100
    x['b', 'c'] = 200
    x['b', 'd'] = 300
    with self.assertRaises(IndexError):
      x[()] = 42
    self.assertEqual(x['a'], 100)
    self.assertEqual(x['b', 'c'], 200)
    self.assertSetEqual(set(x.get('b').keys()), set(['c', 'd']))
    self.assertEqual(x.get('b')['d'], 300)
    with self.assertRaises(IndexError):
      _ = x['b']
    with self.assertRaises(KeyError):
      _ = x['c']
    with self.assertRaises(KeyError):
      _ = x['b', 'e']
    self.assertSetEqual(set(x.shallow_keys()), set(['a', 'b']))
    self.assertIn('a', x)
    self.assertIn(('a',), x)
    self.assertIn('b', x)
    self.assertIn(('b', 'c'), x)
    self.assertNotIn(('b', 'c', 'e'), x)
    self.assertNotIn('d', x)
    self.assertNotIn(('e',), x)
    self.assertSetEqual(set(iter(x)),
                        set(['a', ('b', 'c'), ('b', 'd')]))
    self.assertLen(x, 3)
    self.assertDictEqual(x.asdict(), {'a': 100, 'b': {'c': 200, 'd': 300}})
    y = x.copy()
    y['a'] = 500
    y['b', 'c'] = 600
    self.assertEqual(x.get('a'), 100)
    self.assertEqual(x.get(('b', 'c')), 200)
    x['b', 'e'] = 400
    del x['b', 'c']
    self.assertIn(('b', 'e'), x)
    self.assertNotIn(('b', 'c'), x)
    del x['b']
    self.assertSetEqual(set(x.keys()), set(['a']))
    x['z'] = {'a': 1, 'b': 2}
    self.assertSetEqual(set(x.keys()), set(['a', ('z', 'a'), ('z', 'b')]))
    with self.assertRaises(IndexError):
      x['a', 'b'] = 4

  def test_nested_dict_nested_index(self):
    x = types.NestedDict()
    x['a', ('b', 'c')] = 100
    self.assertEqual(x['a', 'b', 'c'], 100)
    self.assertSetEqual(set(x.keys()), set([('a', 'b', 'c')]))
    y = types.NestedDict({('a', ('b', 'c')): 42})
    self.assertDictEqual(y.asdict(), {'a': {'b': {'c': 42}}})

  def test_set_difference(self):
    x = types.NestedDict()
    y = types.NestedDict()
    x['a', ('b', 'c')] = 100
    y['a', ('b', 'c')] = 200
    x['d'] = 100
    y['d'] = 200
    x['e', ('f')] = 112
    y['g', ('h')] = 220
    self.assertSetEqual(set(x), set(x.keys()))
    self.assertSetEqual(set(y), set(y.keys()))
    self.assertNotEmpty(set(x).difference(set(y)))

  def test_nested_dict_filter(self):
    x = types.NestedDict()
    x['a'] = 100
    x['b', 'c'] = 200
    x['b', 'd'] = 300
    x['b', 'e'] = 400
    x['f', 'g'] = 500
    y = x.filter([('a',), ('b', 'c'), 'f'])
    self.assertSetEqual(set(y.keys()), set(['a', ('b', 'c'), ('f', 'g')]))
    y['a'] = 200
    self.assertEqual(x['a'], 100)
    z = x.filter(
        types.NestedDict([('a', 42), (('b', 'd'), 4), (('b', 'e'), 3)]))
    self.assertDictEqual(z.asdict(), {'a': 100, 'b': {'d': 300, 'e': 400}})

  def test_nested_dict_init(self):
    d1 = types.NestedDict(
        [(('a', 'b'), 42), ('b', 3), (('a', 'c'), 4), (('d',), 5)])
    self.assertDictEqual(d1.asdict(), {'b': 3, 'a': {'b': 42, 'c': 4}, 'd': 5})
    d2 = types.NestedDict((('a', 3), (('b', 'c'), 4)))
    self.assertDictEqual(d2.asdict(), {'a': 3, 'b': {'c': 4}})
    d3 = types.NestedDict({'a': 3, 'b': {'c': 4, 'd': 5}})
    self.assertDictEqual(d3.asdict(), {'a': 3, 'b': {'c': 4, 'd': 5}})
    d4 = types.NestedDict({'a': 3, 'b': types.NestedDict((('c', 4), ('d', 5)))})
    self.assertDictEqual(d4.asdict(), {'a': 3, 'b': {'c': 4, 'd': 5}})
    d5 = types.NestedDict({'a': 3, ('b', 'c'): 4, ('b', 'd'): 5, ('e',): 6})
    self.assertDictEqual(d5.asdict(), {'a': 3, 'b': {'c': 4, 'd': 5}, 'e': 6})
    d6 = types.NestedDict(types.NestedDict({'a': 3, 'b': {'c': 4, 'd': 5}}))
    self.assertDictEqual(d6.asdict(), {'a': 3, 'b': {'c': 4, 'd': 5}})

  def test_spec_dict(self):
    spec = types.SpecDict({
        'a': specs.Array((), np.bool_),
        ('b', 'c'): specs.Array((4,), np.float32),
        ('b', 'd'): specs.BoundedArray((), np.int32, 0, 5)})

    data1 = types.NestedDict({
        'a': np.zeros((), np.bool_),
        ('b', 'c'): np.ones((4,), np.float32),
        ('b', 'd'): np.array(1, np.int32)})
    spec.validate(data1)

    data2 = types.NestedDict({
        'a': np.zeros((), np.bool_),
        ('b', 'c'): np.ones((4,), np.float32),
        ('b', 'd'): np.ones((), np.int32),
        ('b', 'e'): np.ones((), np.int32),
        'f': np.zeros((42, 3), np.float32)})
    spec.validate(data2)

    data3 = types.NestedDict({
        'a': np.zeros((), np.int32),
        ('b', 'c'): np.ones((4,), np.float32),
        ('b', 'd'): np.ones((), np.int32)})
    with self.assertRaises(ValueError):
      # Wrong dtype
      spec.validate(data3)

    data4 = types.NestedDict({
        'a': np.zeros((), np.bool_),
        ('b', 'd'): np.ones((), np.int32)})
    with self.assertRaises(ValueError):
      # Missing data
      spec.validate(data4)

    data5 = types.NestedDict({
        'a': np.zeros((), np.bool_),
        ('b', 'c'): np.ones((4,), np.float32),
        ('b', 'd'): np.array(10, np.int32)})
    with self.assertRaises(ValueError):
      # Out of bounds
      spec.validate(data5)
      spec['b', 'd'].validate(np.array((10,)))

    data6 = types.NestedDict({
        'a': np.zeros((), np.bool_),
        ('b', 'c'): np.ones((4,), np.float32),
        ('b', 'd'): np.ones((), np.int32),
        'f': np.zeros((42, 3), np.float32)})
    with self.assertRaises(ValueError):
      # Extra data
      spec.validate(data6, exact_match=True)

    data7 = types.NestedDict({
        'a': np.zeros((4,), np.bool_),
        ('b', 'c'): np.ones((4, 4), np.float32),
        ('b', 'd'): np.ones((4,), np.int32)})
    spec.validate(data7, num_leading_dims_to_ignore=1)

    data8 = types.NestedDict({
        'a': np.zeros((4,), np.bool_),
        ('b', 'c'): np.ones((4, 4), np.float32),
        ('b', 'd'): np.ones((5,), np.int32)})
    with self.assertRaises(ValueError):
      # Leading dimensions not matching
      spec.validate(data8, num_leading_dims_to_ignore=1)

    data9 = types.NestedDict({
        'a': jnp.zeros((), jnp.bool_),
        ('b', 'c'): jnp.ones((4,), jnp.float32),
        ('b', 'd'): jnp.array(1, jnp.int32)})
    spec.validate(data9)

  def test_nested_dict_flatted(self):
    x = types.NestedDict({'a': 4, ('b', 'c'): 5, ('b', 'd'): 6})
    leaf_values, treedef = jax.tree_flatten(x)
    self.assertDictEqual(x.asdict(),
                         jax.tree_unflatten(treedef, leaf_values).asdict())
    self.assertSetEqual(set(jax.tree_leaves(x)), set([4, 5, 6]))
    self.assertDictEqual(jax.tree_map(lambda y: y+1, x).asdict(),
                         {'a': 5, 'b': {'c': 6, 'd': 7}})

  def test_tree_and_chex(self):
    x = types.NestedDict({
        'a': jnp.zeros((50,)),
        'b': {'c': jnp.zeros((50, 11)), 'd': jnp.zeros((50, 11, 134))}
    })
    chex.assert_tree_shape_prefix(x, (50,))
    x_plus_one = types.NestedDict({
        'a': jnp.ones((50,)),
        'b': {'c': jnp.ones((50, 11)), 'd': jnp.ones((50, 11, 134))}
    })
    chex.assert_trees_all_equal(tree.map_structure(lambda y: y+1, x),
                                x_plus_one)
    chex.assert_trees_all_equal(tree.map_structure(lambda y: y+1, x).asdict(),
                                x_plus_one.asdict())

  def test_copy(self):
    x = types.NestedDict({'a': {'b': {'c': 1}}})
    y1 = x.get(('a',)).copy()
    y1['c'] = 2
    self.assertEqual(x['a', 'b', 'c'], 1)
    y2 = copy.copy(x.get(('a',)))
    y2['c'] = 2
    self.assertEqual(x['a', 'b', 'c'], 1)
    y3 = copy.deepcopy(x.get(('a',)))
    y3['b', 'c'] = 3
    self.assertEqual(x['a', 'b', 'c'], 1)


if __name__ == '__main__':
  absltest.main()

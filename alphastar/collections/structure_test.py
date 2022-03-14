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

"""Test for alphastar.collections.structure.py."""

import copy
import pickle

from absl.testing import absltest
from alphastar.collections import structure
import tree


class TestStruct(absltest.TestCase):

  def testPrettyPrinting(self):
    result = str(structure.Struct(a=1, b=structure.Struct(c=2, d=[1, 2, 3])))
    self.assertEqual(
        result,
        "Struct(\n  a=1,\n  b=Struct(\n    c=2,\n    d=[1, 2, 3],\n  ),\n)")

  def testAttributeAccess(self):
    data = structure.Struct(x=1, y=2)
    self.assertEqual(data.x, 1)
    self.assertEqual(data.y, 2)
    self.assertEqual(data["x"], 1)
    self.assertEqual(data["y"], 2)
    with self.assertRaisesRegex(AttributeError, "z"):
      print(data.z)

    with self.assertRaisesRegex(KeyError, "'z'"):
      print(data["z"])

  def testHasAttr(self):
    data = structure.Struct(x=1, y=2)
    self.assertTrue(hasattr(data, "x"))
    self.assertFalse(hasattr(data, "z"))

  def testGetAttr(self):
    data = structure.Struct(x=1, y=2)
    self.assertEqual(getattr(data, "x", None), 1)
    self.assertIsNone(getattr(data, "z", None))

  def testIn(self):
    data = structure.Struct(x=1, y=2)
    self.assertIn("x", data)
    self.assertIn("y", data)
    self.assertNotIn("z", data)

  def testConstructionFromDict(self):
    data_dict = dict(x=1, y=2)
    data = structure.Struct(data_dict)
    data_dict["x"] = 3
    del data_dict["y"]
    self.assertEqual(data["x"], 1)
    self.assertEqual(data["y"], 2)

  def testExtension(self):
    data1 = structure.Struct(x=1, y=2)
    data2 = structure.Struct(data1, z=3)
    self.assertEqual(data2.x, 1)
    self.assertEqual(data2.y, 2)
    self.assertEqual(data2.z, 3)

  def testReduction(self):
    data1 = structure.Struct(x=1, y=2, z=3)
    data2 = structure.Struct({k: v for k, v in data1.items() if k != "z"})
    self.assertEqual(data2.x, 1)
    self.assertEqual(data2.y, 2)
    with self.assertRaisesRegex(AttributeError, "z"):
      print(data2.z)

  def testExtensionWithOverrideAllowed(self):
    data1 = structure.Struct(x=1, y=2)
    data2 = structure.Struct(data1, x=3, z=4)
    self.assertEqual(data2.x, 3)
    self.assertEqual(data2.y, 2)
    self.assertEqual(data2.z, 4)

  def testExtensionWithOverrideDisallowed(self):
    data = structure.Struct(x=1, y=2)
    with self.assertRaisesRegex(TypeError, ".*multiple values.*"):
      structure.Struct(x=3, z=4, **data)

  def testImmutability(self):
    data = structure.Struct(x=1, y=2)
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data["x"] = 5
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.x = 5
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.pop()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.popitem()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.clear()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.update(dict(z=3))
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.setdefault("z", 3)
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      del data["x"]

  def testMutability(self):
    data = structure.Struct()
    data["x"] = 5
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.x = 5
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.pop()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.popitem()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.clear()
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.update(dict(z=3))
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      data.setdefault("z", 3)
    with self.assertRaisesRegex(RuntimeError, "Modifications.*not permitted"):
      del data["x"]

  def testIncrementalBuildingSortsCorrectly(self):
    data = structure.Struct()
    data["c"] = 1
    self.assertEqual(list(data.values()), [1])
    data["b"] = 7
    self.assertEqual(list(data.values()), [7, 1])
    data["d"] = 4
    self.assertEqual(list(data.values()), [7, 1, 4])
    data["a"] = 3
    self.assertEqual(list(data.values()), [3, 7, 1, 4])

  def testIteratesInSortedKeyOrder(self):
    data = structure.Struct(e=3, d=4, f=5, a=1, b=2, c=17)
    self.assertEqual(list(data.keys()), ["a", "b", "c", "d", "e", "f"])
    self.assertEqual(list(data.values()), [1, 2, 17, 4, 3, 5])

  def testFlatten(self):
    data = structure.Struct(e=3, d=4, f=5, a=1, b=2, c=17)
    flattened = tree.flatten(data)
    self.assertEqual(flattened, [1, 2, 17, 4, 3, 5])

  def testReduced(self):
    data = structure.Struct(e=3, d=4)
    reduced = data.__reduce__()
    self.assertEqual(reduced, (structure.Struct, (None, [("d", 4), ("e", 3)])))

  def testShallowCopy(self):
    struct = structure.Struct(a=1, b=dict(c=2))
    copied = copy.copy(struct)
    self.assertEqual(struct, copied)
    struct.b["c"] = 3
    self.assertEqual(struct, copied)

  def testDeepCopy(self):
    struct = structure.Struct(a=1, b=dict(c=2))
    copied = copy.deepcopy(struct)
    self.assertEqual(struct, copied)
    struct.b["c"] = 3
    self.assertNotEqual(struct, copied)

  def testViewGetitem(self):
    data = structure.Struct(e=3, d=4, f=5, a=1, b=2, c=17)
    self.assertEqual(data.values()[2], 17)
    self.assertEqual(data.keys()[2], "c")
    self.assertEqual(data.items()[2], ("c", 17))

  def testPickle(self):
    original = structure.Struct(a=1, b=2.0, c=structure.Struct(d="hi", e=5))
    data = pickle.dumps(original)
    reconstructed = pickle.loads(data)
    self.assertEqual(original, reconstructed)

if __name__ == "__main__":
  absltest.main()

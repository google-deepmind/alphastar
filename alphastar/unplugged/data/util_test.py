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

"""Test for data/util.py."""

from absl.testing import parameterized
from alphastar.unplugged.data import util
import chex
import numpy as np
import tensorflow as tf
import tree


class TFCoderTest(tf.test.TestCase, parameterized.TestCase):

  @parameterized.parameters([
      [False,],
      [True,],
  ])
  def test_tf_coder(self, compress):
    foo = np.asarray([[1., 2., 3.], [4., 5., 6.]], dtype=np.float32)
    bar = np.asarray([1, 2, 3], dtype=np.int32)
    baz = np.asarray(5, dtype=np.int32)
    qux = np.asarray([50., 44.], dtype=np.float32)
    bla = np.random.randint(-100, 100, size=[256, 53, 1])
    blu = np.random.randn(64, 128, 128)

    data = dict(
        foo=foo,
        bar=bar,
        foo_bar=dict(baz=baz, qux=qux),
        bla_blu=dict(bla=bla, blu=blu))
    features = tree.map_structure(
        lambda x: util.FeatureSpec(dtype=x.dtype, shape=x.shape), data)

    coder = util.TFExampleCoder(features=features, compress=compress)
    encoded_feature_str = coder.encode(data)
    decoded_features = coder.decode(encoded_feature_str)
    decoded_features_np = tree.map_structure(lambda x: x.numpy(),
                                             decoded_features)
    chex.assert_trees_all_close(decoded_features_np, data)


if __name__ == '__main__':
  tf.test.main()

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

from absl.testing import absltest
from alphastar.commons import jax_utils
import jax
import jax.numpy as jnp


class JaxUtilsTest(absltest.TestCase):

  def test_no_compilation_allowed(self):
    @jax.jit
    def model1(x):
      return jnp.zeros_like(x)

    @jax.jit
    def model2(x):
      return jnp.ones_like(x)

    self.assertIsNotNone(model1(jnp.array([1, 2, 3])))
    with jax_utils.no_jax_compilation_allowed():
      self.assertIsNotNone(model1(jnp.array([1, 2, 3])))
      with self.assertRaisesRegex(RuntimeError,
                                  'compilation is not allowed in this scope'):
        model1(jnp.array([1]))
      with self.assertRaisesRegex(RuntimeError,
                                  'compilation is not allowed in this scope'):
        model2(jnp.array([1, 2, 3]))


if __name__ == '__main__':
  absltest.main()

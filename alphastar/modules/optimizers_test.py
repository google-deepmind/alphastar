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

"""Tests for optimizers."""

from absl.testing import absltest
from absl.testing import parameterized
from alphastar.modules import optimizers
import chex
import haiku as hk
from jax import test_util as jtu
import jax.numpy as jnp


class OptimizersTest(jtu.JaxTestCase):
  """Optimizer Tests."""

  def get_params(self):
    return hk.data_structures.to_haiku_dict(
        dict(
            module_1=dict(
                layer_norm=jnp.array([1., 2.]), weight=jnp.array([3., 4.]))))

  def get_gradients(self):
    return hk.data_structures.to_haiku_dict(
        dict(
            module_1=dict(
                layer_norm=jnp.array([10., 20.]), weight=jnp.array([30., 40.
                                                                   ]))))

  @parameterized.parameters(
      (optimizers.LearningRateScheduleType.STAIRCASE, True, 0.4, 1.0, 1.0,
       [[-0.0642, -0.1176], [-0.1657, -0.2111]]),
      (optimizers.LearningRateScheduleType.COSINE, True, 0.4, 1.0, 1.0,
       [[-0.1480, -0.2710], [-0.3820, -0.4864]]),
      (optimizers.LearningRateScheduleType.COSINE, False, 0.4, 1.0, 1.0,
       [[-0.1849, -0.2917], [-0.3614, -0.4103]]),
      (optimizers.LearningRateScheduleType.COSINE, False, 0.4, 1.0, 0.03,
       [[-0.1849, -0.2917], [-0.3614, -0.4103]]),
  )
  def test_get_optimizer(self, lr_schedule, use_adamw, weight_decay,
                         before_norm, after_norm, results):
    optimizer, _ = optimizers.get_optimizer(
        num_frames_per_learner_update=8,
        total_num_training_frames=100,
        extra_weight_decay_mask_fn=None,
        weight_decay_filter_out=['layer_norm'],
        learning_rate=1.0,
        learning_rate_schedule_type=lr_schedule,
        lr_frames_before_decay=32,
        lr_num_warmup_frames=16,
        adam_b1=0.99,
        adam_b2=0.99,
        adam_eps=0.5,
        use_adamw=use_adamw,
        weight_decay=weight_decay,
        staircase_lr_drop_factor=0.3,
        before_adam_gradient_clipping_norm=before_norm,
        after_adam_gradient_clipping_norm=after_norm)

    state = optimizer.init(self.get_params())
    for _ in range(8):
      updates, state = optimizer.update(self.get_gradients(), state,
                                        self.get_params())
    chex.assert_trees_all_equal_structs(
        hk.data_structures.to_haiku_dict({
            'module_1': {
                'layer_norm': jnp.array(results[0]),
                'weight': jnp.array(results[1])
            }
        }), updates)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

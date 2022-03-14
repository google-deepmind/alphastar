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

"""Tests for supervised."""

from absl.testing import absltest
from alphastar import types
from alphastar.commons import log_utils
from alphastar.unplugged.losses import supervised
import dm_env
from dm_env import specs
import jax
from jax import test_util as jtu
import jax.numpy as jnp
import numpy as np


class SupervisedTest(jtu.JaxTestCase):

  def test_supervised_loss(self):
    unroll_len = 50
    max_num_selected_units = 10
    burnin_len = 2
    overlap_len = 3
    action_spec = {
        'function': specs.BoundedArray((), np.int32, minimum=0, maximum=10),
        'delay': specs.BoundedArray((), np.int32, minimum=0, maximum=6),
        'queued': specs.BoundedArray((), np.int32, minimum=0, maximum=2),
        'repeat': specs.BoundedArray((), np.int32, minimum=0, maximum=4),
        'unit_tags': specs.BoundedArray(
            (max_num_selected_units,), np.int32, minimum=0, maximum=10),
        'target_unit_tag': specs.BoundedArray(
            (), jnp.int32, minimum=0, maximum=10),
        'world': specs.BoundedArray((), np.int32, minimum=0, maximum=25)}

    action = {k: np.random.randint(v.maximum + 1, size=(unroll_len,) + v.shape)
              for k, v in action_spec.items()}
    logits = {k: np.random.rand(unroll_len, *v.shape, v.maximum + 1,)
              for k, v in action_spec.items()}
    masks = {k: np.random.randint(2, size=v.shape).astype(jnp.bool_)
             for k, v in logits.items()}
    argument_masks = {
        k: np.random.randint(2, size=(unroll_len,)).astype(jnp.bool_)
        for k in action_spec}
    step_type = np.random.randint(3, size=(unroll_len,))

    inputs = {
        'action': action,
        'logits': logits,
        'masks': masks,
        'argument_masks': argument_masks,
        'step_type': step_type
    }

    weights = dict(
        function=40.,
        delay=9.,
        queued=1.,
        repeat=0.1,
        target_unit_tag=30.,
        unit_tags=320.,
        world=11.)
    supervised_loss = supervised.Supervised(
        action_spec=action_spec,
        weights=weights,
        overlap_len=overlap_len,
        burnin_len=burnin_len,
        name='supervised')

    jnp_input = jax.tree_map(jnp.asarray, inputs)
    loss, logs = supervised_loss.loss(types.StreamDict(jnp_input))

    self.assertEqual(
        jnp.sum(loss),
        jnp.sum(logs['[supervised] loss'][log_utils.ReduceType.MEAN]))

    for i in range(unroll_len):
      total_xentropy = np.zeros((), np.float32)
      for arg in action_spec:
        log_prob = np.array(jax.nn.log_softmax(logits[arg][i]))
        xentropy = np.take_along_axis(
            -log_prob, action[arg][i][..., np.newaxis], axis=-1)
        xentropy = np.squeeze(xentropy, axis=-1)
        if arg == 'unit_tags':
          for j in range(xentropy.shape[0]):
            xentropy[j] *= masks[arg][i, j][action[arg][i, j]]
          xentropy = np.mean(xentropy)
        else:
          xentropy *= masks[arg][i][action[arg][i]]
        weight = weights[arg]
        if (argument_masks[arg][i]
            and step_type[i] != int(dm_env.StepType.LAST)
            and (burnin_len <= i < unroll_len - overlap_len)):
          total_xentropy += weight * xentropy
      self.assertArraysAllClose(loss[i], total_xentropy)


if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

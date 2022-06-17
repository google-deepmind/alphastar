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

"""Tests for learner."""

import functools

from absl.testing import absltest
from absl.testing import parameterized
from alphastar.architectures import architectures
from alphastar.commons import jax_utils
from alphastar.modules import optimizers
from alphastar.unplugged import losses
from alphastar.unplugged.configs import alphastar_supervised as expt_config_module
from alphastar.unplugged.data import data_source
from alphastar.unplugged.data import data_source_base
from alphastar.unplugged.modules import learner
import jax


def setUpModule():
  # Disable JAX optimizations in order to speed up compilation.
  jax_utils.disable_jax_optimizations()


def tearDownModule():
  jax_utils.restore_jax_config()


class DistributedAgentLearnerTest(parameterized.TestCase):
  """Simple integration/smoke test for the distributed agent."""

  # Evaluator needs to be tested separately as guitar test.
  @parameterized.parameters('alphastar.dummy', 'alphastar.lite')
  def test_agent_learner(self, architecture):
    expt_config = expt_config_module.get_config(architecture)
    expt_config.train.learner_kwargs.batch_size = 4
    expt_config.train.learner_kwargs.unroll_len = 3
    expt_config.train.learner_kwargs.log_to_csv = False
    expt_config.train.datasource.kwargs.shuffle_buffer_size = 16
    expt_config.train.max_number_of_frames = 96
    expt_config.architecture.name = architecture
    expt_config.train.optimizer_kwargs.lr_frames_before_decay = 12
    expt_config.train.datasource.name = 'DummyDataSource'

    frames_per_step = int(expt_config.train.learner_kwargs.batch_size *
                          expt_config.train.learner_kwargs.unroll_len)

    architecture = architectures.get_architecture(expt_config.architecture.name)
    architecture = functools.partial(architecture,
                                     **expt_config.architecture.kwargs)
    loss = functools.partial(
        getattr(losses,
                expt_config.train.loss.name), **expt_config.train.loss.kwargs)

    optimizer, optimizer_logs_fn = optimizers.get_optimizer(
        num_frames_per_learner_update=frames_per_step,
        total_num_training_frames=expt_config.train.max_number_of_frames,
        **expt_config.train.optimizer_kwargs)

    train_data_source = getattr(data_source, expt_config.train.datasource.name)(
        data_split=data_source_base.DataSplit.DEBUG,
        converter_settings=expt_config.converter_settings.train,
        batch_size=expt_config.train.learner_kwargs.batch_size,
        unroll_len=expt_config.train.learner_kwargs.unroll_len,
        overlap_len=expt_config.train.learner_kwargs.overlap_len,
        **expt_config.train.datasource.kwargs)

    learner_node = learner.SupervisedLearner(
        data_source=train_data_source,
        architecture_builder=architecture,
        loss_builder=loss,
        optimizer=optimizer,
        optimizer_logs_fn=optimizer_logs_fn,
        counter=None,
        logger=None,
        rng_key=jax.random.PRNGKey(22),
        frames_per_step=frames_per_step,
        increment_counts=False,
        **expt_config.train.learner_kwargs)

    for _ in range(2):
      learner_node.step()


if __name__ == '__main__':
  absltest.main()

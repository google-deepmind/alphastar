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

"""Tests for common."""

import functools

from absl.testing import absltest
from alphastar.architectures import architectures
from alphastar.modules import common
from alphastar.modules import optimizers
from alphastar.unplugged import losses
from alphastar.unplugged.configs import alphastar_supervised as expt_config_module
from alphastar.unplugged.data import data_source
from alphastar.unplugged.modules import learner as supervised_learner
import jax
from acme.tf import savers as tf_savers
from jax import test_util as jtu
import chex
import os


class CommonTest(jtu.JaxTestCase):
  """Simple tests for common.py."""

  def test_checkpoint(self):
    expt_config = expt_config_module.get_config('alphastar.dummy')
    expt_config.train.learner_kwargs.batch_size = 4
    expt_config.train.learner_kwargs.unroll_len = 3
    expt_config.train.datasource.kwargs.shuffle_buffer_size = 16
    expt_config.train.max_number_of_frames = 96
    expt_config.architecture.name = 'alphastar.dummy'
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

    train_data_source = data_source.DummyDataSource(
        batch_size=expt_config.train.learner_kwargs.batch_size,
        unroll_len=expt_config.train.learner_kwargs.unroll_len,
        converter_settings=expt_config.converter_settings.train)

    learner = supervised_learner.SupervisedLearner(
        data_source=train_data_source,
        architecture_builder=architecture,
        loss_builder=loss,
        optimizer=optimizer,
        optimizer_logs_fn=optimizer_logs_fn,
        counter=None,
        logger=None,
        rng_key=jax.random.PRNGKey(42),
        frames_per_step=frames_per_step,
        increment_counts=False,
        **expt_config.train.learner_kwargs)
    checkpointer = tf_savers.Checkpointer(
        {'wrapped': learner},
        directory=f'{self.create_tempdir().full_path}/alphastar')
    checkpointer.save(force=True)

    restored_learner = supervised_learner.SupervisedLearner(
        data_source=train_data_source,
        architecture_builder=architecture,
        loss_builder=loss,
        optimizer=optimizer,
        optimizer_logs_fn=optimizer_logs_fn,
        counter=None,
        logger=None,
        rng_key=jax.random.PRNGKey(21),
        frames_per_step=frames_per_step,
        increment_counts=False,
        **expt_config.train.learner_kwargs)
    checkpoint_path = os.path.join(checkpointer.directory, 'ckpt-1')
    common.restore_from_checkpoint(
        restored_learner, checkpoint_path)
    chex.assert_trees_all_close(restored_learner.save(), learner.save())

    ckpt_gen = common.get_checkpoint_generator(checkpointer.directory)
    state, _ = next(ckpt_gen)
    chex.assert_trees_all_close(state, learner.save())

    ckpt_gen_path = common.get_checkpoint_generator_for_path(checkpoint_path)
    state, _ = next(ckpt_gen_path)
    chex.assert_trees_all_close(state, learner.save())

if __name__ == '__main__':
  absltest.main(testLoader=jtu.JaxTestLoader())

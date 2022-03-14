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

r"""An example script that trains an Alphastar agent and saves checkpoints.

The default arguments in the config will run a full-fledged training of an
AlphaStar agent -- it does training, stores checkpoints and logs training
details such as different losses, gradients etc. to CSV by default. This script
does not do any evaluation during training. Please run `scripts/evaluate.py` in
parallel to evaluate from the stored checkpoints. More instructions on how to
run evaluation can be found in the docstring of `scripts/evaluate.py`.

To run training with small batch size over 16 frames (real data consists of
billions of frames) for debugging purposes, run:

```shell
python alphastar/scripts/train.py \
  --config=${PWD}/configs/alphastar_supervised.py:alphastar.dummy \
  --config.train.max_number_of_frames=16 \
  --config.train.learner_kwargs.batch_size=4 \
  --config.train.datasource.kwargs.shuffle_buffer_size=16 \
  --config.train.optimizer_kwargs.lr_frames_before_decay=4 \
  --config.train.learner_kwargs.unroll_len=3 \
  --config.train.datasource.name=DummyDataSource
```

Information about different architecture names can be found in
`architectures/README.md`.

* For full fledged training, adjust the config kwargs accordingly or use the
  defaults provided in the config.
* To warmstart from a checkpoint, set config.train.init_checkpoint_path
* Set config.train.checkpoint_kwargs.add_uid = True for getting the new
  checkpoints in a unique directory. A UID is appended to the directory path
  upon new instantiation. Set this to False if you want to write in a directory
  without the UID (note that this could lead to overriding an existing
  checkpoint if the checkpoint directory already exists.)
"""

import functools

from absl import app
from absl import flags
from absl import logging
from acme.jax import utils
from alphastar.architectures import architectures
from alphastar.modules import common as acme_common
from alphastar.modules import optimizers
from alphastar.unplugged import losses
from alphastar.unplugged.data import data_source
from alphastar.unplugged.data import data_source_base
from alphastar.unplugged.modules import learner
import jax
from ml_collections import config_flags

FLAGS = flags.FLAGS
_CONFIG = config_flags.DEFINE_config_file(
    'config', help_string='Configuration file')


def main(_):
  config = _CONFIG.value
  num_devices = 1  # Number of devices over which training is run.
  frames_per_step = int(config.train.batch_size *
                        config.train.unroll_len) * num_devices
  architecture = architectures.get_architecture(config.architecture.name)
  architecture = functools.partial(architecture,
                                   **config.architecture.kwargs)
  loss = functools.partial(
      getattr(losses,
              config.train.loss.name), **config.train.loss.kwargs)

  optimizer, optimizer_logs_fn = optimizers.get_optimizer(
      num_frames_per_learner_update=frames_per_step,
      total_num_training_frames=config.train.max_number_of_frames,
      **config.train.optimizer_kwargs)

  data_source_kwargs = dict(
      data_split=data_source_base.DataSplit.DEBUG,
      converter_settings=config.converter_settings.train,
      batch_size=config.train.batch_size,
      unroll_len=config.train.unroll_len,
      overlap_len=config.train.overlap_len,
      **config.train.datasource.kwargs)

  train_data_source = getattr(data_source, config.train.datasource.name)(
      **data_source_kwargs)

  logger = acme_common.make_default_logger(
      'learner',
      log_to_csv=config.train.log_to_csv,
      time_delta=config.train.log_every_n_seconds,
      asynchronous=True,
      serialize_fn=utils.fetch_devicearray)

  learner_node = learner.SupervisedLearner(
      data_source=train_data_source,
      architecture_builder=architecture,
      loss_builder=loss,
      optimizer=optimizer,
      optimizer_logs_fn=optimizer_logs_fn,
      logger=logger,
      rng_key=jax.random.PRNGKey(22),
      frames_per_step=frames_per_step,
      increment_counts=False,
      **config.train)

  checkpointed_learner_node = acme_common.FramesLimitedCheckpointingRunner(
      max_number_of_frames=config.train.max_number_of_frames,
      num_frames_per_step=frames_per_step,
      wrapped=learner_node,
      **config.train.checkpoint_kwargs
  )

  checkpointed_learner_node.run()
  logging.info('Finished training. GG!')


if __name__ == '__main__':
  app.run(main)

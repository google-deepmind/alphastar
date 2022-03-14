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

r"""Script to run an evaluator for alphastar.

Running this script sets up an architecture, loads any associated checkpoints
if specified, spins up a game against a bot (with a randomly chosen map and
home+opponent races) at random using the PySC2 environment
wrappers. In this script, we run one full episode for a chosen setting. At the
end of the episode, the game outcome and useful stats are logged to a CSV file
by ACME loggers (in ~/acme). To get noise-free results that can be reported for
comparison across SOTA benchmarks, we need to average results across these maps
and races and to do that, we recommend running the script across several CPU
workers (~50 per setting) with different random seeds (`config.eval.rng_seed`).

We list a few ways in which you could use this script. Information about
different architecture names can be found in architectures/README.md.

To use this script just to check if an evaluator is working correctly,
run it with random parameters for a few steps (This plays a hundred steps only
and not a full episode):

```shell
python alphastar/scripts/evaluate.py \
  --config=${PWD}/configs/alphastar_supervised.py:alphastar.dummy \
  --config.eval.log_to_csv=False \
  --config.eval.max_num_steps=100 \
  --config.eval.evaluator_type='random_params'
```

To run evaluation on an existing checkpoint for one full episode, run:

```shell
python alphastar/scripts/evaluate.py \
  --config=${PWD}/configs/alphastar_supervised.py:<ARCHITECTURE_NAME> \
  --config.eval.eval_checkpoint_path=<EVAL_CHECKPOINT_PATH>
```

where `<ARCHITECTURE_NAME>` is the architecture that the checkpoint was trained
with and `<EVAL_CHECKPOINT_PATH>` is the path to the checkpoint to evaluate.

To run evaluation on the most recent checkpoint existing in a directory (which
is usually done when evaluation is done in parallel to training)

```shell
python alphastar/scripts/evaluate.py \
  --config=${PWD}/configs/alphastar_supervised.py:<ARCHITECTURE_NAME> \
  --config.eval.eval_checkpoint_dir=<EVAL_CHECKPOINT_DIR>
```
"""

import functools

from absl import app
from alphastar.architectures import architectures
from alphastar.loggers import eval_episode_logger
from alphastar.modules import evaluator
from ml_collections import config_flags


_CONFIG = config_flags.DEFINE_config_file(
    'config', help_string='Configuration file')


def main(_):
  config = _CONFIG.value
  architecture = architectures.get_architecture(config.architecture.name)
  architecture = functools.partial(architecture,
                                   **config.architecture.kwargs)

  eval_actor = evaluator.EvalActor(
      architecture=architecture,
      learner_frames_per_step=config.eval.default_learner_frames_per_step,
      episode_logger=eval_episode_logger.EvalEpisodeLogger(
          log_name='eval',
          log_to_csv=config.eval.log_to_csv),
      learner_node=None,
      converter_settings=config.converter_settings.eval,
      competitor_name='very_easy',
      **config.eval)

  eval_actor.run_episode(eval_actor.setup_agent())

if __name__ == '__main__':
  app.run(main)

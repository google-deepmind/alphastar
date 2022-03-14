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

"""Launches and tests the evaluator for a few steps.

The test spins up the evaluator with a v3 architecture and runs just 100 steps
(config.eval.max_num_steps = 100)
of one episode to make sure that the evaluation pipeline is mistake free.
"""

import functools
from absl.testing import absltest
from absl.testing import parameterized
from alphastar.architectures import architectures
from alphastar.loggers import eval_episode_logger
from alphastar.modules import evaluator
from alphastar.unplugged.configs import alphastar_supervised as expt_config_module
import jax


class EvaluatorTest(parameterized.TestCase):
  """Simple integration/smoke test for the ACME Evaluator."""

  # Evaluator needs to be tested separately as guitar test.
  # TODO(b/206426779) : Add a variant with a checkpoint trained in TP with v3.
  @parameterized.parameters(
      ['alphastar.lite', 'EvalActor'],
      ['alphastar.dummy', 'EvalActor'],
      ['alphastar.dummy', 'ThreadedUnbatchedEvalActor'])
  def test_evaluator_with_random_params(self, architecture, evaluator_type):
    expt_config = expt_config_module.get_config(architecture)
    expt_config.eval.max_num_steps = 100

    architecture = architectures.get_architecture(expt_config.architecture.name)
    architecture = functools.partial(architecture,
                                     **expt_config.architecture.kwargs)
    expt_config.eval.eval_checkpoint_dir = None
    expt_config.eval.log_to_csv = False
    expt_config.eval.evaluator_type = 'random_params'
    expt_config.eval.evaluator_name = evaluator_type
    expt_config.eval.num_threads_per_inference_device = 3

    eval_actor = getattr(evaluator, expt_config.eval.evaluator_name)(
        rng=jax.random.PRNGKey(42),
        learner_frames_per_step=expt_config.eval
        .default_learner_frames_per_step,
        architecture=architecture,
        episode_logger=eval_episode_logger.EvalEpisodeLogger(
            log_name='eval',
            log_to_csv=expt_config.eval.log_to_csv),
        learner_node=None,
        converter_settings=expt_config.converter_settings.eval,
        competitor_name='very_easy',
        total_num_episodes_per_thread=1,
        **expt_config.eval)

    if evaluator_type == 'ThreadedUnbatchedEvalActor':
      eval_actor.run()
    else:
      eval_actor.run_episode(eval_actor.setup_agent())

if __name__ == '__main__':
  absltest.main()

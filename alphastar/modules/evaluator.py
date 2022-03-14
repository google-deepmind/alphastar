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


"""Actors used for evaluation."""

# pylint: disable=logging-fstring-interpolation
import concurrent
import contextlib
import re
import sys
import time
import traceback
from typing import Any, Dict, Iterator, Mapping, Optional, Sequence, Tuple, Union

from absl import logging
import acme
from acme.jax import savers
from alphastar import types
from alphastar.architectures import modular
from alphastar.commons import competitors
from alphastar.commons import jax_utils
from alphastar.commons import log_utils
from alphastar.loggers import episode_logger as episode_logger_lib
from alphastar.modules import agent as agent_lib
from alphastar.modules import common as acme_common
from alphastar.modules import evaluator_base
from alphastar.modules import match_generator
from alphastar.modules import run_loop
from alphastar.unplugged.data import util as data_utils
from alphastar.unplugged.modules import learner  # pylint: disable=unused-import
import chex
import dm_env
import haiku as hk
import jax
import numpy as np
from pysc2.env import converted_env
from pysc2.env import sc2_env
from pysc2.env.converter import derive_interface_options
from pysc2.env.converter.proto import converter_pb2
from pysc2.maps import ladder as sc2_ladder
from pysc2.maps import lib as sc2_map_lib
import tensorflow as tf

# Needed to load learner checkpoints:

EvaluatorType = evaluator_base.EvaluatorType


# TODO(b/208422091): Cleanup.
if 'Automaton_v2' not in sc2_map_lib.get_maps():
  # Inject custom battle net maps into pysc2.
  custom_ladder_maps = [
      ('Automaton_v2', 'Ladder2019Season1May', 'AutomatonLE', 2),
      ('CyberForest_v2', 'Ladder2019Season1May', 'CyberForestLE', 2),
      ('PortAleksander_v2', 'Ladder2019Season1May', 'PortAleksanderLE', 2),
      ('KairosJunction_v2', 'Ladder2019Season1May', 'KairosJunctionLE', 2),
      ('KingsCove_v2', 'Ladder2019Season1May', 'KingsCoveLE', 2),
      ('NewRepugnancy_v2', 'Ladder2019Season1May', 'NewRepugnancyLE', 2),
      ('YearZero_v2', 'Ladder2019Season1May', 'YearZeroLE', 2),
  ]

  # PySC2 finds maps by looking at subclasses of its map class.
  # We inject the new maps into the global namespace of this file so that a
  # reference is kept to them.
  for name_, directory_, map_file_, players_ in custom_ladder_maps:
    globals()[name_] = type(
        name_, (sc2_ladder.Ladder,),
        dict(filename=map_file_, directory=directory_, players=players_))


def _tree_has_nan(x):
  return any(jax.tree_leaves(jax.tree_map(lambda y: np.isnan(np.sum(y)), x)))


def _get_ckpt_index(ckpt_path_str: str) -> int:
  if ckpt_path_str is None:
    return 0
  else:
    # checkpoints are of the form {dir}/ckpt-{index}
    return int(ckpt_path_str.split('/')[-1].split('-')[-1])


def _get_checkpoint_generator(
    checkpoint_dir: str,) -> Iterator[Tuple[Mapping[str, Any], int]]:
  """Generator that returns the latest checkpoint, blocks if there are none."""
  # ACME uses TF Checkpoint Manager and hence we need to use the same here to
  # obtain checkpoints and do the necessary indexing.
  directory, subdirectory = re.split(r'/checkpoints/', checkpoint_dir)
  logging.log_first_n(
      logging.INFO, f'Searching for checkpoints in directory : {directory} '
      f'and subdirectory: {subdirectory}', 1)
  latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
  last_index = _get_ckpt_index(latest_checkpoint)

  while last_index == 0:
    logging.info('Waiting for a checkpoint in %s', checkpoint_dir)
    time.sleep(10)
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    last_index = _get_ckpt_index(latest_checkpoint)

  cached_index, cached_state = 0, None
  num_ckpt_restore_attempts = 0

  saveable_learner = acme_common.MockSaveableLearner()
  while True:
    if num_ckpt_restore_attempts > 10:
      raise RuntimeError('Tried restoring checkpoint 10 times. Failing.')
    latest_checkpoint = tf.train.latest_checkpoint(checkpoint_dir)
    last_index = _get_ckpt_index(latest_checkpoint)
    if last_index > cached_index:
      logging.info(f'Found new checkpoint {last_index} in '
                   f'{checkpoint_dir}, reloading')
      # pylint: disable=broad-except
      try:
        acme_common.restore_from_checkpoint(saveable_learner, latest_checkpoint)
        logging.info(f'Restored checkpoint {latest_checkpoint} successfully.')
        # pylint: disable=protected-access
        cached_state = saveable_learner._state
        # pylint: enable=protected-access
        cached_index = last_index
        num_ckpt_restore_attempts = 0
      except Exception as e:
        num_ckpt_restore_attempts += 1
        logging.warning(f'Caught exception while loading checkpoint : {e}. '
                        'Sleeping for 10 seconds and retrying.')
        time.sleep(10)
    else:
      time.sleep(10)
    if cached_state is None:
      raise RuntimeError(f'State from {latest_checkpoint} cannot be None.')
    yield cached_state, cached_index


def _get_checkpoint_generator_for_path(
    checkpoint_path: str,) -> Iterator[Tuple[Mapping[str, Any], int]]:
  """Given a full checkpoint path, generates cached state and index."""
  saveable_learner = acme_common.MockSaveableLearner()
  cached_state = None
  # ACME Checkpoint is of the form dir/subdir/ckpt-<index>
  _, cached_index = re.split(r'/ckpt-', checkpoint_path)

  while True:
    if cached_state is None:
      acme_common.restore_from_checkpoint(saveable_learner, checkpoint_path)
      logging.info(f'Restored checkpoint {checkpoint_path} successfully.')
      # pylint: disable=protected-access
      cached_state = saveable_learner._state
      if cached_state is None:
        raise RuntimeError(f'State from {checkpoint_path} cannot be None.')
    yield cached_state, int(cached_index)


class StepEvaluationMixin(object):
  """A mixin that steps an agent with a model and a params getter."""

  def __init__(
      self,
      agent: agent_lib.AlphaStarAgent,
      rng: chex.PRNGKey,
      output_features: Sequence[str] = ('action',),
      warmup_agent: bool = True,
      # Try not to set this to False as compilation is costly.
      prohibit_recompilation: bool = True
  ):
    """Initializes a one-step evaluation mixin.

    Args:
        agent : An Alphastar agent object.
        rng : A jax random number generator.
        output_features : Sequence of output features for the agent.
        warmup_agent: Boolean to decide if an agent needs a warmup step.
        prohibit_recompilation : Boolean to decide whether to prohibit any
          recompilation of the model.
    """
    self._agent = agent
    self._rng = rng
    self._expand_fn = lambda x: np.expand_dims(x, axis=(0, 1))
    self._output_features = output_features
    self._state = None
    self._warmup_params = None
    self._prohibit_recompilation = prohibit_recompilation
    if warmup_agent:
      self._warmup_params = agent.warmup()

  def set_warmup_params(self, params):
    self._warmup_params = params

  def _step(
      self, timestep: dm_env.TimeStep,
      agent_params: hk.Params,
  ) -> Tuple[types.StreamDict, log_utils.Log]:
    """Step through the environment for one step."""
    agent_obs = types.StreamDict()
    agent_obs['step_type'] = np.array([[timestep.step_type]], dtype=np.int32)
    agent_obs['observation'] = types.StreamDict(
        jax.tree_map(self._expand_fn, timestep.observation))
    if timestep.step_type == dm_env.StepType.FIRST:
      self._rng, rng = jax.random.split(self._rng)
      self._state = self._agent.initial_state(rng, batch_size=1)
    self._rng, rng = jax.random.split(self._rng)
    # Get only online params for inference.
    online_params = hk.data_structures.filter(
        lambda module, name, value: 'target_network' not in module,
        agent_params)

    chex.assert_trees_all_equal_shapes(online_params, self._warmup_params)
    with jax_utils.no_jax_compilation_allowed(
    ) if self._prohibit_recompilation else contextlib.suppress():
      output, self._state, logs = self._agent.apply(online_params, rng,
                                                    agent_obs, self._state)
    for k, v in output.items():
      if _tree_has_nan(v):
        logging.info('Output[%s]: %s', k, v)
        raise ValueError(f'Architecture output {k} has NaNs.')
    filtered_output = output.filter(self._output_features)
    squeeze_batch_and_time = lambda x: np.squeeze(x, axis=(0, 1))
    filtered_output = jax.tree_map(squeeze_batch_and_time, filtered_output)
    return filtered_output, logs


class CheckpointEvaluator(evaluator_base.Evaluator, StepEvaluationMixin):
  """Performs inference step based on a model checkpoint."""

  def __init__(
      self,
      agent: agent_lib.AlphaStarAgent,
      checkpoint_generator: Iterator[Tuple[Any, int]],
      learner_frames_per_step: int,
      rng: Optional[chex.PRNGKey] = None,
      warmup_agent: bool = True,
      prohibit_recompilation: bool = True,
      output_features: Sequence[str] = ('action',),
  ):
    """Initializes an evaluator that evaluates a checkpoint.

    Args:
      agent : An Alphastar agent object.
      checkpoint_generator : An iterator for checkpoint state and checkpoint
        index.
      learner_frames_per_step : Number of frames per step of training used in
        the learner.
      rng : A jax random number generator.
      warmup_agent: Boolean to decide if an agent needs a warmup step.
      prohibit_recompilation : Boolean to decide whether to prohibit any
        recompilation of the model.
      output_features : Sequence of output features for the agent.
    """
    super().__init__(
        agent=agent,
        rng=rng,
        warmup_agent=warmup_agent,
        prohibit_recompilation=prohibit_recompilation,
        output_features=output_features)
    self._checkpoint_generator = checkpoint_generator
    self._checkpoint_state = None
    self._checkpoint_index = None
    self._learner_frames_per_step = learner_frames_per_step

  def reset(self) -> Dict[str, chex.Array]:
    self._checkpoint_state, checkpoint_index = next(self._checkpoint_generator)
    if _tree_has_nan(self._checkpoint_state.params):
      raise ValueError(
          f'NaN found in checkpoint parameters (index={checkpoint_index}).')
    return dict(
        checkpoint_index=checkpoint_index,
        learner_step=int(self._checkpoint_state.step),
        home_steps=int(self._checkpoint_state.step) *
        self._learner_frames_per_step)

  def _get_params(self):
    if self._checkpoint_state is None:
      raise RuntimeError('Params retrieval called before checkpoint is ready.')
    return self._checkpoint_state.params

  def step(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[types.StreamDict, log_utils.Log]:
    return self._step(timestep, self._get_params())


class RandomParamsEvaluator(evaluator_base.Evaluator, StepEvaluationMixin):
  """Performs inference step based on randomly initialized model params."""

  def __init__(
      self,
      agent: agent_lib.AlphaStarAgent,
      learner_frames_per_step: int,
      rng: Optional[chex.PRNGKey] = None,
      warmup_agent: bool = True,
      prohibit_recompilation: bool = True,
      output_features: Sequence[str] = ('action',),
  ):
    """Initializes an evaluator that uses random params.

    Args:
      agent : An Alphastar agent object.
      learner_frames_per_step : Number of frames per step of training used in
        the learner.
      rng : A jax random number generator.
      warmup_agent: Boolean to decide if an agent needs a warmup step.
      prohibit_recompilation : Boolean to decide whether to prohibit any
        recompilation of the model.
      output_features : Sequence of output features for the agent.
    """
    super().__init__(
        agent=agent,
        rng=rng,
        warmup_agent=warmup_agent,
        prohibit_recompilation=prohibit_recompilation,
        output_features=output_features)
    self._learner_frames_per_step = learner_frames_per_step

  def reset(self) -> Dict[str, chex.Array]:
    return dict(checkpoint_index=-1, learner_step=-1, home_steps=-1)

  def _get_params(self):
    if self._warmup_params is None:
      raise RuntimeError('Params retrieval called before warmup')
    return self._warmup_params

  def step(
      self, timestep: dm_env.TimeStep
  ) -> Tuple[types.StreamDict, log_utils.Log]:
    return self._step(timestep, self._get_params())


def _make_environment_factory(game_steps_per_episode: int):
  """Returns a factory for agent vs built-in bot environments."""

  def _environment_factory(
      my_race: sc2_env.Race,
      my_converter_settings: converter_pb2.ConverterSettings, opponent: str,
      opponent_race: sc2_env.Race, map_name: str):

    return converted_env.make_streams(
        converted_env.ConvertedEnvironment(
            env=sc2_env.SC2Env(
                map_name=map_name,
                players=[
                    sc2_env.Agent(my_race),
                    sc2_env.Bot(opponent_race,
                                competitors.difficulty_string_to_enum(opponent))
                ],
                game_steps_per_episode=game_steps_per_episode,
                agent_interface_format=derive_interface_options.from_settings(
                    my_converter_settings)),
            converter_factories=converted_env.make_converter_factories(
                [my_converter_settings])))[0]

  return _environment_factory


class EvalActor(acme.Worker):
  """Evaluation actor used for evaluating a given agent."""

  def __init__(
      self,
      home_races: Sequence[sc2_env.Race],
      away_races: Sequence[sc2_env.Race],
      map_names: Sequence[str],
      competitor_name: str,
      converter_settings: converter_pb2.ConverterSettings,
      architecture: modular.ArchitectureBuilder,
      episode_logger: episode_logger_lib.EpisodeLogger,
      learner_frames_per_step: int,
      learner_node: Optional[savers.CheckpointingRunner] = None,
      eval_checkpoint_dir: Optional[str] = None,
      eval_checkpoint_path: Optional[str] = None,
      agent_output_features: Sequence[str] = ('action',),
      game_steps_per_episode: Optional[int] = None,
      rng: Optional[chex.PRNGKey] = None,
      rng_seed: Optional[int] = None,
      max_num_steps: Optional[int] = None,
      evaluator_type: Union[str, EvaluatorType] = EvaluatorType.CHECKPOINT,
      warmup_agent: bool = True,
      prohibit_recompilation: bool = True,
      **unused_kwargs):
    """Initializes the evaluation actor.

    Args:
      home_races : Sequence of SC2 races that can be used for our agent to be
        selected randomly.
      away_races : Sequence of SC2 races that can be used for the opponent to
        be selected randomly.
      map_names : Sequence of SC2 maps that are used randomly for the game.
      competitor_name : Name of the competitor -- for example, very_easy
      converter_settings : Settings used for the converter that transforms the
        observations and actions.
      architecture : Architecture used in the agent.
      episode_logger : Logger used to log episode stats and other evaluation
        metrics.
      learner_frames_per_step : Number of learner frames that are used per step
        of training in the learner.
      learner_node : Learner object(node) that can be used to query the
        checkpoint directory to evaluate from.
      eval_checkpoint_dir : Directory to evaluate checkpoints from. The most
        recent checkpoint is evaluated if this field is set and it supersedes
        the checkpoint directory obtained from the learner node.
      eval_checkpoint_path : Checkpoint path to evaluate. This supersedes all
        other checkpoint related path information.
      agent_output_features : Sequence of features that the agent needs to
        output during evaluation.
      game_steps_per_episode : Number of games steps per episode set as part
        of initializing the environment.
      rng: A jax Random number generator key.
      rng_seed : Seed used to create the random number generator. This is used
        incase a rng is not passed directly.
      max_num_steps: Number of steps per episode to run the evaluation for.
      evaluator_type: Type of the evaluator used. Currently supports checkpoint
        evaluator and random params evaluator.
      warmup_agent : Boolean to decide if we need to warm-up the agent.
      prohibit_recompilation : Boolean to decide if we need to prohibit any
        recompilation of the model graph.
    """

    if isinstance(evaluator_type, str):
      evaluator_type = getattr(EvaluatorType, evaluator_type.upper())

    self._opponent_name = competitor_name
    self._match_generator = match_generator.MatchGenerator(
        home_races=home_races, away_races=away_races, map_names=map_names)

    self._episode_logger = episode_logger
    self._converter_settings = converter_settings
    self._mmr = self._converter_settings.mmr
    self._learner_frames_per_step = learner_frames_per_step

    self._eval_checkpoint_dir = eval_checkpoint_dir
    self._eval_checkpoint_path = eval_checkpoint_path
    self._checkpointed_learner = learner_node

    self._agent_output_features = agent_output_features
    self._game_steps_per_episode = game_steps_per_episode

    self._architecture = architecture

    if rng is None:
      rng = jax.random.PRNGKey(rng_seed)
    self._rng = rng

    self._environment_factory = _make_environment_factory(
        game_steps_per_episode=game_steps_per_episode)
    self._max_num_steps = max_num_steps
    self._evaluator_type = evaluator_type
    self._warmup_agent = warmup_agent
    self._prohibit_recompilation = prohibit_recompilation
    self._checkpoint_generator = None

  @property
  def evaluator_type(self):
    return self._evaluator_type

  def get_agent_architecture(self):
    obs_spec, action_spec = converted_env.get_environment_spec(
        self._converter_settings)
    input_spec = data_utils.get_input_spec(obs_spec)  # pytype: disable=wrong-arg-types  # strict_namedtuple_checks
    return self._architecture(input_spec, action_spec, False)

  def build_evaluator(
      self, agent, checkpoint_generator: Optional[Iterator[Tuple[Any, int]]]
  ) -> evaluator_base.Evaluator:
    if self._evaluator_type == EvaluatorType.CHECKPOINT:
      evaluator = CheckpointEvaluator(
          agent=agent,
          rng=self._rng,
          learner_frames_per_step=self._learner_frames_per_step,
          checkpoint_generator=checkpoint_generator,
          warmup_agent=self._warmup_agent,
          prohibit_recompilation=self._prohibit_recompilation,
          output_features=self._agent_output_features)
    elif self._evaluator_type == EvaluatorType.RANDOM_PARAMS:
      evaluator = RandomParamsEvaluator(
          agent=agent,
          rng=self._rng,
          warmup_agent=self._warmup_agent,
          prohibit_recompilation=self._prohibit_recompilation,
          learner_frames_per_step=self._learner_frames_per_step,
          output_features=self._agent_output_features)

    return evaluator

  def build_checkpoint_generator(self):
    if self.evaluator_type == EvaluatorType.RANDOM_PARAMS:
      return None
    elif self._eval_checkpoint_path:
      return _get_checkpoint_generator_for_path(self._eval_checkpoint_path)
    else:
      eval_checkpoint_dir = self._eval_checkpoint_dir
      if not eval_checkpoint_dir and self._checkpointed_learner:
        eval_checkpoint_dir = self._checkpointed_learner.get_directory()
        if eval_checkpoint_dir is None:
          raise ValueError('Checkpoint directory cannot be ''None')
      logging.info(
          f'Eval Checkpoint directory is set as {eval_checkpoint_dir}')
      return _get_checkpoint_generator(eval_checkpoint_dir)

  def setup_agent(self):
    agent_architecture = self.get_agent_architecture()
    agent = agent_lib.AlphaStarAgent(agent_architecture)
    checkpoint_generator = self.build_checkpoint_generator()
    return self.build_evaluator(agent, checkpoint_generator)

  def run(self) -> None:
    eval_agent = self.setup_agent()
    while True:
      self.run_episode(eval_agent)

  def run_episode(self, agent: evaluator_base.Evaluator):
    home_race, away_race, map_name = self._match_generator.generate(
        self._opponent_name)
    agent_info = agent.reset()
    home_static_logs = dict(
        agent_info,
        competitor_name=self._opponent_name,
        mmr=self._mmr,
        map_name=map_name,
        home_race=home_race.name,
        away_race=away_race.name,
        competitor_type='bot',
        actor_type=(f'{self._opponent_name}:{self._mmr}:{map_name}:'
                    f'{home_race.name}_v_{away_race.name}'))

    with self._environment_factory(
        my_race=home_race,
        my_converter_settings=self._converter_settings,
        opponent=self._opponent_name,
        opponent_race=away_race,
        map_name=map_name) as env:
      run_loop.play_episode(
          env=env,
          agent=agent,
          player=0,
          episode_logger=self._episode_logger,
          static_log=home_static_logs,
          max_num_steps=self._max_num_steps)


def _traceback_exception():
  return traceback.print_exception(*sys.exc_info())

class ThreadedUnbatchedEvalActor(acme.Worker):
  """Multiple actor threads on a CPU with unbatched evaluation."""

  def __init__(self,
               num_threads_per_inference_device: int,
               competitor_name: str,
               competitor_names: Sequence[str],
               use_warmup: bool = False,
               total_num_episodes_per_thread: Optional[int] = None,
               **eval_actor_kwargs):
    """Initializes a threaded unbatched evaluation actor.

    Args:
      num_threads_per_inference_device : Number of actor threads per inference
        device.
      competitor_name: Name of competitor (unused and kept only for
        interface compatibility between different evaluator actors.)
      competitor_names : Sequence of competitor names to choose from for each
        actor thread.
      use_warmup : Boolean to decide if actor thread needs to be warmed up.
      total_num_episodes_per_thread : Number of episodes to be run on
        each actor thread.
      **eval_actor_kwargs : Keyword args passed on to each `EvalActor` thread.
    """
    del competitor_name
    self._num_agents = jax.device_count()
    self._eval_actors = []
    self._agents = []
    self._total_num_episodes_per_thread = total_num_episodes_per_thread
    self._num_evaluation_threads = int(
        num_threads_per_inference_device * self._num_agents)
    num_competitors = len(competitor_names)
    for evaluator_id in range(self._num_evaluation_threads):
      self._eval_actors.append(
          EvalActor(
              **eval_actor_kwargs,
              competitor_name=competitor_names[evaluator_id % num_competitors],
              warmup_agent=False,
              prohibit_recompilation=False))

    architecture = self._eval_actors[0].get_agent_architecture()

    # JIT a model on each device core.
    for device in jax.devices():
      self._agents.append(
          agent_lib.AlphaStarAgent(
              architecture, jit_device=device))
    logging.info(f'Jitted {self._num_agents} agents -- one per core.')
    self._warmup_params = None
    self._warmup_all_agents()
    logging.info('All cores warmed up. Agents ready for inference.')
    self._checkpoint_generator = None

  def _warmup_all_agents(self):
    warmup_fn = lambda agent: agent.warmup()
    future_map = {}
    with concurrent.futures.ThreadPoolExecutor(self._num_agents) as executor:
      for agent_num, agent in enumerate(self._agents):
        future_map[agent_num] = executor.submit(warmup_fn, agent)
      results = {}

      for agent_id, future in future_map.items():
        try:
          results[agent_id] = future.result()
        # pylint: disable=broad-except
        except Exception as e:
          logging.info(f'Error occurred in warmup for agent id {agent_id}: {e}')
        # pylint: enable=broad-except
    self._warmup_params = results[0]

  def _play_agents(self, actor_id):
    """Plays agents on an episode loop until termination."""

    # Assignment is done such that task 0 goes to core 0, task 1 to core 1
    # and so on. Tasks are executed in the order which they are submitted.
    eval_actor = self._eval_actors[actor_id]
    device_id = actor_id % self._num_agents
    agent = self._agents[device_id]
    logging.info(f'Setting up agent of actor {actor_id} on device {device_id}')

    if eval_actor.evaluator_type == EvaluatorType.CHECKPOINT:
      if self._checkpoint_generator is None:
        raise ValueError('Checkpoint generator cannot be None.')

    # Use same checkpoint generator across different threads. Guarantee thread
    # safety with locked iterators.
    eval_agent = eval_actor.build_evaluator(agent, self._checkpoint_generator)

    # Since we disabled warmup when the evaluator was built,
    # we are allowing for warmup params to be set.
    eval_agent.set_warmup_params(self._warmup_params)
    episode_count = 0

    while True:
      logging.info(f'Running new episode on actor {actor_id}')
      try:
        eval_actor.run_episode(eval_agent)
        episode_count += 1
        if episode_count > self._total_num_episodes_per_thread:
          break
      # pylint: disable=broad-except
      except Exception:
        logging.error(
            'Error occurred while running the episode on actor '
            f'{actor_id}:: {_traceback_exception()}')

  def _set_checkpoint_generator(self):
    generator = self._eval_actors[0].build_checkpoint_generator()
    # Use checkpoint generator as a locked iterator to make sure it is
    # thread-safe when multiple actor threads query it.
    if generator:
      self._checkpoint_generator = acme_common.LockedIterator(generator)

  def run(self):
    future_map = {}
    self._set_checkpoint_generator()
    with concurrent.futures.ThreadPoolExecutor(
        self._num_evaluation_threads) as executor:
      for thread_id in range(self._num_evaluation_threads):
        future = executor.submit(self._play_agents, thread_id)
        future_map[thread_id] = future

      for thread_id, future in future_map.items():
        try:
          _ = future.result()
        # pylint: disable=broad-except
        except Exception:
          logging.error(
              f'Error in thread {thread_id} :: '
              f' {_traceback_exception()}')

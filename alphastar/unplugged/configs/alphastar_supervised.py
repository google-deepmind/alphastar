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
"""Configuration for the supervised StarCraft II JaxCraft pipeline."""

import datetime
from typing import Optional

from alphastar.architectures import architectures as arch_util
from alphastar.modules import optimizers
import ml_collections
from pysc2.env.converter.proto import converter_pb2

from s2clientprotocol import common_pb2

_EVAL_MAP_NAMES = ("KairosJunction", "KingsCove", "CyberForest",
                   "NewRepugnancy")

_EVAL_COMPETITORS = (
    # Built-in AI
    "very_easy",
    "very_hard",
)

_DEFAULT_REPLAY_VERSIONS = ("4.8.2", "4.8.3", "4.8.4", "4.8.6", "4.9.0",
                            "4.9.1", "4.9.2",)

# These parameters are only used internally in this config:
_NUM_UPGRADES = 90  # v3
_NUM_BUFFS = 46  # v2
_USE_CAMERA = True
_NUM_UNIT_FEATURES = 46  # v7
_USE_PLACEHOLDER = True
_SHOW_BURROWED_SHADOWS = True
_MAX_NUM_SELECTED_UNITS = 64
_SCREEN_DIM = 1
_MINIMAP_DIM = 128
_WORLD_DIM = 256

# These parameters are used outside:
NUM_UNIT_TYPES = 243  # v4
NUM_RAW_FUNCTIONS = 556  # v4


def get_converter_settings(
    use_supervised: bool,) -> converter_pb2.ConverterSettings:
  return converter_pb2.ConverterSettings(
      raw_settings=converter_pb2.ConverterSettings.RawSettings(
          resolution=common_pb2.Size2DI(x=_WORLD_DIM, y=_WORLD_DIM),
          max_unit_count=512,
          num_unit_features=_NUM_UNIT_FEATURES,
          max_unit_selection_size=_MAX_NUM_SELECTED_UNITS,
          shuffle_unit_tags=True,
          enable_action_repeat=True,
          use_camera_position=_USE_CAMERA,
          camera=_USE_CAMERA,
          use_virtual_camera=_USE_CAMERA,
          virtual_camera_dimensions=converter_pb2.ConverterSettings.RawSettings
          .CameraDimensions(left=16, right=16, top=13, bottom=7),
          add_effects_to_units=True,
          add_cargo_to_units=True,
          mask_offscreen_enemies=_USE_CAMERA),
      minimap=common_pb2.Size2DI(x=_MINIMAP_DIM, y=_MINIMAP_DIM),
      minimap_features=[
          "height_map", "visibility_map", "creep", "player_relative", "alerts",
          "pathable", "buildable"
      ],
      num_action_types=NUM_RAW_FUNCTIONS,
      num_unit_types=NUM_UNIT_TYPES,
      num_upgrade_types=_NUM_UPGRADES,
      max_num_upgrades=40,
      camera_width_world_units=24,
      mmr=6000,
      supervised=use_supervised,
      crop_to_playable_area=False)


SUPERVISED_CONVERTER_SETTINGS = get_converter_settings(use_supervised=True)
EVAL_CONVERTER_SETTINGS = get_converter_settings(use_supervised=False)


def get_config(arch_str: str) -> ml_collections.ConfigDict:
  """Sets up base config which can be overridden."""
  config = ml_collections.ConfigDict()

  # EVALUATION SETTINGS
  config.eval = ml_collections.ConfigDict()
  # We use common_pb2 instead of sc2_env to avoid depending on pygame.
  config.eval.home_races = [
      common_pb2.Protoss, common_pb2.Terran, common_pb2.Zerg
  ]
  config.eval.away_races = [
      common_pb2.Protoss, common_pb2.Terran, common_pb2.Zerg
  ]
  config.eval.genrl_env_prob = 0.5
  config.eval.map_names = _EVAL_MAP_NAMES
  config.eval.competitor_names = _EVAL_COMPETITORS
  # Specify the directory and the latest checkpoint from the directory will be
  # evaluated.
  config.eval.eval_checkpoint_dir = ""
  config.eval.eval_checkpoint_path = ""
  # Number of steps to run the eval module for each episode. Use for debugging.
  config.eval.max_num_steps: Optional[int] = 1_000_000
  config.eval.log_to_csv: bool = True
  config.eval.evaluator_name: str = "EvalActor"
  config.eval.evaluator_type: str = "checkpoint"
  config.eval.num_threads_per_inference_device: int = 1
  # Use this to set number of learner frames per learner step. Only used for
  # logging purposes.
  config.eval.default_learner_frames_per_step: int = 1
  config.eval.rng_seed: int = 42

  # TRAINING SETTINGS
  config.train = ml_collections.ConfigDict()
  config.train.learner_kwargs = ml_collections.ConfigDict(
      dict(
          unroll_len=1,
          overlap_len=0,
          batch_size=1024,
          log_every_n_seconds=60,
          reduce_metrics_all_devices=True,
          log_to_csv=True))

  config.train.max_number_of_frames: float = 1e10
  config.train.init_checkpoint_path: Optional[str] = None

  # All valid Kwargs for Checkpointer() in acme/tf/savers.py
  config.train.checkpoint_kwargs = ml_collections.ConfigDict(
      dict(
          subdirectory="learner",
          checkpoint_ttl_seconds=int(
              datetime.timedelta(days=90).total_seconds()),
          time_delta_minutes=5,
          add_uid=True,
          max_to_keep=5))

  config.train.datasource = ml_collections.ConfigDict(
      dict(
          name="OfflineTFRecordDataSource",
          kwargs=dict(
              replay_versions=_DEFAULT_REPLAY_VERSIONS,
              player_min_mmr=3500,
              # This file is dynamically imported during training using
              # importlib.
              dataset_paths_fname="",
              home_race=None,
              away_race=None,
              use_prev_features=True,
              shuffle_buffer_size=1024,
              extra_replay_filters=dict())),
      convert_dict=True)

  config.train.optimizer_kwargs = ml_collections.ConfigDict(
      dict(
          extra_weight_decay_mask_fn=None,
          learning_rate=5e-4,
          learning_rate_schedule_type=optimizers.LearningRateScheduleType
          .COSINE,
          lr_frames_before_decay=0,
          lr_num_warmup_frames=0,
          adam_b1=0.9,
          adam_b2=0.98,
          adam_eps=1e-8,
          weight_decay=1e-5,
          use_adamw=False,
          before_adam_gradient_clipping_norm=10.0,
          after_adam_gradient_clipping_norm=None,
          weight_decay_filter_out=[],
          staircase_lr_drop_factor=0.2))

  config.train.loss = ml_collections.ConfigDict(
      dict(
          name="Supervised",
          kwargs=dict(
              weights=dict(
                  function=40.,
                  delay=9.,
                  queued=1.,
                  repeat=0.1,
                  target_unit_tag=30.,
                  unit_tags=320.,
                  world=11.),
              burnin_len=0,
              overlap_len=0,
              name="supervised_loss")))

  # ARCHITECTURE SETTINGS
  config.architecture = ml_collections.ConfigDict(
      dict(
          name=arch_str,
          kwargs=ml_collections.ConfigDict(dict(overlap_len=0, burnin_len=0))))

  # TODO(b/207760816) : Make arch config shorter to call for ease of use.
  config.architecture.kwargs.config = arch_util.get_default_config(arch_str)

  config.converter_settings = ml_collections.ConfigDict(
      dict(train=SUPERVISED_CONVERTER_SETTINGS, eval=EVAL_CONVERTER_SETTINGS))
  return config

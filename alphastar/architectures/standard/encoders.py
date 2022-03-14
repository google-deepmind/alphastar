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

"""Encoder blocks."""

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures import util
from alphastar.architectures.components import common
from alphastar.architectures.components import merge
from alphastar.architectures.components import units
from alphastar.architectures.components import vector
from alphastar.architectures.components import visual
import jax.numpy as jnp
import ml_collections


def _get_prev_action_encoder(action_spec: types.ActionSpec,
                             argument_name: types.ArgumentName,
                             is_training: bool,
                             vector_stream_size: int,
                             ) -> modular.Component:
  """Gets encoder for a single previous (scalar) action."""
  component = modular.SequentialComponent(
      name=f"prev_{argument_name}_encoder")
  component.append(common.FeatureFromPrevState(
      name=f"{argument_name}_from_prev_state",
      input_name=("action", argument_name),
      output_name=("prev_action", argument_name),
      is_training=is_training,
      stream_shape=(),
      stream_dtype=jnp.int32))
  if argument_name == util.Argument.FUNCTION:
    component.append(common.ArgumentMasks(
        name="argument_masks",
        input_name=("prev_action", util.Argument.FUNCTION),
        output_name="prev_argument_masks",
        action_spec=action_spec))
  component.append(vector.Embedding(
      name=f"prev_{argument_name}_embedding",
      input_name=("prev_action", argument_name),
      output_name=("prev_action_embeddings", argument_name),
      num_classes=action_spec[argument_name].maximum + 1,
      output_size=vector_stream_size,
      mask_name=("prev_argument_masks", argument_name)))
  return component


def get_vector_encoder(obs_spec: types.ObsSpec,
                       action_spec: types.ActionSpec,
                       is_training: bool,
                       vector_stream_size: int,
                       config: ml_collections.ConfigDict
                       ) -> modular.Component:
  """The full vector encoder, encodes the vector inputs into vector_stream."""
  component = modular.SequentialComponent(name="vector_encoder")
  component.append(vector.ClockFeatureEncoder(
      name="game_loop",
      input_name=("observation", "game_loop"),
      output_name="game_loop_embedding",
      output_size=vector_stream_size,
      **config.game_loop))
  component.append(vector.VectorEncoder(
      name="unit_counts_bow",
      input_name=("observation", "unit_counts_bow"),
      output_name="unit_counts_bow_embedding",
      num_features=obs_spec["unit_counts_bow"].shape[0],
      output_size=vector_stream_size,
      **config.unit_counts_bow))
  component.append(vector.VectorEncoder(
      name="player",
      input_name=("observation", "player"),
      output_name="player_embedding",
      num_features=obs_spec["player"].shape[0],
      output_size=vector_stream_size,
      **config.player))
  component.append(vector.Embedding(
      name="mmr",
      input_name=("observation", "mmr"),
      output_name="mmr_embedding",
      output_size=vector_stream_size,
      **config.mmr))
  component.append(vector.Embedding(
      name="home_race_requested",
      input_name=("observation", "home_race_requested"),
      output_name="home_race_req_embedding",
      num_classes=5,
      output_size=vector_stream_size))
  component.append(vector.Embedding(
      name="away_race_requested",
      input_name=("observation", "away_race_requested"),
      output_name="away_race_req_embedding",
      num_classes=5,
      output_size=vector_stream_size))
  component.append(vector.Embedding(
      name="away_race_observed",
      input_name=("observation", "away_race_observed"),
      output_name="away_race_obs_embedding",
      num_classes=5,
      output_size=vector_stream_size))
  component.append(vector.FixedLengthToMask(
      name="upgrades_to_mask_encoder",
      input_name=("observation", "upgrades_fixed_length"),
      output_name="upgrades_boolean_mask",
      input_size=obs_spec["upgrades_fixed_length"].shape[0],
      num_classes=obs_spec["upgrades_fixed_length"].maximum + 1))
  component.append(vector.BinaryVectorEmbedding(
      name="upgrades_encoder",
      input_name="upgrades_boolean_mask",
      output_name="upgrades_embedding",
      input_size=obs_spec["upgrades_fixed_length"].maximum + 1,
      output_size=vector_stream_size))
  prev_actions_embeddings = []
  for arg in [util.Argument.FUNCTION,
              util.Argument.DELAY,
              util.Argument.QUEUED,
              util.Argument.REPEAT]:
    component.append(_get_prev_action_encoder(
        action_spec=action_spec,
        argument_name=arg,
        is_training=is_training,
        vector_stream_size=vector_stream_size))
    prev_actions_embeddings.append(("prev_action_embeddings", arg))
  component.append(merge.SumMerge(
      name="vector_encoder_merge",
      input_names=["game_loop_embedding", "unit_counts_bow_embedding",
                   "player_embedding", "mmr_embedding",
                   "home_race_req_embedding", "away_race_req_embedding",
                   "away_race_obs_embedding",
                   "upgrades_embedding"] + prev_actions_embeddings,
      output_name="vector_stream",
      stream_shape=(vector_stream_size,)))
  return component


def get_units_encoder(obs_spec: types.ObsSpec,
                      action_spec: types.ActionSpec,
                      units_stream_size: int,
                      config: ml_collections.ConfigDict
                      ) -> modular.Component:
  """The full units encoder, encodes the units inputs into units_stream."""
  component = modular.SequentialComponent(name="units_encoder")
  component.append(units.UnitsEncoder(
      name="raw_units",
      output_name="units_stream",
      max_num_observed_units=obs_spec["raw_units"].shape[0],
      num_raw_unit_features=obs_spec["raw_units"].shape[1],
      units_stream_size=units_stream_size,
      action_spec=action_spec,
      **config.raw_units))
  return component


def get_visual_encoder(obs_spec: types.ObsSpec,
                       action_spec: types.ActionSpec,
                       visual_features_size: int,
                       config: ml_collections.ConfigDict
                       ) -> modular.Component:
  """The full visual encoder, encodes the visual inputs into visual_stream."""
  world_size = util.get_world_size(action_spec)
  minimap_features = [
      "minimap_height_map",
      "minimap_visibility_map",
      "minimap_creep",
      "minimap_player_relative",
      "minimap_alerts",
      "minimap_pathable",
      "minimap_buildable"]
  spatial_size = obs_spec[minimap_features[0]].shape[0]
  # We assume all visual features have the same spatial size:
  for feature_name in minimap_features:
    assert obs_spec[feature_name].shape[0] == spatial_size
    assert obs_spec[feature_name].shape[1] == spatial_size
  component = modular.SequentialComponent(name="visual_encoder")
  streams_to_merge = []
  for feature_name in minimap_features:
    if feature_name == "minimap_height_map":
      component.append(visual.SingleFeatureEncoder(
          name="minimap_height_map",
          input_name=("observation", feature_name),
          output_name=f"{feature_name}_embedding",
          input_spatial_size=spatial_size,
          downscale_factor=config.downscale_factor,
          output_features_size=visual_features_size,
          **config.minimap_height_map))
    else:
      component.append(visual.Embedding(
          name=feature_name,
          input_name=("observation", feature_name),
          output_name=f"{feature_name}_embedding",
          input_spatial_size=spatial_size,
          downscale_factor=config.downscale_factor,
          num_classes=obs_spec[feature_name].maximum + 1,
          output_features_size=visual_features_size,
          **config[feature_name]))
    streams_to_merge.append(f"{feature_name}_embedding")
  component.append(visual.CameraEncoder(
      name="camera_encoder",
      output_name="camera_embedding",
      input_spatial_size=world_size,
      downscale_factor=config.downscale_factor * world_size // spatial_size,
      output_features_size=visual_features_size))
  streams_to_merge.append("camera_embedding")
  component.append(merge.SumMerge(
      name="visual_encoder_merge",
      input_names=streams_to_merge,
      output_name=f"visual_stream_ds{config.downscale_factor}",
      stream_shape=(spatial_size // config.downscale_factor,
                    spatial_size // config.downscale_factor,
                    visual_features_size)))
  return component

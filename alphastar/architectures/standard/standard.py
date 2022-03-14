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

"""AlphaStar standard v3 architecture."""

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures import util
from alphastar.architectures.standard import encoders
from alphastar.architectures.standard import heads
from alphastar.architectures.standard import torso
import ml_collections


def get_alphastar_standard(
    input_spec: types.InputSpec,
    action_spec: types.ActionSpec,
    is_training: bool,
    overlap_len: int,
    burnin_len: int,
    config: ml_collections.ConfigDict,
    name: str = 'alpha_star'
    ) -> modular.Component:
  """Returns the alphastar lite architecture."""
  del burnin_len
  obs_spec = input_spec.get('observation')
  component = modular.SequentialComponent(name=name)

  # Encoders:
  component.append(encoders.get_vector_encoder(
      obs_spec=obs_spec,
      action_spec=action_spec,
      is_training=is_training,
      vector_stream_size=config.vector_stream_size,
      config=config.encoders.vector))
  component.append(encoders.get_units_encoder(
      obs_spec=obs_spec,
      action_spec=action_spec,
      units_stream_size=config.units_stream_size,
      config=config.encoders.units))
  component.append(encoders.get_visual_encoder(
      obs_spec=obs_spec,
      action_spec=action_spec,
      visual_features_size=config.visual_stream_sizes[0],
      config=config.encoders.visual))

  # Torso:
  component.append(torso.get_torso(
      obs_spec=obs_spec,
      action_spec=action_spec,
      vector_stream_size=config.vector_stream_size,
      units_stream_size=config.units_stream_size,
      visual_stream_sizes=config.visual_stream_sizes,
      config=config.torso))

  # Heads:
  for arg in [util.Argument.FUNCTION,
              util.Argument.DELAY,
              util.Argument.QUEUED,
              util.Argument.REPEAT]:
    component.append(heads.get_vector_head(
        argument_name=arg,
        action_spec=action_spec,
        vector_stream_size=config.vector_stream_size,
        is_training=is_training,
        overlap_len=overlap_len,
        config=config.heads[arg]))

  component.append(heads.get_unit_tags_head(
      obs_spec=obs_spec,
      action_spec=action_spec,
      vector_stream_size=config.vector_stream_size,
      units_stream_size=config.units_stream_size,
      is_training=is_training,
      config=config.heads.unit_tags))
  component.append(heads.get_target_unit_tag_head(
      obs_spec=obs_spec,
      action_spec=action_spec,
      vector_stream_size=config.vector_stream_size,
      units_stream_size=config.units_stream_size,
      is_training=is_training,
      config=config.heads.target_unit_tag))
  component.append(heads.get_world_head(
      obs_spec=obs_spec,
      action_spec=action_spec,
      vector_stream_size=config.vector_stream_size,
      visual_stream_sizes=config.visual_stream_sizes,
      is_training=is_training,
      config=config.heads.world))

  return component

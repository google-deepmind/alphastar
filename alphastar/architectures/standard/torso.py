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

"""Torso blocks."""

from typing import Sequence

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures import util
from alphastar.architectures.components import merge
from alphastar.architectures.components import units
from alphastar.architectures.components import vector
from alphastar.architectures.components import visual
import ml_collections


def get_torso(obs_spec: types.ObsSpec,
              action_spec: types.ActionSpec,
              vector_stream_size: int,
              units_stream_size: int,
              visual_stream_sizes: Sequence[int],
              config: ml_collections.ConfigDict,
              ) -> modular.Component:
  """Gets the agent torso, where streams are combined and processed."""
  max_num_observed_units = obs_spec["raw_units"].shape[0]
  num_raw_unit_features = obs_spec["raw_units"].shape[1]
  spatial_size = obs_spec["minimap_height_map"].shape[0]
  first_downscale_factor = config.visual_downscale.downscale_factor
  component = modular.SequentialComponent(name="torso")
  component.append(vector.Resnet(
      name="vector_resnet_1",
      input_size=vector_stream_size,
      input_name="vector_stream",
      output_name="vector_stream",
      **config.vector_resnet_1))
  component.append(units.Transformer(
      name="units_transformer",
      max_num_observed_units=obs_spec["raw_units"].shape[0],
      units_stream_size=units_stream_size,
      input_name="units_stream",
      output_name="units_stream",
      **config.units_transformer))
  component.append(units.ToVisualScatter(
      name="scatter",
      input_name="units_stream",
      output_name=f"visual_stream_ds{first_downscale_factor}_from_units",
      max_num_observed_units=max_num_observed_units,
      num_raw_unit_features=num_raw_unit_features,
      units_stream_size=units_stream_size,
      units_world_dim=util.get_world_size(action_spec),
      output_spatial_size=spatial_size // first_downscale_factor,
      output_features_size=visual_stream_sizes[0],
      **config.scatter))
  component.append(merge.SumMerge(
      name="scatter_merge",
      input_names=[f"visual_stream_ds{first_downscale_factor}",
                   f"visual_stream_ds{first_downscale_factor}_from_units"],
      output_name=f"visual_stream_ds{first_downscale_factor}",
      stream_shape=(spatial_size // first_downscale_factor,
                    spatial_size // first_downscale_factor,
                    visual_stream_sizes[0])))
  for i in range(len(visual_stream_sizes) - 1):
    downscale_factor = config.visual_downscale.downscale_factor
    input_downscale_factor = int(downscale_factor**(i + 1))
    output_downscale_factor = int(downscale_factor**(i + 2))
    component.append(visual.Downscale(
        name=f"visual_downscale_ds{output_downscale_factor}",
        input_name=f"visual_stream_ds{input_downscale_factor}",
        output_name=f"visual_stream_ds{output_downscale_factor}",
        input_spatial_size=spatial_size // input_downscale_factor,
        input_features_size=visual_stream_sizes[i],
        output_features_size=visual_stream_sizes[i + 1],
        **config.visual_downscale))
  component.append(visual.Resnet(
      name="visual_resnet",
      input_name=f"visual_stream_ds{output_downscale_factor}",
      output_name=f"visual_stream_ds{output_downscale_factor}",
      input_spatial_size=spatial_size // output_downscale_factor,
      input_features_size=visual_stream_sizes[2],
      **config.visual_resnet))
  component.append(visual.ToVector(
      name="visual_to_vector",
      input_name=f"visual_stream_ds{output_downscale_factor}",
      output_name="vector_stream_from_visual",
      input_spatial_size=spatial_size // output_downscale_factor,
      input_features_size=visual_stream_sizes[2],
      vector_stream_size=vector_stream_size,
      **config.visual_to_vector))
  component.append(units.ToVector(
      name="torso_units_to_vector",
      input_name="units_stream",
      output_name="vector_stream_from_units",
      max_num_observed_units=max_num_observed_units,
      units_stream_size=units_stream_size,
      units_hidden_sizes=(units_stream_size * 2,),
      vector_stream_size=vector_stream_size))
  component.append(merge.VectorMerge(
      name="vector_merge",
      input_sizes={"vector_stream": vector_stream_size,
                   "vector_stream_from_units": vector_stream_size,
                   "vector_stream_from_visual": vector_stream_size},
      output_name="vector_stream",
      output_size=vector_stream_size,
      **config.vector_merge))
  component.append(vector.Resnet(
      name="vector_resnet_2",
      input_size=vector_stream_size,
      input_name="vector_stream",
      output_name="vector_stream",
      **config.vector_resnet_2))
  return component

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

"""Config for a "lite" version of the standard v3 architecture."""


from alphastar.architectures.components import merge
from alphastar.commons import sample
import jax.numpy as jnp
import ml_collections


def get_config() -> ml_collections.ConfigDict:
  """The config for the standard lite architecture."""
  config = ml_collections.ConfigDict()
  config.encoders = ml_collections.ConfigDict()
  config.encoders.vector = ml_collections.ConfigDict()
  config.encoders.vector.game_loop = ml_collections.ConfigDict()
  config.encoders.vector.unit_counts_bow = ml_collections.ConfigDict()
  config.encoders.vector.player = ml_collections.ConfigDict()
  config.encoders.vector.mmr = ml_collections.ConfigDict()
  config.encoders.units = ml_collections.ConfigDict()
  config.encoders.units.raw_units = ml_collections.ConfigDict()
  config.encoders.visual = ml_collections.ConfigDict()
  config.encoders.visual.minimap_height_map = ml_collections.ConfigDict()
  config.encoders.visual.minimap_visibility_map = ml_collections.ConfigDict()
  config.encoders.visual.minimap_creep = ml_collections.ConfigDict()
  config.encoders.visual.minimap_player_relative = ml_collections.ConfigDict()
  config.encoders.visual.minimap_alerts = ml_collections.ConfigDict()
  config.encoders.visual.minimap_pathable = ml_collections.ConfigDict()
  config.encoders.visual.minimap_buildable = ml_collections.ConfigDict()

  config.torso = ml_collections.ConfigDict()
  config.torso.vector_resnet_1 = ml_collections.ConfigDict()
  config.torso.units_transformer = ml_collections.ConfigDict()
  config.torso.scatter = ml_collections.ConfigDict()
  config.torso.visual_downscale = ml_collections.ConfigDict()
  config.torso.visual_resnet = ml_collections.ConfigDict()
  config.torso.visual_to_vector = ml_collections.ConfigDict()
  config.torso.units_to_vector = ml_collections.ConfigDict()
  config.torso.vector_merge = ml_collections.ConfigDict()
  config.torso.vector_resnet_2 = ml_collections.ConfigDict()

  config.heads = ml_collections.ConfigDict()
  config.heads.function = ml_collections.ConfigDict()
  config.heads.function.resnet = ml_collections.ConfigDict()
  config.heads.function.logits = ml_collections.ConfigDict()
  config.heads.function.sampling = ml_collections.ConfigDict()
  config.heads.function.sampling.sample = ml_collections.ConfigDict()
  config.heads.function.embedding_merge = ml_collections.ConfigDict()
  config.heads.delay = ml_collections.ConfigDict()
  config.heads.delay.resnet = ml_collections.ConfigDict()
  config.heads.delay.logits = ml_collections.ConfigDict()
  config.heads.delay.sampling = ml_collections.ConfigDict()
  config.heads.delay.sampling.sample = ml_collections.ConfigDict()
  config.heads.delay.embedding_merge = ml_collections.ConfigDict()
  config.heads.queued = ml_collections.ConfigDict()
  config.heads.queued.resnet = ml_collections.ConfigDict()
  config.heads.queued.logits = ml_collections.ConfigDict()
  config.heads.queued.sampling = ml_collections.ConfigDict()
  config.heads.queued.sampling.sample = ml_collections.ConfigDict()
  config.heads.queued.embedding_merge = ml_collections.ConfigDict()
  config.heads.repeat = ml_collections.ConfigDict()
  config.heads.repeat.resnet = ml_collections.ConfigDict()
  config.heads.repeat.logits = ml_collections.ConfigDict()
  config.heads.repeat.sampling = ml_collections.ConfigDict()
  config.heads.repeat.sampling.sample = ml_collections.ConfigDict()
  config.heads.repeat.embedding_merge = ml_collections.ConfigDict()
  config.heads.unit_tags = ml_collections.ConfigDict()
  config.heads.unit_tags.query_resnet = ml_collections.ConfigDict()
  config.heads.unit_tags.keys_mlp = ml_collections.ConfigDict()
  config.heads.unit_tags.small_embeddings_mlp = ml_collections.ConfigDict()
  config.heads.unit_tags.large_embeddings_mlp = ml_collections.ConfigDict()
  config.heads.unit_tags.inner_component = ml_collections.ConfigDict()
  config.heads.unit_tags.inner_component.logits = ml_collections.ConfigDict()
  config.heads.unit_tags.inner_component.sampling = ml_collections.ConfigDict()
  config.heads.unit_tags.inner_component.sampling.sample = (
      ml_collections.ConfigDict())
  config.heads.unit_tags.inner_component.embedding_merge = (
      ml_collections.ConfigDict())
  config.heads.unit_tags.embedding_merge = ml_collections.ConfigDict()
  config.heads.unit_tags.selected_units_merge = ml_collections.ConfigDict()
  config.heads.target_unit_tag = ml_collections.ConfigDict()
  config.heads.target_unit_tag.resnet = ml_collections.ConfigDict()
  config.heads.target_unit_tag.logits = ml_collections.ConfigDict()
  config.heads.target_unit_tag.sampling = ml_collections.ConfigDict()
  config.heads.target_unit_tag.sampling.sample = ml_collections.ConfigDict()
  config.heads.world = ml_collections.ConfigDict()
  config.heads.world.vector_to_visual = ml_collections.ConfigDict()
  config.heads.world.resnet = ml_collections.ConfigDict()
  config.heads.world.visual_upscale = ml_collections.ConfigDict()
  config.heads.world.logits = ml_collections.ConfigDict()
  config.heads.world.sampling = ml_collections.ConfigDict()
  config.heads.world.sampling.sample = ml_collections.ConfigDict()

  # Streams:
  config.vector_stream_size = 128
  config.units_stream_size = 32
  config.visual_stream_sizes = [4, 8, 16]

  # Encoders:
  config.encoders.vector.game_loop.encoding_size = 512
  config.encoders.vector.game_loop.t_min = 1
  config.encoders.vector.game_loop.t_max = 100_000
  config.encoders.vector.unit_counts_bow.fun = jnp.sqrt
  config.encoders.vector.player.fun = jnp.log1p
  config.encoders.vector.mmr.num_classes = 7
  config.encoders.vector.mmr.fun = lambda x: x * 0.001

  # Note: these numbers are environment dependent:
  config.encoders.units.raw_units.num_unit_types = 256
  config.encoders.units.raw_units.num_buff_types = 46

  config.encoders.visual.downscale_factor = 2
  config.encoders.visual.minimap_height_map.fun = lambda x: x / 255.
  config.encoders.visual.minimap_height_map.kernel_size = 2
  config.encoders.visual.minimap_visibility_map.kernel_size = 2
  config.encoders.visual.minimap_creep.kernel_size = 2
  config.encoders.visual.minimap_player_relative.kernel_size = 2
  config.encoders.visual.minimap_alerts.kernel_size = 2
  config.encoders.visual.minimap_pathable.kernel_size = 2
  config.encoders.visual.minimap_buildable.kernel_size = 2

  # Torso:
  config.torso.vector_resnet_1.num_resblocks = 2
  config.torso.vector_resnet_1.use_layer_norm = True

  config.torso.units_transformer.transformer_num_layers = 1
  config.torso.units_transformer.transformer_num_heads = 2
  config.torso.units_transformer.transformer_key_size = 32
  config.torso.units_transformer.transformer_value_size = 16
  config.torso.units_transformer.resblocks_num_before = 1
  config.torso.units_transformer.resblocks_num_after = 1
  config.torso.units_transformer.resblocks_hidden_size = None
  config.torso.units_transformer.use_layer_norm = True

  config.torso.scatter.units_hidden_sizes = [16]
  config.torso.scatter.kernel_size = 3
  config.torso.scatter.use_layer_norm = True

  config.torso.visual_downscale.downscale_factor = 2
  config.torso.visual_downscale.kernel_size = 2

  config.torso.visual_resnet.num_resblocks = 2
  config.torso.visual_resnet.kernel_size = 3
  config.torso.visual_resnet.use_layer_norm = True
  config.torso.visual_resnet.num_hidden_feature_planes = None

  config.torso.visual_to_vector.hidden_feature_sizes = [32, 64]
  config.torso.visual_to_vector.downscale_factor = 2
  config.torso.visual_to_vector.use_layer_norm = True
  config.torso.visual_to_vector.kernel_size = 2

  config.torso.units_to_vector.units_hidden_sizes = [64]
  config.torso.units_to_vector.use_layer_norm = True

  config.torso.vector_merge.gating_type = merge.GatingType.POINTWISE

  config.torso.vector_resnet_2.num_resblocks = 2
  config.torso.vector_resnet_2.use_layer_norm = True

  # Heads:
  config.heads.function.resnet.num_resblocks = 4
  config.heads.function.resnet.use_layer_norm = True
  config.heads.function.logits.num_linear_layers = 1
  config.heads.function.logits.use_layer_norm = True

  config.heads.function.sampling.sample.sample_fn = sample.sample
  config.heads.function.embedding_merge.gating_type = merge.GatingType.GLOBAL

  config.heads.delay.resnet.num_resblocks = 2
  config.heads.delay.resnet.use_layer_norm = True
  config.heads.delay.logits.num_linear_layers = 1
  config.heads.delay.logits.use_layer_norm = True
  config.heads.delay.sampling.sample.sample_fn = sample.sample
  config.heads.delay.embedding_merge.gating_type = merge.GatingType.GLOBAL

  config.heads.queued.resnet.num_resblocks = 0
  config.heads.queued.resnet.use_layer_norm = True
  config.heads.queued.logits.num_linear_layers = 1
  config.heads.queued.logits.use_layer_norm = True
  config.heads.queued.sampling.sample.sample_fn = sample.sample
  config.heads.queued.embedding_merge.gating_type = merge.GatingType.GLOBAL

  config.heads.repeat.resnet.num_resblocks = 0
  config.heads.repeat.resnet.use_layer_norm = True
  config.heads.repeat.logits.num_linear_layers = 1
  config.heads.repeat.logits.use_layer_norm = True
  config.heads.repeat.sampling.sample.sample_fn = sample.sample
  config.heads.repeat.embedding_merge.gating_type = merge.GatingType.GLOBAL

  config.heads.unit_tags.query_resnet.num_resblocks = 1
  config.heads.unit_tags.query_resnet.use_layer_norm = True
  config.heads.unit_tags.keys_mlp.layer_sizes = [64, 16]
  config.heads.unit_tags.small_embeddings_mlp.layer_sizes = [32, 8]
  config.heads.unit_tags.large_embeddings_mlp.layer_sizes = [64, 64]
  config.heads.unit_tags.inner_component.logits.num_layers_query = 1
  config.heads.unit_tags.inner_component.logits.num_layers_keys = 0
  config.heads.unit_tags.inner_component.logits.use_layer_norm = True
  config.heads.unit_tags.inner_component.sampling.sample.sample_fn = sample.sample
  config.heads.unit_tags.inner_component.embedding_merge.gating_type = merge.GatingType.NONE
  config.heads.unit_tags.embedding_merge.gating_type = merge.GatingType.GLOBAL
  config.heads.unit_tags.selected_units_merge.gating_type = merge.GatingType.GLOBAL

  config.heads.target_unit_tag.resnet.num_resblocks = 1
  config.heads.target_unit_tag.resnet.use_layer_norm = True
  config.heads.target_unit_tag.logits.num_layers_query = 1
  config.heads.target_unit_tag.logits.num_layers_keys = 1
  config.heads.target_unit_tag.logits.key_size = 16
  config.heads.target_unit_tag.logits.use_layer_norm = True
  config.heads.target_unit_tag.sampling.sample.sample_fn = sample.sample

  config.heads.world.visual_feature_size_ds1 = 2
  config.heads.world.vector_to_visual.hidden_feature_sizes = [32, 32]
  config.heads.world.vector_to_visual.upscale_factor = 2
  config.heads.world.vector_to_visual.use_layer_norm = True
  config.heads.world.vector_to_visual.kernel_size = 2
  config.heads.world.resnet.num_resblocks = 2
  config.heads.world.resnet.kernel_size = 3
  config.heads.world.resnet.use_layer_norm = True
  config.heads.world.resnet.num_hidden_feature_planes = None
  config.heads.world.visual_upscale.upscale_factor = 2
  config.heads.world.visual_upscale.kernel_size = 2
  config.heads.world.logits.use_layer_norm = True
  config.heads.world.logits.use_depth_to_space = False
  config.heads.world.sampling.sample.sample_fn = sample.sample

  return config

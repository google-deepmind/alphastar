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

"""AlphaStar dummy (v3) architecture.

Minimal architecture, should not train (or terribly) but the interface
is correct.
"""

from typing import Mapping, Optional

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures import util
from alphastar.architectures.components import common
from alphastar.architectures.components import units
from alphastar.architectures.components import vector
from alphastar.commons import sample
import jax.numpy as jnp


def _get_vector_head(argument_name: types.ArgumentName,
                     action_spec: types.ActionSpec,
                     vector_stream_size: int,
                     is_training: bool,
                     sample_fn: sample.SampleFn
                     ) -> modular.Component:
  """Produces logits and action for a vector argument."""
  num_logits = action_spec[argument_name].maximum + 1
  component = modular.SequentialComponent(name=f"{argument_name}_head")
  component.append(vector.Logits(
      name=f"{argument_name}_logits",
      logits_output_name=("logits", argument_name),
      mask_output_name=("masks", argument_name),
      num_logits=num_logits,
      input_size=vector_stream_size,
      input_name="vector_stream",
      num_linear_layers=1))
  if is_training:
    component.append(common.ActionFromBehaviourFeatures(
        name=f"{argument_name}_action",
        argument_name=argument_name))
  else:
    component.append(common.Sample(
        name=f"{argument_name}_sample",
        argument_name=argument_name,
        num_logits=num_logits,
        sample_fn=sample_fn))
  if argument_name == util.Argument.FUNCTION:
    component.append(common.ArgumentMasks(
        name="argument_masks",
        action_spec=action_spec))
  return component


def _get_unit_tags_head(action_spec: types.ActionSpec,
                        vector_stream_size: int,
                        is_training: bool,
                        sample_fn: sample.SampleFn) -> modular.Component:
  """Produces logits and action for unit_tag argument."""
  num_logits = action_spec["unit_tags"].maximum + 1
  max_num_selected_units = action_spec["unit_tags"].shape[0]
  max_num_observed_units = int(action_spec["unit_tags"].maximum)
  inner_component = modular.SequentialComponent(
      name="unit_tags_inner_component")
  inner_component.append(vector.Logits(
      name="unit_tags_logits",
      logits_output_name=("logits", util.Argument.UNIT_TAGS),
      mask_output_name=("masks", util.Argument.UNIT_TAGS),
      num_logits=num_logits,
      input_size=vector_stream_size,
      input_name="vector_stream",
      num_linear_layers=1))
  if is_training:
    inner_component.append(common.ActionFromBehaviourFeatures(
        name="unit_tags_action",
        argument_name=util.Argument.UNIT_TAGS))
  else:
    inner_component.append(common.Sample(
        name="unit_tags_sample",
        argument_name=util.Argument.UNIT_TAGS,
        num_logits=num_logits,
        sample_fn=sample_fn))

  unit_tags_head_per_step_inputs = []
  if is_training:
    unit_tags_head_per_step_inputs.append(
        ("behaviour_features", "action", util.Argument.UNIT_TAGS))
  return units.UnitTagsHead(
      name="unit_tags_head",
      inner_component=inner_component,
      constant_inputs=["vector_stream"],
      carries=[],
      per_step_inputs=unit_tags_head_per_step_inputs,
      per_step_outputs=[("logits", util.Argument.UNIT_TAGS),
                        ("masks", util.Argument.UNIT_TAGS),
                        ("action", util.Argument.UNIT_TAGS)],
      max_num_selected_units=max_num_selected_units,
      max_num_observed_units=max_num_observed_units,
      action_output=("action", util.Argument.UNIT_TAGS))


def get_alphastar_dummy(
    input_spec: types.InputSpec,
    action_spec: types.ActionSpec,
    is_training: bool,
    overlap_len: int,
    burnin_len: int,
    sample_fns: Optional[Mapping[types.ArgumentName, sample.SampleFn]] = None,
    name: str = "alpha_star_dummy",
    **unused_kwargs
    ) -> modular.Component:
  """Returns the alphastar dummy architecture."""
  del overlap_len
  del burnin_len
  obs_spec = input_spec.get("observation")
  vector_stream_size = 1024
  if sample_fns is None:
    sample_fns = {k: sample.sample for k in action_spec}
  component = modular.SequentialComponent(name=name)

  # Encoders:
  component.append(vector.VectorEncoder(
      name="player_encoder",
      input_name=("observation", "player"),
      output_name="vector_stream",
      num_features=obs_spec["player"].shape[0],
      output_size=vector_stream_size,
      fun=jnp.log1p))

  # Heads:
  for arg in [util.Argument.FUNCTION,
              util.Argument.DELAY,
              util.Argument.QUEUED,
              util.Argument.REPEAT,
              util.Argument.TARGET_UNIT_TAG,
              util.Argument.WORLD]:
    component.append(_get_vector_head(
        argument_name=arg,
        action_spec=action_spec,
        vector_stream_size=vector_stream_size,
        is_training=is_training,
        sample_fn=sample_fns[arg]))

  component.append(_get_unit_tags_head(
      action_spec=action_spec,
      vector_stream_size=vector_stream_size,
      is_training=is_training,
      sample_fn=sample_fns[util.Argument.UNIT_TAGS]))

  return component

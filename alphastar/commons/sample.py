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

"""Utilities for sampling actions from logits proposed by a model."""

from typing import Callable, Optional

import chex
import haiku as hk
import jax
import jax.numpy as jnp

SampleFn = Callable[[chex.Array], chex.Array]

# TODO(b/207777292) : Add NaN checks to logits.


def mask_logits(logits: chex.Array,
                mask: chex.Array,
                unavailable_logits_bias: float = 1e10) -> chex.Array:
  """Mask logits with a tiny bias on elements to be masked."""
  chex.assert_equal_shape([logits, mask])
  chex.assert_type([logits, mask], [jnp.float32, jnp.bool_])
  assert len(logits.shape) == len(mask.shape)
  return jnp.where(mask, logits, -unavailable_logits_bias)


def apply_temperature_to_logits(logits: chex.Array,
                                temperature: Optional[float] = None
                                ) -> chex.Array:
  """Apply a temperature (scale up the logits) to logits."""
  chex.assert_type(logits, jnp.float32)
  if temperature is not None:
    logits /= temperature
  return logits


def finalize_logits(logits: chex.Array,
                    mask: chex.Array,
                    temperature: Optional[float] = None) -> chex.Array:
  """Apply a temperature and mask logits."""
  logits = apply_temperature_to_logits(logits, temperature)
  return mask_logits(logits, mask)


def sample(logits: chex.Array,
           temperature: Optional[float] = None) -> chex.Array:
  """Sample from logits, given a temperature."""
  logits = apply_temperature_to_logits(logits, temperature)
  logits = logits.astype(jnp.float32)
  x = jax.random.categorical(hk.next_rng_key(), logits)
  return x.astype(jnp.int32)


def nucleus_sample(
    logits: chex.Array,
    top_p: float = 1.0
    ) -> chex.Array:
  """Performs nucleus sampling on logits."""
  sorted_logits = jax.lax.sort(logits, is_stable=False)
  sorted_probs = jax.nn.softmax(sorted_logits)
  threshold_idx = jnp.argmax(
      jnp.cumsum(sorted_probs, -1) >= 1 - top_p, axis=-1)
  threshold_largest_logits = jnp.take_along_axis(
      sorted_logits, threshold_idx[..., jnp.newaxis], axis=-1)
  assert threshold_largest_logits.shape == logits.shape[:-1] + (1,)
  mask = logits >= threshold_largest_logits
  logits = mask_logits(logits, mask)  # Set unused logits to -inf.
  return sample(logits)

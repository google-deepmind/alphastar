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

"""Vector-based components, acting on 1d vectors."""

from typing import Callable, Optional, Sequence

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import util
import chex
from dm_env import specs
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class VectorEncoder(modular.BatchedComponent):
  """Encodes a vector of features by applying a function and a linear layer."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               num_features: int,
               output_size: int,
               fun: Optional[Callable[[chex.Array], chex.Array]] = None,
               name: Optional[str] = None):
    """Initializes VectorEncoder module.

    Args:
      input_name: The name of the input to use, of shape
        [num_features] and dtype int32.
      output_name: The name to give to the output, of shape
        [output_size] and dtype float32.
      num_features: The size of the input to encode.
      output_size: The size of the output vector.
      fun: An optional function to apply to the input before applying the linear
        transformation.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._num_features = num_features
    self._output_size = output_size
    self._fun = fun

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._num_features,), np.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_size,), np.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    x = util.astype(x, jnp.float32)
    if self._fun is not None:
      x = self._fun(x)
    x = hk.Linear(self._output_size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Embedding(modular.BatchedComponent):
  """Embeds a single feature (a int32) into a float32 vector (hk.Embed)."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               num_classes: int,
               output_size: int,
               mask_name: Optional[types.StreamType] = None,
               fun: Optional[Callable[[chex.Array], chex.Array]] = None,
               name: Optional[str] = None):
    """Initializes Embedding module.

    Args:
      input_name: The name of the input to use, of shape [] and dtype int32.
      output_name: The name to give to the output, of shape
        [output_size] and dtype float32.
      num_classes: The number of values the input can take, ie. max(input)-1.
        For safety, the input is clipped to stay within [0, num_classes-1], but
        it probably should never be larger than num_classes-1. If using `fun`,
        this is the maximum value after applying the function `fun`.
      output_size: The size of the output vector.
      mask_name: If specified, this determines a mask input, of shape [] and
        dtype bool, to mask the output. The output will be 0 if the mask is
        set to False, otherwise it will be unaffected.
      fun: An optional function to apply to the input before embedding.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._num_classes = num_classes
    self._output_size = output_size
    self._mask_name = mask_name
    self._fun = fun

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict({self._input_name: specs.Array((), np.int32)})
    if self._mask_name is not None:
      spec[self._mask_name] = specs.Array((), np.bool_)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_size,), np.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    if self._fun is not None:
      x = self._fun(x)
    x = jnp.minimum(util.astype(x, jnp.int32), self._num_classes - 1)
    x = hk.Embed(vocab_size=self._num_classes, embed_dim=self._output_size)(x)
    if self._mask_name:
      x = jnp.where(inputs[self._mask_name], x, 0)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class FixedLengthToMask(modular.BatchedComponent):
  """Converts a fixed length list of integer into a boolean mask."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_size: int,
               num_classes: int,
               name: Optional[str] = None):
    """Initializes FixedLengthToMask module.

    Args:
      input_name: The name of the input to use, of shape [input_size] and
        dtype int32.
      output_name: The name to give to the output, of shape
        [num_classes] and dtype bool.
      input_size: The size of the input. The output will contain `input_size`
        ones, or less if the same value appears several times in the input.
      num_classes: The number of values the inputs can take, ie. max(input)-1.
        For safety, the input is clipped to stay within [0, num_classes-1], but
        it probably should never be larger than num_classes-1. This is also the
        size of the output.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._input_size = input_size
    self._num_classes = num_classes

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_size,), np.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._num_classes,), np.bool_)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    indices = inputs[self._input_name]
    indices = jnp.minimum(indices, self._num_classes - 1)
    mask = jnp.matmul(jnp.ones_like(indices),
                      indices[:, jnp.newaxis] == jnp.arange(self._num_classes))
    mask = util.astype(mask, jnp.bool_)
    outputs = types.StreamDict({self._output_name: mask})
    return outputs, {}


class BinaryVectorEmbedding(modular.BatchedComponent):
  """Encodes a boolean mask."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_size: int,
               output_size: int,
               mask_name: Optional[types.StreamType] = None,
               name: Optional[str] = None):
    """Initializes BinaryVectorEmbedding module.

    Args:
      input_name: The name of the input to use, of shape [input_size] and
        dtype bool.
      output_name: The name to give to the output, of shape
        [output_size] and dtype bool.
      input_size: The size of the input vector.
      output_size: The size of the output vector.
      mask_name: If specified, this determines a mask input, of shape [] and
        dtype bool, to mask the output. The output will be 0 if the mask is
        set to False, otherwise it will be unaffected.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._input_size = input_size
    self._output_size = output_size
    self._mask_name = mask_name

  @property
  def input_spec(self) -> types.SpecDict:
    spec = types.SpecDict({
        self._input_name: specs.Array((self._input_size,), np.bool_)})
    if self._mask_name is not None:
      spec[self._mask_name] = specs.Array((), np.bool_)
    return spec

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_size,), np.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    x = hk.Linear(output_size=self._output_size)(util.astype(x, jnp.float32))
    if self._mask_name:
      x = jnp.where(inputs[self._mask_name], x, 0)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class ClockFeatureEncoder(modular.BatchedComponent):
  """Embedding for the game clock value position."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               encoding_size: int,
               output_size: int,
               t_min: int = 1,
               t_max: int = 100_000,
               name: Optional[str] = None):
    """Initializes ClockFeatureEncoder module.

    Args:
      input_name: The name of the input to use, of shape [] and dtype int32.
      output_name: The name to give to the output, of shape [output_size] and
        dtype float32.
      encoding_size: The size of the sine-wave encoding. It must be even.
      output_size: The size of the output after the linear layer is applied
        to the sine-wave encoding.
      t_min: The minimum expected clock value.
      t_max: The maximum expected clock value.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    if encoding_size % 2:
      raise ValueError(f"Encoding size (set to {encoding_size}) must be even.")
    self._encoding_size = encoding_size
    self._output_size = output_size
    self._t_min = float(t_min)
    self._t_max = float(t_max)

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({self._input_name: specs.Array((), np.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_size,), np.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    num_timescales = float(self._encoding_size // 2)
    log_timescale_step = np.log(self._t_max / self._t_min) / (num_timescales-1)
    timescales = np.arange(num_timescales, dtype=np.float32)
    inv_timescales = np.exp(timescales * -log_timescale_step) / self._t_min
    inv_timescales = jnp.asarray(inv_timescales)
    x = util.astype(inputs[self._input_name], jnp.float32)
    rescaled_time = x[jnp.newaxis] * inv_timescales
    x = jnp.concatenate(
        [jnp.sin(rescaled_time), jnp.cos(rescaled_time)], axis=-1)
    x = hk.Linear(self._output_size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Resnet(modular.BatchedComponent):
  """A fully-connected resnet."""

  def __init__(self,
               input_size: int,
               num_resblocks: int,
               use_layer_norm: bool = True,
               input_name: types.StreamType = "vector_stream",
               output_name: types.StreamType = "vector_stream",
               name: Optional[str] = None):
    """Initializes Resnet module.

    Args:
      input_size: The size of the input vector.
      num_resblocks: The number of residual blocks.
      use_layer_norm: Whether to use layer normalization.
      input_name: The name of the input to use, of shape [input_size] and
        dtype float32.
      output_name: The name to give to the output, of shape [input_size] and
        dtype float32.
      name: Name of the component.
    """
    super().__init__(name=name)
    self._input_size = input_size
    self._num_resblocks = num_resblocks
    self._use_layer_norm = use_layer_norm
    self._input_name = input_name
    self._output_name = output_name

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_size,), jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._input_size,), jnp.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for _ in range(self._num_resblocks):
      x = util.VectorResblock(use_layer_norm=self._use_layer_norm)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class ToVisual(modular.BatchedComponent):
  """Vector to visual using linear layer + strided convolutions."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_size: int,
               output_spatial_size: int,
               output_features_size: int,
               hidden_feature_sizes: Sequence[int],
               upscale_factor: int = 2,
               use_layer_norm: bool = True,
               kernel_size: int = 4,
               name: Optional[str] = None):
    """Initializes ToVisual module.

    Args:
      input_name: The name of the input to use, of shape [input_size] and
        dtype float32.
      output_name: The name to give to the output, of shape
        [output_spatial_size, output_spatial_size, output_features_size] and
        dtype float32.
      input_size: The size of the input vector.
      output_spatial_size: The spatial size of the output (2d feature maps).
      output_features_size: The number of the feature planes in the output (2d
        feature maps).
      hidden_feature_sizes: The number of feature planes before the output. Each
        convolution is strided, ie. increasing the spatial resolution.
      upscale_factor: The upscale factor of each strided convolution.
      use_layer_norm: Whether to use layer normalization.
      kernel_size: The size of the convolution kernel to use. Note that with
        upsampling, a `kernel_size` not multiple of the `upscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      name: Name of the component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._output_spatial_size = output_spatial_size
    self._input_size = input_size
    self._feature_sizes = list(hidden_feature_sizes) + [output_features_size]
    if output_spatial_size % (upscale_factor ** (len(self._feature_sizes) - 1)):
      raise ValueError(f"The spatial output size (set to {output_spatial_size})"
                       " must be a multiple of the upscale factor "
                       f"({upscale_factor}) to the power of the number of "
                       f"convolutions ({len(self._feature_sizes)}).")
    self._use_layer_norm = use_layer_norm
    self._upscale_factor = upscale_factor
    if upscale_factor < 1:
      raise ValueError(f"upscale_factor (set to {upscale_factor}) must be > 0.")
    self._kernel_size = kernel_size

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_size,), jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._output_spatial_size,
                                        self._output_spatial_size,
                                        self._feature_sizes[-1]),
                                       jnp.float32)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    res = (self._output_spatial_size //
           (self._upscale_factor ** (len(self._feature_sizes) - 1)))
    if self._use_layer_norm:
      x = util.vector_layer_norm(x)
    x = jax.nn.relu(x)
    x = hk.Linear(output_size=self._feature_sizes[0] * res * res)(x)
    x = jnp.reshape(x, [res, res, self._feature_sizes[0]])
    for num_features in self._feature_sizes[1:]:
      if self._use_layer_norm:
        x = util.visual_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Conv2DTranspose(
          output_channels=num_features,
          kernel_shape=self._kernel_size,
          stride=self._upscale_factor)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Logits(modular.BatchedComponent):
  """Logits for scalar heads (function, delay, queued, repeat)."""

  def __init__(
      self,
      num_logits: int,
      input_size: int,
      logits_output_name: types.StreamType,
      mask_output_name: types.StreamType,
      input_name: types.StreamType = "vector_stream",
      num_linear_layers: int = 2,
      use_layer_norm: bool = True,
      name: Optional[str] = None):
    """Initializes Logits module.

    Args:
      num_logits: The number of logits to produce.
      input_size: The size of the input vector.
      logits_output_name: The name to give to the output for the logits, of
        shape [num_logits] and dtype float32.
      mask_output_name: The name to give to the output for the logits, of
        shape [num_logits] and dtype bool. Logits from vector logits are not
        masked, so the mask only contains ones.
      input_name: The name of the input to use, of shape [input_size] and
        dtype float32.
      num_linear_layers: Number of linear layers to use to compute the logits
        (ie. the depth of the MLP).
      use_layer_norm: Whether to use layer normalization.
      name: Name of the component.
    """
    super().__init__(name)
    self._num_logits = num_logits
    self._input_size = input_size
    self._logits_output_name = logits_output_name
    self._mask_output_name = mask_output_name
    self._input_name = input_name
    if num_linear_layers < 1:
      raise ValueError(
          f"num_linear_layers (set to {num_linear_layers}) must be > 0.")
    self._num_linear_layers = num_linear_layers
    self._use_layer_norm = use_layer_norm

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_size,), np.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._logits_output_name: specs.Array((self._num_logits,), np.float32),
        self._mask_output_name: specs.Array((self._num_logits,), np.bool_)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for i in range(self._num_linear_layers):
      if i == self._num_linear_layers - 1:
        output_size = self._num_logits
      else:
        output_size = x.shape[-1]
      if self._use_layer_norm:
        x = util.vector_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Linear(output_size=output_size)(x)

    # Vector heads do not use masks (in this implementation).
    mask = jnp.ones((self._num_logits,), dtype=jnp.bool_)

    # Note: if mask is used, add `logits = common.mask_logits(logits, mask)`
    outputs = types.StreamDict({self._logits_output_name: x,
                                self._mask_output_name: mask})
    return outputs, {}

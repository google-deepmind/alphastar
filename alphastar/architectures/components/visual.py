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

"""Visual-based modules, acting on 2d feature maps."""

from typing import Callable, Optional, Sequence, Tuple, Union

from alphastar import types
from alphastar.architectures import modular
from alphastar.architectures.components import util
from alphastar.architectures.components.static_data import camera_masks
from alphastar.commons import sample
import chex
from dm_env import specs
import dm_pix as pix
import haiku as hk
import jax
import jax.numpy as jnp
import numpy as np


class FeatureEncoder(modular.BatchedComponent):
  """Encodes 2d feature maps applying a function and a convolution."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: Union[int, Tuple[int, int]],
               input_feature_size: Optional[int],
               downscale_factor: int,
               output_features_size: int,
               kernel_size: int,
               fun: Callable[[chex.Array], chex.Array] = lambda x: x,
               input_dtype: jnp.dtype = jnp.uint8,
               name: Optional[str] = None):
    """Initializes FeatureEncoder module.

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size[0], input_spatial_size[1]] and dtype int32.
      output_name: The name to give to the output, of shape
        [input_spatial_size[0] / downscale_factor,
         input_spatial_size[1] / downscale_factor,
         output_features_size] and dtype float32.
      input_spatial_size: The spatial size of the input to encode.
        If the input is square, a single int can be used, otherwise
        a pair is required.
      input_feature_size: The number of feature planes of the input, or None
        if the input does not have a feature dimension.
      downscale_factor: The downscale factor to apply to the input.
      output_features_size: The number of feature planes of the output.
      kernel_size: The size of the convolution kernel to use. Note that with
        downsampling, a `kernel_size` not multiple of the `downscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      fun: An optional function to apply to the input before applying the
        convolution.
      input_dtype: The type of the input.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    if isinstance(input_spatial_size, int):
      self._input_spatial_size = (input_spatial_size, input_spatial_size)
    else:
      self._input_spatial_size = tuple(input_spatial_size)
    for i in range(2):
      if self._input_spatial_size[i] % downscale_factor:
        raise ValueError(f'input_spatial_size[{i}] must be a multiple of '
                         f'downscale_factor ({downscale_factor}) but is '
                         f'({self._input_spatial_size[i]}).')
    self._input_feature_size = input_feature_size
    self._downscale_factor = downscale_factor
    self._output_features_size = output_features_size
    self._kernel_size = kernel_size
    self._fun = fun
    self._input_dtype = input_dtype

  @property
  def input_spec(self) -> types.SpecDict:
    if self._input_feature_size is None:
      input_size = self._input_spatial_size
    else:
      input_size = self._input_spatial_size + (self._input_feature_size,)
    return types.SpecDict({
        self._input_name: specs.Array(input_size, self._input_dtype)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = tuple(size // self._downscale_factor
                        for size in self._input_spatial_size)
    return types.SpecDict({
        self._output_name: specs.Array(
            output_size + (self._output_features_size,), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    x = util.astype(x, jnp.float32)
    x = self._fun(x)
    if self._input_feature_size is None:
      x = x[..., jnp.newaxis]
    x = hk.Conv2D(output_channels=self._output_features_size,
                  kernel_shape=self._kernel_size,
                  stride=self._downscale_factor)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Embedding(modular.BatchedComponent):
  """Encodes a visual (2d) int32 input, embedding each pixel independently."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: int,
               downscale_factor: int,
               num_classes: int,
               output_features_size: int,
               kernel_size: int,
               name: Optional[str] = None):
    """Initializes Embedding module.

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size] and dtype int32.
      output_name: The name to give to the output, of shape
        [input_spatial_size / downscale_factor,
         input_spatial_size / downscale_factor,
         output_features_size] and dtype float32.
      input_spatial_size: The spatial size of the input to encode.
      downscale_factor: The downscale factor to apply to the input.
      num_classes: The number of values the input can take, ie. max(input)-1.
        For safety, the input is clipped to stay within [0, num_classes-1], but
        it probably should never be larger than num_classes-1.
      output_features_size: The number of feature planes of the output.
      kernel_size: The size of the convolution kernel to use. Note that with
        downsampling, a `kernel_size` not multiple of the `downscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._input_spatial_size = input_spatial_size
    if input_spatial_size % downscale_factor:
      raise ValueError(f'input_spatial_size ({input_spatial_size}) must be a '
                       f'multiple of downscale_factor ({downscale_factor}).')
    self._downscale_factor = downscale_factor
    self._num_classes = num_classes
    self._output_features_size = output_features_size
    self._kernel_size = kernel_size

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array(
            (self._input_spatial_size, self._input_spatial_size), jnp.uint8)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = self._input_spatial_size // self._downscale_factor
    return types.SpecDict({
        self._output_name: specs.Array(
            (output_size, output_size, self._output_features_size), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    x = jnp.minimum(x, self._num_classes - 1)
    x = jax.nn.one_hot(x, self._num_classes, dtype=jnp.float32)
    x = hk.Conv2D(output_channels=self._output_features_size,
                  kernel_shape=self._kernel_size,
                  stride=self._downscale_factor)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class CameraEncoder(modular.BatchedComponent):
  """Encodes the camera visual feature."""

  def __init__(self,
               output_name: types.StreamType,
               input_spatial_size: int,
               downscale_factor: int,
               output_features_size: int,
               name: Optional[str] = None):
    """Initializes CameraEncoder module..

    Args:
      output_name: The name to give to the output, of shape
        [input_spatial_size / downscale_factor,
         input_spatial_size / downscale_factor,
         output_features_size] and dtype float32.
      input_spatial_size: The spatial size of the input to encode.
      downscale_factor: The downscale factor to apply to the input.
      output_features_size: The number of feature planes of the output.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._output_name = output_name
    self._input_spatial_size = input_spatial_size
    if input_spatial_size % downscale_factor:
      raise ValueError(f'input_spatial_size ({input_spatial_size}) must be a '
                       f'multiple of downscale_factor ({downscale_factor}).')
    self._downscale_factor = downscale_factor
    self._output_features_size = output_features_size

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        ('observation', 'camera'): specs.Array(
            (self._input_spatial_size, self._input_spatial_size), jnp.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = self._input_spatial_size // self._downscale_factor
    return types.SpecDict({
        self._output_name: specs.Array(
            (output_size, output_size, self._output_features_size), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs['observation', 'camera']
    ds = self._downscale_factor
    x = hk.AvgPool((ds, ds), (ds, ds), 'SAME')(x)[..., jnp.newaxis]
    x = hk.Linear(self._output_features_size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Downscale(modular.BatchedComponent):
  """Downscale the visual stream.."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: int,
               input_features_size: int,
               output_features_size: int,
               downscale_factor: int,
               kernel_size: int,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes Downscale module..

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      output_name: The name to give to the output, of shape
        [input_spatial_size / downscale_factor,
         input_spatial_size / downscale_factor,
         output_features_size] and dtype float32.
      input_spatial_size: The spatial size of the input.
      input_features_size: The number of feature planes of the input.
      output_features_size: The number of feature planes of the output.
      downscale_factor: The downscale factor to apply to the input.
      kernel_size: The size of the convolution kernel to use. Note that with
        downsampling, a `kernel_size` not multiple of the `downscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    if input_spatial_size % downscale_factor:
      raise ValueError(f'input_spatial_size ({input_spatial_size}) must be a '
                       f'multiple of downscale_factor ({downscale_factor}).')
    self._input_spatial_size = input_spatial_size
    self._input_features_size = input_features_size
    self._output_features_size = output_features_size
    self._downscale_factor = downscale_factor
    self._kernel_size = kernel_size
    self._use_layer_norm = use_layer_norm

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_spatial_size,
                                       self._input_spatial_size,
                                       self._input_features_size),
                                      jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = self._input_spatial_size // self._downscale_factor
    return types.SpecDict({
        self._output_name: specs.Array(
            (output_size, output_size, self._output_features_size), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    if self._use_layer_norm:
      x = util.visual_layer_norm(x)
    x = jax.nn.relu(x)
    x = hk.Conv2D(
        output_channels=self._output_features_size,
        kernel_shape=self._kernel_size,
        stride=self._downscale_factor)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Upscale(modular.BatchedComponent):
  """Upscale the visual stream."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: int,
               input_features_size: int,
               output_features_size: int,
               upscale_factor: int,
               kernel_size: int,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes Upscale module..

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      output_name: The name to give to the output, of shape
        [input_spatial_size * upscale_factor,
         input_spatial_size * upscale_factor,
         output_features_size] and dtype float32.
      input_spatial_size: The spatial size of the input.
      input_features_size: The number of feature planes of the input.
      output_features_size: The number of feature planes of the output.
      upscale_factor: The upscale factor to apply to the input.
      kernel_size: The size of the convolution kernel to use. Note that with
        upsampling, a `kernel_size` not multiple of the `upscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    self._input_spatial_size = input_spatial_size
    self._input_features_size = input_features_size
    self._output_features_size = output_features_size
    self._upscale_factor = upscale_factor
    self._kernel_size = kernel_size
    self._use_layer_norm = use_layer_norm

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_spatial_size,
                                       self._input_spatial_size,
                                       self._input_features_size),
                                      jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = self._input_spatial_size * self._upscale_factor
    return types.SpecDict({
        self._output_name: specs.Array(
            (output_size, output_size, self._output_features_size), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    if self._use_layer_norm:
      x = util.visual_layer_norm(x)
    x = jax.nn.relu(x)
    x = hk.Conv2DTranspose(
        output_channels=self._output_features_size,
        kernel_shape=self._kernel_size,
        stride=self._upscale_factor)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Resnet(modular.BatchedComponent):
  """Resnet processing of the visual stream."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: int,
               input_features_size: int,
               num_resblocks: int,
               kernel_size: int = 3,
               use_layer_norm: bool = True,
               num_hidden_feature_planes: Optional[int] = None,
               name: Optional[str] = None):
    """Initializes Resnet module..

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      output_name: The name to give to the output, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      input_spatial_size: The spatial size of the input.
      input_features_size: The number of feature planes of the input.
      num_resblocks: The number of residual blocks.
      kernel_size: The size of the convolution kernel to use.
      use_layer_norm: Whether to use layer normalization.
      num_hidden_feature_planes: Optional number of feature planes in the
        hidden layers of the residual blocks. If None, the number of feature
        planes of the input is used.
      name: The name of this component.
    """
    super().__init__(name)
    self._input_name = input_name
    self._output_name = output_name
    self._input_spec = specs.Array(
        (input_spatial_size, input_spatial_size, input_features_size),
        jnp.float32)
    self._num_resblocks = num_resblocks
    self._kernel_size = kernel_size
    self._use_layer_norm = use_layer_norm
    self._num_hidden_feature_planes = num_hidden_feature_planes

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({self._input_name: self._input_spec})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({self._output_name: self._input_spec})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for _ in range(self._num_resblocks):
      x = util.VisualResblock(
          kernel_size=self._kernel_size,
          hidden_size=self._num_hidden_feature_planes,
          use_layer_norm=self._use_layer_norm)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class ToVector(modular.BatchedComponent):
  """Strided convolutions (downscales) followed by a linear layer."""

  def __init__(self,
               input_name: types.StreamType,
               output_name: types.StreamType,
               input_spatial_size: int,
               input_features_size: int,
               vector_stream_size: int,
               hidden_feature_sizes: Sequence[int],
               downscale_factor: int = 2,
               kernel_size: int = 4,
               use_layer_norm: bool = True,
               name: Optional[str] = None):
    """Initializes ToVector module..

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      output_name: The name to give to the output, of shape [output_size] and
        dtype float32.
      input_spatial_size: The spatial size of the input.
      input_features_size: The number of feature planes of the input.
      vector_stream_size: The size of the output (1d vector representation).
      hidden_feature_sizes: The list of number of feature planes in the
        convolutional hidden layers, before reshaping into a single vector.
        Each convolution is strided, decreasing the spatial resolution.
      downscale_factor: The downscale factor of each strided convolution.
      kernel_size: The size of the convolution kernel to use. Note that with
        downsampling, a `kernel_size` not multiple of the `downscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
      use_layer_norm: Whether to use layer normalization.
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._output_name = output_name
    if input_spatial_size % (downscale_factor ** len(hidden_feature_sizes)):
      raise ValueError(
          f'input_spatial_size ({input_spatial_size}) must be a multiple of '
          f'downscale_factor ({downscale_factor}) to the power of '
          f'len(hidden_feature_sizes) ({len(hidden_feature_sizes)}).')
    self._input_spatial_size = input_spatial_size
    self._input_features_size = input_features_size
    self._vector_stream_size = vector_stream_size
    self._hidden_feature_sizes = hidden_feature_sizes
    self._use_layer_norm = use_layer_norm
    self._downscale_factor = downscale_factor
    self._kernel_size = kernel_size

  @property
  def input_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._input_name: specs.Array((self._input_spatial_size,
                                       self._input_spatial_size,
                                       self._input_features_size),
                                      jnp.float32)})

  @property
  def output_spec(self) -> types.SpecDict:
    return types.SpecDict({
        self._output_name: specs.Array((self._vector_stream_size,), jnp.float32)
    })

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    x = inputs[self._input_name]
    for num_hidden_features in self._hidden_feature_sizes:
      if self._use_layer_norm:
        x = util.visual_layer_norm(x)
      x = jax.nn.relu(x)
      x = hk.Conv2D(
          output_channels=num_hidden_features,
          kernel_shape=self._kernel_size,
          stride=self._downscale_factor)(x)
    x = jnp.reshape(x, [-1])
    if self._use_layer_norm:
      x = util.vector_layer_norm(x)
    x = jax.nn.relu(x)
    x = hk.Linear(output_size=self._vector_stream_size)(x)
    outputs = types.StreamDict({self._output_name: x})
    return outputs, {}


class Logits(modular.BatchedComponent):
  """Visual logits: generate a 2d headmap."""

  def __init__(self,
               input_name: types.StreamType,
               input_spatial_size: int,
               input_features_size: int,
               logits_output_name: types.StreamType,
               mask_output_name: types.StreamType,
               upscale_factor: int,
               kernel_size: int,
               use_layer_norm: bool = True,
               use_depth_to_space: bool = False,
               name: Optional[str] = None):
    """Initializes Logits module..

    Args:
      input_name: The name of the input to use, of shape
        [input_spatial_size, input_spatial_size, input_features_size] and dtype
        float32.
      input_spatial_size: The spatial size of the input.
      input_features_size: The number of feature planes of the input.
      logits_output_name: The name to give to the output for the logits,
        of shape [(input_spatial_size*upscale_factor) ** 2] and dtype float32.
      mask_output_name: The name to give to the output for the mask,
        of shape [(input_spatial_size*upscale_factor) ** 2] and dtype bool.
      upscale_factor: The upscale factor to apply to the input.
      kernel_size: The size of the convolution kernel to use. Note that with
        upsampling, a `kernel_size` not multiple of the `upscale_factor`
        will result in a checkerboard pattern, so it is not recommended.
        If `use_depth_to_space` is set to True, then the kernel size is actually
        required to be a multiple of upscale_factor.
      use_layer_norm: Whether to use layer normalization.
      use_depth_to_space: If False, strided convolutions are used. If True, a
        regular convolution (stride=1) is used, and the output is reshaped to
        produce an upsampled result. The operations are equivalent, but result
        in reshuffled weight vectors, so the trained models are not
        interchangeable. Using this option gives a performance boost on some
        hardwares (GPUs).
      name: The name of this component.
    """
    super().__init__(name=name)
    self._input_name = input_name
    self._input_spatial_size = input_spatial_size
    self._input_features_size = input_features_size
    self._logits_output_name = logits_output_name
    self._mask_output_name = mask_output_name
    if upscale_factor < 1:
      raise ValueError(
          f'upscale_factor must be > 0, but was set to {upscale_factor}.')
    self._upscale_factor = upscale_factor
    self._kernel_size = kernel_size
    if use_depth_to_space and (kernel_size % upscale_factor):
      raise ValueError(
          f'upscale_factor ({upscale_factor}) must be a multiple of kernel_size'
          f' ({kernel_size}) when use_depth_to_space is set to True.')
    self._use_layer_norm = use_layer_norm
    self._use_depth_to_space = use_depth_to_space

  @property
  def input_spec(self) -> types.SpecDict:
    camera_size = self._input_spatial_size * self._upscale_factor
    return types.SpecDict({
        self._input_name: specs.Array((self._input_spatial_size,
                                       self._input_spatial_size,
                                       self._input_features_size),
                                      jnp.float32),
        ('observation', 'camera'): specs.Array(
            (camera_size, camera_size), jnp.int32),
        ('action', 'function'): specs.Array((), jnp.int32)})

  @property
  def output_spec(self) -> types.SpecDict:
    output_size = self._input_spatial_size * self._upscale_factor
    return types.SpecDict({
        self._logits_output_name: specs.Array(
            (output_size * output_size,), jnp.float32),
        self._mask_output_name: specs.Array(
            (output_size * output_size,), jnp.bool_)})

  def _forward(self, inputs: types.StreamDict) -> modular.ForwardOutputType:
    # mask: with some functions arguments, the agent can only click on the
    # screen (ie. in the camera view).
    camera = inputs['observation', 'camera']
    function_arg = inputs['action', 'function']
    all_camera_only_functions = camera_masks.get_on_camera_only_functions_pt()
    all_not_camera_only_functions = jnp.asarray(
        np.logical_not(all_camera_only_functions), dtype=jnp.bool_)
    is_not_camera_only = all_not_camera_only_functions[function_arg]
    camera_mask = jnp.logical_or(is_not_camera_only[jnp.newaxis], camera)
    camera_mask = jnp.reshape(camera_mask, [-1])

    x = inputs[self._input_name]
    if self._use_layer_norm:
      x = util.visual_layer_norm(x)
    x = jax.nn.relu(x)
    if self._upscale_factor > 1:
      if self._use_depth_to_space:
        output_channels = self._upscale_factor * self._upscale_factor
        kernel_shape = self._kernel_size // self._upscale_factor
        x = hk.Conv2D(
            output_channels=output_channels, kernel_shape=kernel_shape)(x)
        x = pix.depth_to_space(x, self._upscale_factor)[:, :, 0]
      else:
        x = hk.Conv2DTranspose(  # upscale
            output_channels=1,
            kernel_shape=self._kernel_size,
            stride=self._upscale_factor)(x)[:, :, 0]
    else:
      x = hk.Conv2D(
          output_channels=1,
          kernel_shape=self._kernel_size)(x)[:, :, 0]
    logits = jnp.reshape(x, [-1])
    logits = sample.mask_logits(logits, camera_mask)

    outputs = types.StreamDict({self._logits_output_name: logits,
                                self._mask_output_name: camera_mask})
    return outputs, {}

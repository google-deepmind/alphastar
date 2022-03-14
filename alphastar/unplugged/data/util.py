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

"""Utility functions for data."""

import queue
import threading
from typing import Any, Callable, Generator, Iterable, Mapping, List, Optional, TypeVar, Union
import zlib

from absl import logging
from alphastar import types
import apache_beam as beam
from apache_beam import coders
import chex
import dm_env
from dm_env import specs
import jax
import jax.numpy as jnp
import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
import tree


def get_input_spec(
    obs_spec: types.SpecDict,
    behaviour_features_spec: Optional[types.SpecDict] = None,
    prev_features_spec: Optional[types.SpecDict] = None
    ) -> types.SpecDict:
  """Get full input spec given obs_spec and optional behaviour_action_spec."""
  spec = types.SpecDict()
  spec['step_type'] = specs.BoundedArray(
      (), jnp.int32, minimum=0, maximum=int(max(dm_env.StepType)))
  spec['observation'] = obs_spec
  if behaviour_features_spec:
    spec['behaviour_features'] = behaviour_features_spec
  if prev_features_spec:
    spec['prev_features'] = prev_features_spec
  return spec


def get_dummy_observation(input_spec: types.SpecDict,
                          batch_size: int,
                          unroll_len: Optional[int]) -> types.StreamDict:
  """Return a dummy observation matching the spec."""
  if unroll_len is None:
    def zeros_like_spec(spec):
      return jnp.zeros((batch_size,) + spec.shape, spec.dtype)
  else:
    def zeros_like_spec(spec):
      return jnp.zeros((batch_size, unroll_len) + spec.shape, spec.dtype)
  return jax.tree_map(zeros_like_spec, input_spec)


class FeatureSpec:
  """A description of the features used in the dataset.

  Feature values are numpy arrays which are serialized as bytes.
  """

  def __init__(self, dtype, shape):
    """Initializes a FeatureSpec.

    Args:
      dtype: dtype convertible with tf.as_dtype
      shape: shape convertible with tf.TensorShape.Make sure at most one
        dimension is None as features can be reshaped at most along one unknown
        dimension.
    """
    self._dtype = tf.as_dtype(dtype)
    self._numpy_scalar_type = self.dtype.as_numpy_dtype
    self._numpy_dtype = np.dtype(self._numpy_scalar_type).newbyteorder('<')
    self._shape = tf.TensorShape(shape)
    self._reshape_arg = [
        d.value if d.value is not None else -1 for d in self.shape.dims
    ]

  @property
  def dtype(self):
    """Tf datatype for this feature."""
    return self._dtype

  @property
  def shape(self):
    """Tensorshape for this feature."""
    return self._shape

  def _check_shape(self, array):
    if not self.shape.is_compatible_with(array.shape):
      raise ValueError('Incompatible shape between what is expected :'
                       f' {self.shape} and that of the array : {array.shape}.')

  def value_to_string(self, value) -> str:
    """Serializes value according to the spec for this feature.

    Args:
      value: The value to serialize.

    Returns:
      A string/bytes.

    Raises:
      TypeError: if valus cannot be cast into the specified type.
    """
    array = np.asarray(value)
    self._check_shape(array)
    if array.dtype != self._numpy_dtype:
      raise ValueError('Dtype {} do not match {}'.format(
          array.dtype, self._numpy_dtype))
    return array.tobytes()

  def string_to_value(self, byte_string):
    """Deserializes a byte string according to the spec of the feature.

    Args:
       byte_string: byte_string to deserialize

    Returns:
      A value.
    """
    raw_value = tf.io.decode_raw(byte_string, self.dtype)
    return tf.reshape(raw_value, self._reshape_arg)


def _bytes_feature(value):
  """Returns a bytes_list from a string / byte."""
  if not isinstance(value, List):
    value = [value]
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))


class TFExampleCoder(coders.Coder):
  """Helps encodes/decode rows of the dataset to/from numpy arrays/scalars."""

  def __init__(self, features: Mapping[str, Any], compress: bool):
    self._features = features
    self._compress = compress
    self._flat_features_with_paths = tree.flatten_with_path(features)
    self._flat_features = [feat for _, feat in self._flat_features_with_paths]
    flat_feature_paths = [path for path, _ in self._flat_features_with_paths]
    self._string_flat_features = {
        '/'.join(path): tf.io.FixedLenFeature((), tf.string)
        for path in flat_feature_paths
    }

    self._success_counter = beam.metrics.Metrics.counter(
        'SerializeFn', 'Success')
    self._failure_counter = beam.metrics.Metrics.counter(
        'SerializeFn', 'Failure')

  def convert_to_proto(self, row: Mapping[str, Any]) -> tf.train.Example:
    """Encodes a nest of numpy scalars/arrays to a byte string."""
    tree.assert_same_structure(row, self._features)
    flat_row = tree.flatten(row)
    features = {}
    for (path, feature), value in zip(self._flat_features_with_paths, flat_row):
      try:
        value_str = feature.value_to_string(value)
        if self._compress:
          value_str = zlib.compress(value_str)
        value_feature = _bytes_feature(value_str)
        features['/'.join(path)] = value_feature
      except (TypeError, ValueError) as e:
        e.args = (f'Incompatible data when encoding feature at path {path}. '
                  'Value : {value}')
        raise
    example_proto = tf.train.Example(
        features=tf.train.Features(feature=features))
    return example_proto

  def serialize(self, proto: tf.train.Example) -> Optional[str]:
    """Serializes a given example to a proto string."""
    try:
      serialized_proto = proto.SerializeToString()
      self._success_counter.inc()
    except ValueError as e:
      self._failure_counter.inc()
      logging.info('2 GB Protobuf limit hit : %s', e)
      serialized_proto = None
    return serialized_proto

  def encode(self, row: Mapping[str, Any]) -> Optional[str]:
    """Serializes a row of data into a proto."""
    serialized_proto = self.serialize(self.convert_to_proto(row))
    if len(serialized_proto) > 1:
      # Until b/203641663 is fixed, we skip episodes which fail to reconstruct.
      # TODO(b/208420811): Skip if any array is >2gb instead.
      try:
        recons = self.decode(serialized_proto)
        chex.assert_trees_all_close(row, recons)
      except tf.errors.InvalidArgumentError as e:
        logging.info('Failed to compress episode correctly. %s', e.message)
        serialized_proto = None
    return serialized_proto

  def decode(self, example_string: str) -> Mapping[str, Any]:
    parsed_example = tf.io.parse_single_example(example_string,
                                                self._string_flat_features)
    values = []
    for (byte_string, feature) in zip(parsed_example.values(),
                                      self._flat_features):
      if self._compress:
        byte_string = tf.io.decode_compressed(
            byte_string, compression_type='ZLIB')
      value_str = feature.string_to_value(byte_string)
      values.append(value_str)
    return tree.unflatten_as(self._features, values)


class Log(beam.DoFn):
  """Identity with logging. Useful for debugging."""

  def __init__(self, prefix):
    self._prefix = prefix

  def process(self, item):
    logging.info('%s: %s', self._prefix, repr(item))
    yield item


def spec_to_feature(x: Union[np.ndarray, chex.Array]) -> FeatureSpec:
  # Ensure scalars are stored as arrays of shape (1,) and not as scalars of
  # shape (), so that they can be batched into proper arrays later.
  if len(x.shape) == 1 and x.shape[0] == 1:
    shape = (None,)
  else:
    shape = (None,) + x.shape
  return FeatureSpec(dtype=x.dtype, shape=shape)


def get_dataset_specs(
    obs_spec,
    make_spec: Callable[..., FeatureSpec] = FeatureSpec
) -> Mapping[str, FeatureSpec]:
  """Generates dataset feature specs from SC2 observation specs."""

  obs_features = jax.tree_map(spec_to_feature, obs_spec)

  return dict(
      step_type=make_spec(dtype=np.int32, shape=(None,)),
      observation=obs_features)

T = TypeVar('T')


def prefetch(iterable: Iterable[T],
             buffer_size: int) -> Generator[T, None, None]:
  """Performs prefetching of elements from an iterable in a separate thread.

  Args:
    iterable: An iterable to prefetch.
    buffer_size: Maximum number of items to prefetch.

  Yields:
    Prefetched elements from the original iterable.
  Raises:
    Any error thrown by the iterable. Note this is not raised inside
      the producer, but after it finishes executing.
  """
  if not buffer_size >= 1:
    raise ValueError('buffer_size should be at least 1.')

  buffer = queue.Queue(maxsize=buffer_size)
  producer_error = []
  end = object()

  def producer():
    """Enques items from iterable on a given thread."""
    try:
      # Build a new iterable for each thread. This is crucial if working with
      # tensorflow datasets because tf.graph objects are thread local.
      for item in iterable:
        buffer.put(item)
    except Exception as e:  # pylint: disable=broad-except
      logging.exception('Error in producer thread  and will be raised in '
                        'the main thread.')
      producer_error.append(e)
    finally:
      buffer.put(end)

  threading.Thread(target=producer, daemon=True).start()

  # Consumer.
  while True:
    value = buffer.get()
    if value is end:
      break
    yield value

  if producer_error:
    raise producer_error[0]


def iterate(ds: tf.data.Dataset):
  yield from prefetch(iter(tfds.as_numpy(ds)), 1)


def split_behaviour_actions(observation: Mapping[str, Any]):
  """Extracts out behaviour actions from the observation."""

  behaviour_action_keys = [
      key for key in observation.keys() if key.startswith('action/')
  ]
  behaviour_actions = {
      key.replace('action/', ''): observation[key]
      for key in behaviour_action_keys
  }
  other_keys = set(observation.keys()) - set(behaviour_action_keys)
  observation = {key: observation[key] for key in other_keys}
  return behaviour_actions, observation


def as_learner_input(raw_input: Mapping[str, Any],
                     use_prev_features: bool = False) -> types.StreamDict:
  """Transform the raw_input into a StreamDict type usable by the learner.

  If use_prev_features is set, the first timestep is used to generate the
  previous features and is discarded from the main data. This means that
  the output will have one less timestep than the input.

  Args:
    raw_input: The raw input to process.
    use_prev_features: Whether to use the first timestep to generate the
      prev_features field.

  Returns:
    A StreamDict containing data usable by the learner.
  """
  behaviour_actions, observation = split_behaviour_actions(
      raw_input['observation'])
  output = types.StreamDict()
  output['step_type'] = raw_input['step_type']
  output['observation'] = types.StreamDict(observation)
  output['behaviour_features', 'action'] = types.StreamDict(behaviour_actions)
  if use_prev_features:
    prev_output = jax.tree_map(lambda x: x[:, :-1], output)
    output = jax.tree_map(lambda x: x[:, 1:], output)
    output['prev_features', 'action'] = prev_output.get(
        ('behaviour_features', 'action'))
  return output

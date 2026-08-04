# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A TensorFlow model, consisting of TF functions and trackable resources."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from neptune_model._src.export.constants import DEFAULT_METHOD_KEY

DEFAULT_MODEL_NAME: str = 'tensorflow_model'


class TensorflowModel:
  """A TensorFlow model, consisting of TF functions and trackable resources.

  Attributes:
    name: The name of the model.
    tf_functions: A mapping from method keys to TF functions. If a single
      callable is provided, it will be converted to a function map with a
      default key.
    trackable_resources: Trackable resources used by the TF functions. Can be a
      tf.Module.
  """

  def __init__(
      self,
      tf_functions: Callable[..., Any] | Mapping[str, Callable[..., Any]],
      trackable_resources: Any | None = None,
      name: str = DEFAULT_MODEL_NAME,
  ):
    if not name:
      raise ValueError('TensorflowModel name must not be empty')
    self._name = name
    if isinstance(tf_functions, Mapping):
      self._tf_functions = tf_functions
    else:
      self._tf_functions = {DEFAULT_METHOD_KEY: tf_functions}
    self._trackable_resources = trackable_resources

  @property
  def name(self) -> str:
    """The name of the model."""
    return self._name

  @property
  def tf_functions(self) -> Mapping[str, Callable[..., Any]]:
    """A mapping from method keys to TF functions."""
    return self._tf_functions

  @property
  def trackable_resources(self) -> Any | None:
    """Trackable resources used by the TF functions."""
    return self._trackable_resources

  def method_keys(self) -> Sequence[str]:
    return list(self._tf_functions.keys())

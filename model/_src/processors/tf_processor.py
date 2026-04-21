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

"""TensorFlow processor for Neptune."""

from collections.abc import Callable
import dataclasses
from typing import Any

import jaxtyping

from neptune_model._src import tf2npt
from neptune_model._src.core import saveable


@dataclasses.dataclass(frozen=True)
class PreparedTfProcessor:
  """TensorFlow processor that is prepared for export."""

  name: str
  prepared_function: tf2npt.PreparedFunction

  @property
  def input_signature(self) -> jaxtyping.PyTree:
    return self.prepared_function.serializable_fn.input_signature

  @property
  def output_signature(self) -> jaxtyping.PyTree:
    return self.prepared_function.serializable_fn.output_signature

  @property
  def weights_name(self) -> str | None:
    return None

  def to_npt_saveable(self) -> saveable.Saveable:
    return self.prepared_function.serializable_fn


@dataclasses.dataclass(frozen=True)
class TfProcessor:
  """TensorFlow processor for Neptune."""

  name: str
  fn: Callable[..., Any]

  def prepare(
      self,
      input_signature: jaxtyping.PyTree,
      **kwargs: Any,
  ) -> PreparedTfProcessor:
    """Prepares the TensorFlow processor for export."""

    if not self.name:
      raise ValueError('TfProcessor name must not be empty')

    prepared_function = tf2npt.prepare_function(
        self.name,
        self.fn,
        input_signature=input_signature,
        suppress_x64_outputs=kwargs.get('suppress_x64_outputs', False),
    )
    return PreparedTfProcessor(self.name, prepared_function)

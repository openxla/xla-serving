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

"""TensorFlow processor for XSE."""

from collections.abc import Callable, Mapping
import dataclasses
from typing import Any

import jaxtyping

from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.core import saveable
from xla_saved_model._src.processors import processor


def _require_tf2xsm():
  try:
    from xla_saved_model._src import tf2xsm  # pylint: disable=g-import-not-at-top
  except ImportError as e:
    raise ImportError(
        'Failed to import tf2xsm. Please make sure that TF support is enabled.'
    ) from e

  return tf2xsm


def save_tf_functions(
    export_path: str,
    tf_prepared_functions: Mapping[str, Any],
    trackable_resources: Any = None,
) -> dict[str, persistence_lib.GlobalSupplemental]:
  """Saves TensorFlow functions to a TF SavedModel."""
  return _require_tf2xsm().save_tf_functions(
      export_path,
      tf_prepared_functions,
      trackable_resources=trackable_resources,
  )


def find_ragged_inputs_indices_if_any(
    input_signature: jaxtyping.PyTree,
) -> list[tuple[int, int]]:
  """Finds the indices of ragged inputs in the input signature."""
  return _require_tf2xsm().type_conversion.find_ragged_inputs_indices_if_any(
      input_signature
  )


@dataclasses.dataclass(frozen=True)
class PreparedTfProcessor(processor.ProcessorDefaults):
  """TensorFlow processor that is prepared for export."""

  name: str
  prepared_function: Any

  @property
  def input_signature(self) -> jaxtyping.PyTree:
    return self.prepared_function.serializable_fn.input_signature

  @property
  def output_signature(self) -> jaxtyping.PyTree:
    return self.prepared_function.serializable_fn.output_signature

  def to_xsm_saveable(self) -> saveable.Saveable:
    return self.prepared_function.serializable_fn


@dataclasses.dataclass(frozen=True)
class TfProcessor:
  """TensorFlow processor for XSE."""

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

    prepared_function = _require_tf2xsm().prepare_function(
        self.name,
        self.fn,
        input_signature=input_signature,
        suppress_x64_outputs=kwargs.get('suppress_x64_outputs', False),
    )
    return PreparedTfProcessor(self.name, prepared_function)

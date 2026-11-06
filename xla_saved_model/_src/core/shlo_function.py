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

"""Functions backed by StableHLO."""

import abc
from collections.abc import Mapping
from dataclasses import dataclass  # pylint: disable=g-importing-member

import jaxtyping

from xla_saved_model._src.core import function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import unstructured_data


class ShloFunctionSupplementalInfo(abc.ABC):
  """Interface for ShloFunction's supplemental_info."""

  @abc.abstractmethod
  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    """Serializes it to an `UnstructuredDataWithExtName`."""


@dataclass
class ShloFunction(function.Function):
  """A StableHLO function.

  Attributes:
    mlir_module_serialized: the serialized StableHLO.
    calling_convention_version: the calling convention version of the serialized
      StableHLO. See more versioning details at
      [this](https://github.com/google/jax/blob/main/jax/experimental/jax2tf/README.md#native-serialization-versions)
      and
      [this](https://jax.readthedocs.io/en/latest/export/export.html#calling-convention-versions)
      doc.
    module_kept_var_idx: the indices of the arguments that will actually be
      passed to the StableHLO. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for its use in the calling convention.
    lowering_platforms: a tuple containing at least one of 'tpu', 'cpu', 'cuda',
      'rocm'. See this
      [doc](https://jax.readthedocs.io/en/latest/export/export.html#module-calling-convention)
      for the calling convention for when there are multiple lowering platforms.
    supplemental_info: an opaque supplemental to the function. It can be used to
      carry e.g. JAX-specific information.
    physical_in_dtypes: physical input dtypes, which (if given) will override
      `input_signature`.
    physical_out_dtypes: physical output dtypes, which (if given) will override
      `output_signature`.
    num_input_aliased_output: the number of trailing arguments that are donated
      output buffers rather than caller-supplied inputs. A function exported for
      output donation ends its argument list with one extra argument per output,
      aliased to that output; those arguments carry no input data and are
      supplied by the runtime. Zero means no output donation.
    optimized_input_signature: optional post-optimization physical input
      signature when export-time TPU I/O shape optimizations were applied.
    optimized_output_signature: optional post-optimization physical output
      signature when export-time TPU I/O shape optimizations were applied.
  """

  mlir_module_serialized: bytes
  calling_convention_version: int
  module_kept_var_idx: tuple[int, ...]
  lowering_platforms: tuple[str, ...]

  supplemental_info: Mapping[str, ShloFunctionSupplementalInfo]

  physical_in_dtypes: tuple[shlo_type.ShloDType | None, ...]
  physical_out_dtypes: tuple[shlo_type.ShloDType | None, ...]

  num_input_aliased_output: int = 0
  optimized_input_signature: (
      jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None
  ) = None
  optimized_output_signature: (
      jaxtyping.PyTree[shlo_type.ShloTensorSpec] | None
  ) = None

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

"""ExportableTransform implementation for UnpackHostBufferVectorTransform."""

from typing import Any, Sequence

import numpy as np

from data.kernels.cpu_embedding import unpack_host_buffer_vector_transform_pybind
from data.processor.python import shape_dtype_struct
from data.processor.python import transform


class UnpackHostBufferVectorModule(transform.ExportableTransform):
  """Python wrapper for C++ UnpackHostBufferVectorTransform."""

  def __init__(
      self,
      output_count: int,
      output_shapes: Sequence[Sequence[int | None]],
      dtype: Any = np.float32,
  ):
    super().__init__()
    self._output_count = output_count
    self._dtype = np.dtype(dtype)
    replaced_shapes = (
        shape_dtype_struct._replace_deprecated_negative_one_shapes(  # pylint: disable=protected-access
            output_shapes, "UnpackHostBufferVectorModule output_shapes"
        )
    )
    self._output_shapes = [
        list(shape_dtype_struct.ShapeDtypeStruct(s, self._dtype).shape)
        for s in replaced_shapes
    ]

  def initialize(self, input_signature: Any) -> None:
    self._transform = unpack_host_buffer_vector_transform_pybind.UnpackHostBufferVectorTransform(
        self._output_count
    )

  def get_output_signature(self, input_signature: Any) -> Any:
    specs = [
        shape_dtype_struct.ShapeDtypeStruct(shape=shape, dtype=self._dtype)
        for shape in self._output_shapes
    ]
    if len(specs) == 1:
      return specs[0]
    return specs

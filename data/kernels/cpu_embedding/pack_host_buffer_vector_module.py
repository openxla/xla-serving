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

"""ExportableTransform implementation for PackHostBufferVectorTransform."""

from typing import Any

from data.kernels.cpu_embedding import pack_host_buffer_vector_transform_pybind
from data.processor.python import composite_processor_pybind
from data.processor.python import transform


class PackHostBufferVectorModule(transform.ExportableTransform):
  """Python wrapper for C++ PackHostBufferVectorTransform."""

  def __init__(self, input_count: int):
    super().__init__()
    self._input_count = input_count

  def initialize(self, input_signature: Any) -> None:
    self._transform = (
        pack_host_buffer_vector_transform_pybind.PackHostBufferVectorTransform(
            self._input_count
        )
    )

  def get_output_signature(self, input_signature: Any) -> Any:
    return composite_processor_pybind.TypeInfo.host_buffer_vector()

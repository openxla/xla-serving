"""Native implementation of PackHostBufferVectorModule."""

from typing import Any

from data.kernels.cpu_embedding import pack_host_buffer_vector_transform_pybind
from data.processor.python import composite_processor_pybind


class PackHostBufferVectorModule:
  """Python wrapper for C++ PackHostBufferVectorTransform."""

  def __init__(self, input_count: int):
    self._input_count = input_count
    self._transform = (
        pack_host_buffer_vector_transform_pybind.PackHostBufferVectorTransform(
            input_count
        )
    )

  def initialize(self, input_signature: Any = None) -> None:
    pass

  @property
  def transform(self) -> composite_processor_pybind.TransformBase:
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return {}

  def get_output_signature(self, input_signature: Any) -> Any:
    return composite_processor_pybind.TypeInfo.host_buffer_vector()

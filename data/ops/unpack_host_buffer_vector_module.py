"""Native implementation of UnpackHostBufferVectorModule."""

from typing import Any, Sequence

import jax

from data.kernels.cpu_embedding import unpack_host_buffer_vector_transform_pybind
from data.processor.python import composite_processor_pybind


class UnpackHostBufferVectorModule:
  """Python wrapper for C++ UnpackHostBufferVectorTransform."""

  def __init__(self, output_count: int, output_shapes: Sequence[Sequence[int]]):
    self._output_count = output_count
    self._output_shapes = [list(s) for s in output_shapes]
    self._transform = unpack_host_buffer_vector_transform_pybind.UnpackHostBufferVectorTransform(
        output_count
    )

  def initialize(self, input_signature: Any = None) -> None:
    pass

  @property
  def transform(self) -> composite_processor_pybind.TransformBase:
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return {}

  def get_output_signature(self, input_signature: Any) -> Any:
    return [
        jax.ShapeDtypeStruct(shape=tuple(shape), dtype=jax.numpy.float32)
        for shape in self._output_shapes
    ]

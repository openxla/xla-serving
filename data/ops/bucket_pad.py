# Copyright 2024 The Neptune Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Bucket padding op and transform module for dynamic-to-static tensor padding."""

from collections.abc import Sequence
from typing import Any, Union
from data.kernels.bucket_pad import bucket_pad_transform_pybind
from data.processor.cc.utils import default_value_pb2
from data.processor.python import transform


class BucketPadTransform(transform.ExportableTransform):
  """Pads a tensor along a specified axis to the smallest pre-determined bucket dimension D_fixed >= C."""

  def __init__(
      self,
      target_dimensions: Sequence[int],
      axis: int = -1,
      pad_value: Union[int, float, bool, None] = None,
  ):
    super().__init__(target_dimensions, axis, pad_value)
    self._target_dimensions = list(target_dimensions)
    self._axis = axis
    self._pad_value = pad_value

    pad_proto = default_value_pb2.DefaultValue()
    if isinstance(pad_value, bool):
      pad_proto.bool_value = pad_value
    elif isinstance(pad_value, int):
      pad_proto.int_value = pad_value
    elif isinstance(pad_value, float):
      pad_proto.float_value = pad_value

    self._transform = bucket_pad_transform_pybind.BucketPadTransform(
        self._target_dimensions, self._axis, pad_proto
    )

  def initialize(self, input_signature: Any) -> None:
    pass

  def __call__(self, input_tensor: Any) -> Any:
    return super().__call__(input_tensor)


def bucket_pad(
    input_tensor: Any,
    target_dimensions: Sequence[int],
    axis: int = -1,
    pad_value: Union[int, float, bool, None] = None,
) -> Any:
  """Functional op for bucket padding."""
  t = BucketPadTransform(target_dimensions, axis=axis, pad_value=pad_value)
  return t(input_tensor)


__all__ = [
    "BucketPadTransform",
    "bucket_pad",
]

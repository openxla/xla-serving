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

"""Core types for XLA SavedModel."""

from collections.abc import Sequence
import dataclasses
import enum
from typing import Any, Optional, TypeAlias

import ml_dtypes
import numpy as np

from xla_saved_model.protos import type_pb2
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import

Placement: TypeAlias = type_pb2.Placement
Sharding: TypeAlias = xla_data_pb2.OpSharding
Layout: TypeAlias = xla_data_pb2.LayoutProto
ShloDimSize: TypeAlias = Optional[int]
ShloShape: TypeAlias = Optional[Sequence[ShloDimSize]]


# pylint: disable=invalid-name
# Copied from /third_party/py/jax/_src/export/serialization.fbs
class ShloDType(enum.Enum):  # pylint: disable=missing-class-docstring
  bool = 0
  i8 = 1
  i16 = 2
  i32 = 3
  i64 = 4
  ui8 = 5
  ui16 = 6
  ui32 = 7
  ui64 = 8
  f16 = 9
  f32 = 10
  f64 = 11
  c64 = 12
  c128 = 13

  bf16 = 14

  i4 = 15
  ui4 = 16

  f8_e4m3b11fnuz = 17
  f8_e4m3fn = 18
  f8_e4m3fnuz = 19
  f8_e5m2 = 20
  f8_e5m2fnuz = 21

  str = 100


_NP_DTYPE_TO_SHLO_DTYPE: dict[np.dtype[Any], ShloDType] = {
    np.dtype(np.bool_): ShloDType.bool,
    np.dtype(np.int8): ShloDType.i8,
    np.dtype(np.int16): ShloDType.i16,
    np.dtype(np.int32): ShloDType.i32,
    np.dtype(np.int64): ShloDType.i64,
    np.dtype(np.uint8): ShloDType.ui8,
    np.dtype(np.uint16): ShloDType.ui16,
    np.dtype(np.uint32): ShloDType.ui32,
    np.dtype(np.uint64): ShloDType.ui64,
    np.dtype(np.float16): ShloDType.f16,
    np.dtype(np.float32): ShloDType.f32,
    np.dtype(np.float64): ShloDType.f64,
    np.dtype(np.complex64): ShloDType.c64,
    np.dtype(np.complex128): ShloDType.c128,
    # ml_dtypes
    np.dtype(ml_dtypes.bfloat16): ShloDType.bf16,
    np.dtype(ml_dtypes.int4): ShloDType.i4,
    np.dtype(ml_dtypes.uint4): ShloDType.ui4,
    np.dtype(ml_dtypes.float8_e4m3b11fnuz): ShloDType.f8_e4m3b11fnuz,
    np.dtype(ml_dtypes.float8_e4m3fn): ShloDType.f8_e4m3fn,
    np.dtype(ml_dtypes.float8_e4m3fnuz): ShloDType.f8_e4m3fnuz,
    np.dtype(ml_dtypes.float8_e5m2): ShloDType.f8_e5m2,
    np.dtype(ml_dtypes.float8_e5m2fnuz): ShloDType.f8_e5m2fnuz,
}

_SHLO_DTYPE_TO_NP_DTYPE = {v: k for k, v in _NP_DTYPE_TO_SHLO_DTYPE.items()}
_SHLO_DTYPE_TO_NP_DTYPE[ShloDType.str] = np.dtype(np.object_)


def np_dtype_to_shlo_dtype(dtype: np.dtype[Any]) -> ShloDType:
  if dtype.kind in ("U", "S", "O"):
    return ShloDType.str
  return _NP_DTYPE_TO_SHLO_DTYPE[dtype]


def shlo_dtype_to_np_dtype(dtype: ShloDType) -> np.dtype[Any]:
  return _SHLO_DTYPE_TO_NP_DTYPE[dtype]


@dataclasses.dataclass
class ShloTensorSpec:
  """A specification for the shape, dtype, sharding, and layout of a StableHLO tensor.

  Attributes:
    shape: The shape of the tensor.
    dtype: The dtype of the tensor.
    sharding: The sharding of the tensor. None means unspecified sharding.
    layout: The layout of the tensor. None means the default layout is used.
    name: The name of the tensor.
    default_val: Optional default fallback value for function parameters.
    placement: Optional placement enum (e.g. type_pb2.HOST or type_pb2.DEVICE).
    upper_bound_dims: Optional indices of dimensions whose size in `shape` is an
      upper bound rather than the exact size at serving time. The function is
      compiled for `shape`; a caller may pass fewer elements along such a
      dimension, and the runtime transfers only the elements present (dynamic
      DMA), leaving the rest of the device buffer uninitialized. Each listed
      dimension must have a concrete size. The runtime currently only supports
      dimension 0 of model function inputs.
  """

  shape: ShloShape
  dtype: ShloDType | Any
  sharding: Sharding | None = None
  layout: Layout | None = None
  name: str | None = None
  default_val: Any | None = None
  placement: Placement | None = None
  upper_bound_dims: Sequence[int] | None = None

  def __post_init__(self) -> None:
    if self.shape is not None:
      object.__setattr__(self, "shape", tuple(self.shape))
    if self.upper_bound_dims is not None:
      upper_bound_dims = tuple(self.upper_bound_dims)
      if not upper_bound_dims:
        upper_bound_dims = None
      else:
        if self.shape is None:
          raise ValueError(
              "upper_bound_dims requires a shape of known rank, got shape=None"
          )
        for dim in upper_bound_dims:
          if not 0 <= dim < len(self.shape):
            raise ValueError(
                f"upper_bound_dims entry {dim} is out of range for shape"
                f" {self.shape}"
            )
          if self.shape[dim] is None:
            raise ValueError(
                f"upper_bound_dims entry {dim} refers to a dimension of"
                f" unknown size in shape {self.shape}; an upper bound requires"
                " a concrete size"
            )
      object.__setattr__(self, "upper_bound_dims", upper_bound_dims)
    if not isinstance(self.dtype, ShloDType):
      raw_dt = self.dtype
      if hasattr(raw_dt, "as_numpy_dtype"):
        attr = getattr(raw_dt, "as_numpy_dtype")
        raw_dt = (
            attr()
            if callable(attr) and not isinstance(attr, (type, np.dtype))
            else attr
        )
      np_dt = np.dtype(raw_dt)
      object.__setattr__(self, "dtype", np_dtype_to_shlo_dtype(np_dt))

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

"""Type conversion helpers for tf2npt."""

from typing import Any

import numpy as np
import tensorflow as tf

from neptune_model._src.core import shlo_type


def tf_dtype_from(dtype: shlo_type.ShloDType | tf.DType) -> Any:
  match dtype:

    case shlo_type.ShloDType.bf16:
      return tf.bfloat16

    case shlo_type.ShloDType.str:
      return tf.string

    case shlo_type.ShloDType():
      return shlo_type.shlo_dtype_to_np_dtype(dtype)

    case _:
      return dtype


def shlo_dtype_from(
    dtype: tf.DType | shlo_type.ShloDType,
) -> shlo_type.ShloDType:
  match dtype:

    case tf.bfloat16:
      return shlo_type.ShloDType.bf16

    case tf.string:
      return shlo_type.ShloDType.str

    case tf.resource | tf.variant:
      raise ValueError(f"Can't convert TF dtype {dtype} to a Neptune dtype")

    case tf.DType():
      return shlo_type.np_dtype_to_shlo_dtype(np.dtype(dtype.as_numpy_dtype()))

    case _:
      return dtype


def tf_type_spec_from(
    spec: shlo_type.ShloTensorSpec | tf.TypeSpec,
) -> tf.TypeSpec:
  match spec:

    case tf.TypeSpec():
      return spec

    case _:
      return tf.TensorSpec(
          shape=spec.shape,
          dtype=tf_dtype_from(spec.dtype),
          name=getattr(spec, "name", None),
      )


def shlo_tensor_spec_from(
    spec: tf.TensorSpec | shlo_type.ShloTensorSpec,
) -> shlo_type.ShloTensorSpec | tuple[shlo_type.ShloTensorSpec, ...]:
  """Converts a TF type spec to a ShloTensorSpec."""

  match spec:

    case tf.TypeSpec():
      if not isinstance(spec, tf.TensorSpec):
        # A composite type spec (RaggedTensorSpec, etc.)
        expanded = tf.nest.flatten(spec, expand_composites=True)
        return tuple(shlo_tensor_spec_from(s) for s in expanded)

      # A plain (non-composite) TensorSpec.
      return shlo_type.ShloTensorSpec(
          shape=tuple(spec.shape.as_list())
          if spec.shape.rank is not None
          else None,
          dtype=shlo_dtype_from(spec.dtype),
          name=getattr(spec, "name", None),
      )

    case _:
      return spec

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


def to_tf_dtype(dtype: shlo_type.ShloDType | tf.DType) -> Any:
  if dtype == shlo_type.ShloDType.bf16:
    return tf.bfloat16
  if dtype == shlo_type.ShloDType.str:
    return tf.string
  if isinstance(dtype, shlo_type.ShloDType):
    return shlo_type.shlo_dtype_to_np_dtype(dtype)
  return dtype


def to_tf_spec(
    input_spec: shlo_type.ShloTensorSpec | tf.TensorSpec,
) -> tf.TensorSpec:
  if isinstance(input_spec, tf.TensorSpec):
    return input_spec
  if hasattr(input_spec, "name"):
    name = input_spec.name
  else:
    name = None
  return tf.TensorSpec(
      shape=input_spec.shape,
      dtype=to_tf_dtype(input_spec.dtype),
      name=name,
  )


def to_npt_dtype(tf_dtype: tf.DType) -> shlo_type.ShloDType:
  if tf_dtype == tf.bfloat16:
    return shlo_type.ShloDType.bf16
  if tf_dtype == tf.string:
    return shlo_type.ShloDType.str
  if tf_dtype in (tf.resource, tf.variant):
    raise ValueError(f"Can't convert TF dtype {tf_dtype} to a Neptune dtype")
  return shlo_type.np_dtype_to_shlo_dtype(np.dtype(tf_dtype.as_numpy_dtype()))


def to_npt_spec(
    input_spec: tf.TensorSpec | shlo_type.ShloTensorSpec,
) -> shlo_type.ShloTensorSpec:
  if isinstance(input_spec, shlo_type.ShloTensorSpec):
    return input_spec

  if input_spec.shape.rank is None:
    shape = None
  else:
    shape = tuple(input_spec.shape.as_list())

  return shlo_type.ShloTensorSpec(
      shape=shape,
      dtype=to_npt_dtype(input_spec.dtype),
      name=input_spec.name,
  )

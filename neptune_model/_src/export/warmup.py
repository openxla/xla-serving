# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under/portions of the License.

"""Warmup utilities for Neptune model export."""

from collections.abc import Mapping, Sequence
import io
import os
from typing import Any

from absl import logging
from google.protobuf import proto
import numpy as np

from neptune_model._src.core import file_utils
from neptune_model.protos.grpc import host_buffer_pb2
from neptune_model.protos.grpc import predict_pb2 as neptune_predict_pb2
from neptune_model.protos.grpc import prediction_log_pb2 as neptune_prediction_log_pb2
from neptune_model.protos.xla.python.ifrt import dtype_pb2


ASSETS_EXTRA_DIR = "assets.extra"
WARMUP_REQUESTS_FILENAME = "neptune_serving_warmup_requests"


def _to_host_buffer_proto(val: Any) -> host_buffer_pb2.HostBufferProto:
  """Converts an input value to HostBufferProto.

  The input value is converted to a NumPy array (either by calling its `.numpy()`
  method if it exists, or by passing it to `np.array()`). The resulting array is
  then converted to HostBufferProto.

  Args:
    val: The input value to convert. Can be a tensor, NumPy array, or any object
      convertible to a NumPy array (e.g., nested lists, scalar values).

  Returns:
    The HostBufferProto representation of the input.

  Raises:
    ValueError: If the converted NumPy array has an unsupported dtype.
  """
  if hasattr(val, "numpy"):
    val = val.numpy()
  elif not isinstance(val, np.ndarray):
    val = np.array(val)

  kind_map = {
      np.dtype(np.bool_): dtype_pb2.DTypeProto.KIND_PRED,
      np.dtype(np.int8): dtype_pb2.DTypeProto.KIND_S8,
      np.dtype(np.int16): dtype_pb2.DTypeProto.KIND_S16,
      np.dtype(np.int32): dtype_pb2.DTypeProto.KIND_S32,
      np.dtype(np.int64): dtype_pb2.DTypeProto.KIND_S64,
      np.dtype(np.uint8): dtype_pb2.DTypeProto.KIND_U8,
      np.dtype(np.uint16): dtype_pb2.DTypeProto.KIND_U16,
      np.dtype(np.uint32): dtype_pb2.DTypeProto.KIND_U32,
      np.dtype(np.uint64): dtype_pb2.DTypeProto.KIND_U64,
      np.dtype(np.float16): dtype_pb2.DTypeProto.KIND_F16,
      np.dtype(np.float32): dtype_pb2.DTypeProto.KIND_F32,
      np.dtype(np.float64): dtype_pb2.DTypeProto.KIND_F64,
      np.dtype(np.complex64): dtype_pb2.DTypeProto.KIND_C64,
      np.dtype(np.complex128): dtype_pb2.DTypeProto.KIND_C128,
  }
  if hasattr(val.dtype, "name") and val.dtype.name == "bfloat16":
    kind_map[np.dtype(val.dtype)] = dtype_pb2.DTypeProto.KIND_BF16

  host_buffer = host_buffer_pb2.HostBufferProto()
  host_buffer.shape.dims.extend(list(val.shape))

  if (
      val.dtype.type == np.str_
      or val.dtype.type == np.bytes_
      or val.dtype == object
  ):
    host_buffer.dtype.kind = dtype_pb2.DTypeProto.KIND_STRING
    elements = [
        s if isinstance(s, bytes) else str(s).encode("utf-8") for s in val.flat
    ]
    host_buffer.string_data.elements.extend(elements)
  else:
    dt = np.dtype(val.dtype)
    if dt in kind_map:
      host_buffer.dtype.kind = kind_map[dt]
    else:
      raise ValueError(f"Unsupported dtype: {val.dtype}")
    host_buffer.raw_data = val.tobytes()

  return host_buffer


def _get_warmup_request_path(model_dir: str) -> str:
  """Returns the warmup requests file path and creates parent directories."""
  output_dir = os.path.join(model_dir, ASSETS_EXTRA_DIR)
  file_utils.mkdir_p(output_dir)
  return os.path.join(output_dir, WARMUP_REQUESTS_FILENAME)


def _to_predict_request(
    signature_name: str, inputs: Mapping[str, Any]
) -> neptune_predict_pb2.PredictRequest:
  """Converts inputs to a PredictRequest proto."""
  request = neptune_predict_pb2.PredictRequest()
  request.model_spec.signature_name = signature_name
  for key, value in inputs.items():
    request.inputs[key].CopyFrom(_to_host_buffer_proto(value))
  return request


def write_warmup_file(
    examples_map: Mapping[str, Sequence[Any]],
    model_dir: str,
) -> None:
  """Writes warm-up examples to the given model directory for Neptune gRPC serving.

  Args:
    examples_map: the mapping of serving signatures to a list of its warmup
      examples.
    model_dir: The SavedModel directory.
  """
  warmup_path = _get_warmup_request_path(model_dir)

  bytes_io = io.BytesIO()
  for signature, examples in examples_map.items():
    if not examples:
      logging.warning(
          "Warm-up examples for signature '%s' are missing.", signature
      )
      continue
    for inputs in examples:
      request = _to_predict_request(signature, inputs)
      log = neptune_prediction_log_pb2.PredictionLog(
          predict_log=neptune_prediction_log_pb2.PredictLog(request=request)
      )
      proto.serialize_length_prefixed(log, bytes_io)

  with file_utils.open_file(warmup_path, "wb") as f:
    f.write(bytes_io.getvalue())

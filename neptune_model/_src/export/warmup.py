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

from collections.abc import Mapping
import os
import struct
from typing import Any

from absl import logging
from etils import epath
import numpy as np

from neptune_model.protos.grpc import host_buffer_pb2
from neptune_model.protos.grpc import predict_pb2 as neptune_predict_pb2
from neptune_model.protos.grpc import prediction_log_pb2 as neptune_prediction_log_pb2
from neptune_model.protos.xla.python.ifrt import dtype_pb2


def _write_size_prefixed_record(file_handle, value: bytes) -> None:
  """Writes a size-prefixed protobuf binary record.

  Args:
    file_handle: The open file object to write the binary record into.
    value: The serialized protobuf binary payload to write.
  """
  # Write the size of the payload as a 4-byte unsigned integer (little-endian).
  encoded_length = struct.pack(b"<I", len(value))
  file_handle.write(encoded_length)
  file_handle.write(value)


def _to_host_buffer_proto(val: Any) -> host_buffer_pb2.HostBufferProto:
  """Converts a tensor or numpy array to HostBufferProto."""
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


def get_warmup_request_path(model_dir: str) -> str:
  """Returns the warmup requests file path and creates parent directories."""
  output_dir = os.path.join(
      model_dir,
      "assets.extra",
  )
  epath.Path(output_dir).mkdir(parents=True, exist_ok=True)
  return os.path.join(output_dir, "tf_serving_warmup_requests")


def write_warmup_file(
    examples_map: Mapping[str, list[Any]],
    model_dir: str,
) -> None:
  """Writes warm-up examples to the given model directory for Neptune gRPC serving.

  Args:
    examples_map: the mapping of serving signatures to a list of its warmup
      examples.
    model_dir: The SavedModel directory.
  """
  warmup_path = get_warmup_request_path(model_dir)

  with epath.Path(warmup_path).open("wb") as f:

    def _to_predict_request(
        signature_name: str, inputs: Mapping[str, Any]
    ) -> neptune_predict_pb2.PredictRequest:
      request = neptune_predict_pb2.PredictRequest()
      request.model_spec.signature_name = signature_name
      for key, value in inputs.items():
        request.inputs[key].CopyFrom(_to_host_buffer_proto(value))
      return request

    for signature, examples in examples_map.items():
      if not examples:
        logging.warning(
            "Warm-up examples for signature '%s' are missing.", signature
        )
        continue
      if not isinstance(examples, list):
        examples = [examples]
      for inputs in examples:
        request = _to_predict_request(signature, inputs)
        log = neptune_prediction_log_pb2.PredictionLog(
            predict_log=neptune_prediction_log_pb2.PredictLog(request=request)
        )
        _write_size_prefixed_record(f, log.SerializeToString())

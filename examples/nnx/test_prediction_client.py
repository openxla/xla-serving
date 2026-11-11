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

"""End-to-end prediction client test for XLA Serving."""

from collections.abc import Sequence

from absl import app
from absl import flags
import grpc
import numpy as np

from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2


_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", ""})


def _is_loopback_target(target: str) -> bool:
  return (
      target in _LOOPBACK_HOSTS or target.rsplit(":", 1)[0] in _LOOPBACK_HOSTS
  )


def _make_channel(
    server_address: str, use_alts_credentials: bool = False
) -> grpc.Channel:
  if use_alts_credentials:
    return grpc.secure_channel(server_address, grpc.alts_channel_credentials())
  if _is_loopback_target(server_address):
    return grpc.secure_channel(server_address, grpc.local_channel_credentials())
  return grpc.insecure_channel(server_address)


def make_host_buffer(arr: np.ndarray) -> host_buffer_pb2.HostBufferProto:
  hb = host_buffer_pb2.HostBufferProto()
  hb.shape.dims.extend(arr.shape)
  hb.dtype.kind = dtype_pb2.DTypeProto.KIND_S32
  hb.raw_data = arr.tobytes()
  return hb


def run_prediction_test(
    server_address: str = "localhost:8500",
    model_name: str = "nnx",
    signature_name: str = "serving_default",
    batch_size: int = 4,
    timeout: int = 60,
    use_alts_credentials: bool = False,
) -> None:
  """Sends a Predict request to the server and verifies the response."""
  print(f"Connecting to XLA Serving server at {server_address}...")
  channel = _make_channel(
      server_address, use_alts_credentials=use_alts_credentials
  )
  predict_fn = channel.unary_unary(
      f"/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict",
      request_serializer=predict_pb2.PredictRequest.SerializeToString,
      response_deserializer=predict_pb2.PredictResponse.FromString,
  )

  req = predict_pb2.PredictRequest()
  req.model_spec.name = model_name
  req.model_spec.signature_name = signature_name

  for i in range(1, 5):
    arr = np.zeros((batch_size, 1), dtype=np.int32)
    req.inputs[f"feature_{i}"].CopyFrom(make_host_buffer(arr))

  print(
      f"Sending PredictRequest for model '{model_name}' (batch_size={batch_size})..."
  )
  resp = predict_fn(req, timeout=timeout)
  print(
      f"Received response: model='{resp.model_spec.name}',"
      f" signature='{resp.model_spec.signature_name}'"
  )

  if "pred" not in resp.outputs:
    raise AssertionError(
        f"Missing expected output 'pred'. Got outputs: {list(resp.outputs.keys())}"
    )

  output_hb = resp.outputs["pred"]
  out_shape = list(output_hb.shape.dims)
  expected_shape = [batch_size, 1, 1]
  if out_shape != expected_shape:
    raise AssertionError(
        f"Expected output shape {expected_shape}, but got {out_shape}"
    )

  out_data = np.frombuffer(output_hb.raw_data, dtype=np.float32).reshape(
      out_shape
  )
  print(f"Prediction output shape: {out_shape}")
  print(f"Prediction output values:\n{out_data}")
  print("✅ End-to-end inference verification successful!")


_SERVER_ADDRESS = flags.DEFINE_string(
    "server_address",
    "localhost:8500",
    "Server address (default: localhost:8500)",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name",
    "nnx",
    "Model name (default: nnx)",
)
_SIGNATURE_NAME = flags.DEFINE_string(
    "signature_name",
    "serving_default",
    "Signature name (default: serving_default)",
)
_BATCH_SIZE = flags.DEFINE_integer(
    "batch_size",
    4,
    "Batch size for prediction test (default: 4)",
)
_TIMEOUT = flags.DEFINE_integer(
    "timeout",
    60,
    "Timeout in seconds (default: 60)",
)
_USE_ALTS_CREDENTIALS = flags.DEFINE_bool(
    "use_alts_credentials",
    False,
    "Use ALTS credentials for Google Cloud endpoints.",
)


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  run_prediction_test(
      server_address=_SERVER_ADDRESS.value,
      model_name=_MODEL_NAME.value,
      signature_name=_SIGNATURE_NAME.value,
      batch_size=_BATCH_SIZE.value,
      timeout=_TIMEOUT.value,
      use_alts_credentials=_USE_ALTS_CREDENTIALS.value,
  )


if __name__ == "__main__":
  app.run(main)

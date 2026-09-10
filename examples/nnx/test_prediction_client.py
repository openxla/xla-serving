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

"""End-to-end prediction client test for Neptune Serving."""

import argparse
import grpc
import numpy as np

from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2


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
) -> None:
  """Sends a Predict request to the server and verifies the response."""
  print(f"Connecting to Neptune server at {server_address}...")
  channel = grpc.insecure_channel(server_address)
  predict_fn = channel.unary_unary(
      "/neptune.serving.PredictionService/Predict",
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


def main() -> None:
  parser = argparse.ArgumentParser(
      description="Test Neptune serving prediction client."
  )
  parser.add_argument(
      "--server_address",
      default="localhost:8500",
      help="Server address (default: localhost:8500)",
  )
  parser.add_argument(
      "--model_name",
      default="nnx",
      help="Model name (default: nnx)",
  )
  parser.add_argument(
      "--signature_name",
      default="serving_default",
      help="Signature name (default: serving_default)",
  )
  parser.add_argument(
      "--batch_size",
      type=int,
      default=4,
      help="Batch size for prediction test (default: 4)",
  )
  parser.add_argument(
      "--timeout",
      type=int,
      default=60,
      help="Timeout in seconds (default: 60)",
  )
  args = parser.parse_args()

  run_prediction_test(
      server_address=args.server_address,
      model_name=args.model_name,
      signature_name=args.signature_name,
      batch_size=args.batch_size,
      timeout=args.timeout,
  )


if __name__ == "__main__":
  main()

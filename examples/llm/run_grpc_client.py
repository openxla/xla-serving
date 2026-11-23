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

"""Standalone Python client to query a running XLA Serving LLM server."""

from collections.abc import Sequence

from absl import app
from absl import flags
import grpc
import numpy as np

from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2  # pylint: disable=g-direct-tensorflow-import

_SERVER_ADDRESS = flags.DEFINE_string(
    "server_address",
    "localhost:8500",
    "gRPC address of the XLA Serving server.",
)
_MODEL_NAME = flags.DEFINE_string(
    "model_name", "qwen3", "Model name registered on the server."
)
_SIGNATURE_NAME = flags.DEFINE_string(
    "signature_name", "serving_default", "Serving signature name."
)
_PROMPT = flags.DEFINE_string(
    "prompt",
    "The future of artificial intelligence in scientific discovery is",
    "Prompt text to generate completions for.",
)
_USE_ALTS_CREDENTIALS = flags.DEFINE_bool(
    "use_alts_credentials",
    False,
    "Use ALTS credentials for Google Cloud endpoints.",
)

_LOOPBACK_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]", ""})


def _make_channel(
    server_address: str, use_alts_credentials: bool = False
) -> grpc.Channel:
  if use_alts_credentials:
    return grpc.secure_channel(server_address, grpc.alts_channel_credentials())
  if (
      server_address in _LOOPBACK_HOSTS
      or server_address.rsplit(":", 1)[0] in _LOOPBACK_HOSTS
  ):
    return grpc.secure_channel(server_address, grpc.local_channel_credentials())
  return grpc.insecure_channel(server_address)


def query_server(
    server_address: str,
    model_name: str,
    signature_name: str,
    prompt: str,
    use_alts_credentials: bool = False,
) -> list[str]:
  """Sends a single prompt to the server and returns all generated beams."""
  channel = _make_channel(
      server_address, use_alts_credentials=use_alts_credentials
  )
  predict_fn = channel.unary_unary(
      f"/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict",
      request_serializer=predict_pb2.PredictRequest.SerializeToString,
      response_deserializer=predict_pb2.PredictResponse.FromString,
  )

  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name

  hb_proto = host_buffer_pb2.HostBufferProto()
  hb_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
  hb_proto.shape.dims.append(1)
  hb_proto.string_data.elements.append(prompt.encode("utf-8"))
  request.inputs["text_input"].CopyFrom(hb_proto)

  print(f"Sending prompt to {server_address} (model={model_name})...")
  response = predict_fn(request)

  output_buffer = response.outputs["text_output"]
  dims = list(output_buffer.shape.dims)
  elements = [
      raw_bytes.decode("utf-8", errors="replace")
      for raw_bytes in output_buffer.string_data.elements
  ]
  output_array = np.array(elements, dtype=object).reshape(dims)
  if len(dims) > 1:
    return output_array[0].tolist()
  return output_array.tolist()


def main(argv: Sequence[str]) -> None:
  if len(argv) > 1:
    raise app.UsageError("Too many command-line arguments.")

  beams = query_server(
      server_address=_SERVER_ADDRESS.value,
      model_name=_MODEL_NAME.value,
      signature_name=_SIGNATURE_NAME.value,
      prompt=_PROMPT.value,
      use_alts_credentials=_USE_ALTS_CREDENTIALS.value,
  )

  print("\n" + "=" * 50)
  print(f"Prompt: {_PROMPT.value}")
  print("=" * 50)
  if isinstance(beams, list):
    for i, beam in enumerate(beams):
      print(f"  [Beam {i:2d}]: {beam}")
  else:
    print(f"  Output: {beams}")
  print("=" * 50)


if __name__ == "__main__":
  app.run(main)

# Serving and runtime

An exported XLA SavedModel bundle can be run in three ways, all powered by the
same C++ `Interpreter`:

1. **Standalone gRPC server** — prebuilt container images (`us-docker.pkg.dev/xla-serving-engine/xla-serving/*`)
   or the `xla_serving_main_{cpu,gpu,tpu}` binary.
2. **In-process from Python (`ModelRunner`)** — nanobind wrapper around the C++
   `Interpreter` for local testing and notebooks.
3. **Embedded in C++ (`xla_serving::Interpreter`)** — link the runtime directly
   into your own C++ service.

## Running the gRPC server

### Single model

Pass `--model_name`, `--model_path`, and `--orbax_grpc_server_platform` (`cpu`,
`cuda`, or `tpu`):

::::{tab-set}
:::{tab-item} CPU
```bash
docker run \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=cpu \
  --port=8500
```
:::

:::{tab-item} NVIDIA GPU
Pass `--gpus all` (requires the NVIDIA Container Toolkit) and select the
`cuda12` or `cuda13` image matching your driver:

```bash
docker run \
  --gpus all \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=cuda \
  --port=8500
```

To bind a specific GPU or MIG partition, use `--gpus '"device=0"'` or
`--gpus '"device=<mig-uuid>"'`.
:::

:::{tab-item} Cloud TPU
On a Cloud TPU VM, run with `--privileged` so the container can access the TPU
device nodes. The `tpu-jax-only` image bundles `libtpu.so` at
`/usr/lib/libtpu.so`:

```bash
docker run \
  --privileged \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=tpu \
  --port=8500
```

If you run a bare `xla_serving_main_tpu` binary outside the container, set
`TPU_LIBRARY_PATH=/path/to/libtpu.so` in the environment.
:::
::::

If your bundle contains a `TensorflowModel` stage, use the corresponding
`-with-tf` image (`cpu-jax-with-tf`, `gpu-jax-cuda12-with-tf`, or
`gpu-jax-cuda13-with-tf`).

When all models finish loading (and optional [warmup](warmup.md) completes),
the server begins listening on `localhost:8500` by default (or `[::]:8500` when
remote credentials are configured).

### Serving multiple models in one server

Instead of `--model_name` and `--model_path`, pass `--model_config_file`
pointing to a `ModelServerConfig` textproto file:

```text
# models.txtpb
model_config_list {
  configs {
    name: "classifier"
    base_path: "/models/classifier"
  }
  configs {
    name: "ranker"
    base_path: "/models/ranker"
  }
}
```

```bash
docker run \
  --net=host \
  --mount type=bind,src=/tmp/models,dst=/models,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest \
  --model_config_file=/models/models.txtpb \
  --orbax_grpc_server_platform=cpu \
  --port=8500
```

All models in the config share the server's IFRT device client and execution
thread pools. Clients select which model to call via `model_spec.name` in
`PredictRequest`.

### Transport security

XSE supports three credential modes depending on where your clients run:

1. **Loopback default (no flags)** — When no credential flags are passed, the
   server binds to `localhost:<port>` with gRPC `LocalServerCredentials` and
   accepts connections only from local loopback addresses (`127.0.0.1` and
   `::1`). Run Docker with `--net=host` so host clients share the loopback
   interface; Docker bridge port forwarding (`-p 8500:8500`) rewrites peer
   addresses to the bridge gateway and will be rejected. Connect with
   `grpc.secure_channel('localhost:8500', grpc.local_channel_credentials())` in
   Python or `grpc::experimental::LocalCredentials(LOCAL_TCP)` in C++.
2. **Google Cloud ALTS (`--use_alts_credentials=true`)** — On Compute Engine,
   GKE, and Cloud TPU VMs, ALTS provides automatic mutual TLS and service-account
   authentication backed by the GCP metadata server, with no certificate files
   to manage:

   ```bash
     --use_alts_credentials=true
   ```

   The server listens on `[::]:<port>`. Clients on GCP connect with
   `grpc.secure_channel('<host>:8500', grpc.alts_channel_credentials())` in
   Python or `grpc::experimental::AltsCredentials(...)` (`--use_alts_credentials=true`
   on the bundled C++ clients).
3. **Custom TLS and mTLS** — Outside Google Cloud or across trust boundaries,
   supply PEM-encoded key and certificate files:

   ```bash
     --server_private_key_path=/etc/tls/server.key \
     --server_cert_chain_path=/etc/tls/server.crt \
     --client_cert_chain_path=/etc/tls/ca.crt    # optional: requires client certs (mTLS)
   ```

   Clients connect with `grpc.secure_channel('<host>:8500', grpc.ssl_channel_credentials(root_certificates=...))`.

### Server flags reference

| Flag | Default | Description |
|---|---|---|
| `--port` | `8500` | Port for the gRPC `PredictionService`. Binds `localhost:<port>` with default local credentials, or `[::]:<port>` when remote credentials are enabled. |
| `--model_name` | `""` | Name of the single model to serve (paired with `--model_path`). |
| `--model_path` | `""` | Path to the exported bundle directory. |
| `--model_config_file` | `""` | Path to a `ModelServerConfig` textproto for serving one or more models. Mutually exclusive with `--model_name`/`--model_path`. |
| `--orbax_grpc_server_platform` | `""` | Target hardware platform: `cpu`, `cuda`, or `tpu`. |
| `--warmup_server` | `false` | Replay `assets.extra/xla_serving_warmup_requests` before opening the gRPC port. Fails startup if the file is missing. |
| `--use_alts_credentials` | `false` | Use Google Cloud ALTS mutual TLS via the metadata server (Compute Engine, GKE, Cloud TPU) with no certificate files. |
| `--server_private_key_path` | `""` | Path to PEM-encoded server private key for TLS (requires `--server_cert_chain_path`). |
| `--server_cert_chain_path` | `""` | Path to PEM-encoded server certificate chain for TLS (requires `--server_private_key_path`). |
| `--client_cert_chain_path` | `""` | Optional PEM-encoded client CA bundle to require mutual TLS (mTLS). |
| `--use_insecure_grpc_credentials` | `false` | Accept unencrypted, unauthenticated gRPC connections on `[::]:<port>` (isolated testing only, such as Docker bridge networking). |
| `--max_receive_message_size` | `16777216` (16 MiB) | Maximum incoming gRPC message size in bytes. Increase for large image or batch payloads. |
| `--max_send_message_size` | `-1` (unlimited) | Maximum outgoing gRPC message size in bytes. |
| `--thread_pool_num_threads` | `0` (CPU core count) | Size of the server's execution and computation thread pools. |
| `--xprof_port` | `8888` | Port for the embedded XProf profiler service. |
| `--use_nanort_cpu_client` | `true` | CPU only. Use the lightweight NanoRT IFRT client instead of the PJRT CPU client. |
| `--tpu_use_premapped_buffer` | `false` | TPU only. Enable premapped host-to-device buffers in the PJRT TPU client. |
| `--tpu_premapped_buffer_size` | `2147483648` (2 GiB) | TPU only. Size in bytes of the premapped buffer when `--tpu_use_premapped_buffer=true`. |

## Querying the gRPC API

The server implements `xla_serving.serving.PredictionService` (defined in
`protos/grpc/prediction_service.proto`), which exposes two RPCs:

- **`Predict(PredictRequest) -> PredictResponse`** — executes a serving
  signature.
- **`GetModelMetadata(GetModelMetadataRequest) -> GetModelMetadataResponse`** —
  returns input and output signature metadata for a loaded model.

The compiled Python protobuf modules ship inside `xla-saved-model` under
`xla_saved_model.protos.grpc`.

### Calling `Predict`

Each input and output tensor is serialized as a `HostBufferProto` containing a
shape, an IFRT `DTypeProto.Kind`, and either `raw_data` (little-endian bytes
for numeric/boolean tensors) or `string_data` (repeated `bytes` for string
tensors).

```python
import grpc
import numpy as np
from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2

_NP_TO_IFRT_KIND = {
    np.dtype('bool'): dtype_pb2.DTypeProto.KIND_PRED,
    np.dtype('int8'): dtype_pb2.DTypeProto.KIND_S8,
    np.dtype('int16'): dtype_pb2.DTypeProto.KIND_S16,
    np.dtype('int32'): dtype_pb2.DTypeProto.KIND_S32,
    np.dtype('int64'): dtype_pb2.DTypeProto.KIND_S64,
    np.dtype('uint8'): dtype_pb2.DTypeProto.KIND_U8,
    np.dtype('uint16'): dtype_pb2.DTypeProto.KIND_U16,
    np.dtype('uint32'): dtype_pb2.DTypeProto.KIND_U32,
    np.dtype('uint64'): dtype_pb2.DTypeProto.KIND_U64,
    np.dtype('float16'): dtype_pb2.DTypeProto.KIND_F16,
    np.dtype('float32'): dtype_pb2.DTypeProto.KIND_F32,
    np.dtype('float64'): dtype_pb2.DTypeProto.KIND_F64,
}
_IFRT_KIND_TO_NP = {v: k for k, v in _NP_TO_IFRT_KIND.items()}


def to_host_buffer(array: np.ndarray) -> host_buffer_pb2.HostBufferProto:
  array = np.ascontiguousarray(array)
  buf = host_buffer_pb2.HostBufferProto()
  buf.shape.dims.extend(array.shape)
  if array.dtype.kind in ('U', 'S', 'O'):
    buf.dtype.kind = dtype_pb2.DTypeProto.KIND_STRING
    for item in array.flat:
      buf.string_data.append(
          item.encode('utf-8') if isinstance(item, str) else bytes(item)
      )
  else:
    buf.dtype.kind = _NP_TO_IFRT_KIND[array.dtype]
    buf.raw_data = array.tobytes()
  return buf


def from_host_buffer(buf: host_buffer_pb2.HostBufferProto) -> np.ndarray:
  if buf.dtype.kind == dtype_pb2.DTypeProto.KIND_STRING:
    return np.array(list(buf.string_data), dtype=object).reshape(buf.shape.dims)
  dtype = _IFRT_KIND_TO_NP[buf.dtype.kind]
  return np.frombuffer(buf.raw_data, dtype=dtype).reshape(buf.shape.dims)
```

Send a `PredictRequest`:

```python
channel = grpc.secure_channel(
    'localhost:8500', grpc.local_channel_credentials()
)
predict = channel.unary_unary(
    f'/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict',
    request_serializer=predict_pb2.PredictRequest.SerializeToString,
    response_deserializer=predict_pb2.PredictResponse.FromString,
)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'classifier'
request.model_spec.signature_name = 'serving_default'
request.inputs['features'].CopyFrom(
    to_host_buffer(np.zeros((4, 16), dtype=np.float32))
)

response = predict(request, timeout=10.0)
scores = from_host_buffer(response.outputs['score'])
```

If the server has only one model loaded, `model_spec.name` can be left empty;
`model_spec.signature_name` defaults to `'serving_default'`.

### Calling `GetModelMetadata`

Query `signature_def` to inspect a loaded model's inputs and outputs over the
network:

```python
from xla_saved_model.protos.grpc import get_model_metadata_pb2

get_metadata = channel.unary_unary(
    f'/{get_model_metadata_pb2.DESCRIPTOR.package}.PredictionService/GetModelMetadata',
    request_serializer=get_model_metadata_pb2.GetModelMetadataRequest.SerializeToString,
    response_deserializer=get_model_metadata_pb2.GetModelMetadataResponse.FromString,
)

meta_req = get_model_metadata_pb2.GetModelMetadataRequest()
meta_req.model_spec.name = 'classifier'
meta_req.model_spec.signature_name = 'serving_default'
meta_req.metadata_field.append('signature_def')

meta_resp = get_metadata(meta_req, timeout=5.0)
sig_map = get_model_metadata_pb2.SignatureDefMap()
meta_resp.metadata['signature_def'].Unpack(sig_map)
print(sig_map.signature_def['serving_default'])
```

## Running in-process from Python (`ModelRunner`)

The `xla-saved-model-runner` package binds the C++ `Interpreter` directly into
Python. In the open-source build it runs on CPU, making it ideal for unit tests
and export validation without spinning up a container:

```python
import numpy as np
from xla_saved_model_runner import model_runner

runner = model_runner.ModelRunner(
    model_path='/tmp/classifier',
    platform='cpu',
)

print(runner.signature_names())  # ['serving_default']

sig = runner.signature('serving_default')
outputs = sig.run_with_dict_return(
    features=np.zeros((2, 16), dtype=np.float32),
)
print(outputs['score'])
```

When testing [ahead-of-time compiled bundles](aot.md), pass
`target_topology_label='...'` and `allow_jit_fallback_for_aot=False` to
`ModelRunner`.

## Embedding the C++ `Interpreter`

If you have an existing C++ binary, you can embed `xla_serving::Interpreter`
directly instead of running the standalone gRPC server:

```cpp
#include "runtime/core/host_buffer.h"
#include "runtime/core/value.h"
#include "runtime/interpreter/interpreter.h"
#include "runtime/interpreter/interpreter_create_options.h"
#include "runtime/interpreter/run_options.h"

// 1. Configure and create the Interpreter once at startup.
auto options = std::make_unique<xla_serving::InterpreterCreateOptions>();
options->model_path = "/models/classifier";
options->model_name = "classifier";
options->platform_name = "cpu";  // "cpu", "cuda", or "tpu"

TF_ASSIGN_OR_RETURN(std::unique_ptr<xla_serving::Interpreter> interpreter,
                    xla_serving::Interpreter::Create(std::move(options)));

// 2. Run a signature synchronously or asynchronously per request.
absl::flat_hash_map<std::string, xla_serving::Value> inputs;
inputs["features"] = xla_serving::Value(std::move(features_host_buffer));

xla_serving::RunOptions run_options;
run_options.deadline = absl::Now() + absl::Seconds(5);

TF_ASSIGN_OR_RETURN(
    auto outputs,
    interpreter->RunWithMapReturn("serving_default",
                                  /*positional_inputs=*/{}, inputs,
                                  run_options));
const xla_serving::HostBuffer& score =
    outputs.at("score").Get<xla_serving::HostBuffer>();
```

For non-blocking execution, call `interpreter->RunAsyncWithMapReturn(...)`,
which returns a `tsl::Future`. If your bundle includes TensorFlow stages,
instantiate `xla_serving::TfCompatibleInterpreterCreateOptions` (from
`runtime/tensorflow/tf_compatible_interpreter_create_options.h`) in place of
`InterpreterCreateOptions`.

## Profiling with XProf

The server starts an [XProf](https://github.com/openxla/xprof) profiler service
on `--xprof_port` (default `8888`). You can capture a host and device trace
from a running server under load to inspect batch formation, host preprocessing
time, and XLA kernel execution on the timeline.

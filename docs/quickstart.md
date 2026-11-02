# Quickstart

In about ten minutes you will export a small Flax model, inspect the bundle,
run it in-process, serve it from a container, and query it over gRPC. Every
command below was run as written; the model is trivial on purpose so that
nothing here depends on an accelerator.

You need Python 3.12+ and Docker (see [Installation](installation.md) for
details). Install the export library, in-process runner, and the example's
dependencies (`flax` and `grpcio`):

```bash
pip install xla-saved-model xla-saved-model-runner flax grpcio
```

## 1. Export

Save this as `export.py` and run it.

```python
import jax
import jax.numpy as jnp
from flax import nnx
import xla_saved_model


class Classifier(nnx.Module):

  def __init__(self, rngs: nnx.Rngs):
    self.dense1 = nnx.Linear(16, 32, rngs=rngs)
    self.dense2 = nnx.Linear(32, 1, rngs=rngs)

  def __call__(self, x):
    return nnx.sigmoid(self.dense2(nnx.relu(self.dense1(x))))


model = Classifier(nnx.Rngs(0))
graphdef, params = nnx.split(model)


def apply_fn(params, inputs):
  return {'score': nnx.merge(graphdef, params)(inputs['features'])}


xla_saved_model.save(
    xla_saved_model.JaxModel(
        apply_fn_bundle=xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,
            input_polymorphic_shapes={'features': 'b, _'},
            jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
        ),
        params=params,
    ),
    export_path='/tmp/classifier',
    serving_configs=[
        xla_saved_model.ServingConfig(
            serving_signature='serving_default',
            input_signature={
                'features': jax.ShapeDtypeStruct(('b', 16), jnp.float32),
            },
        )
    ],
)
```

Three things in there are doing the work:

- `apply_fn(params, inputs)` is the calling convention XSE expects. It takes
  the parameter PyTree and a dict of named inputs, and returns a dict of named
  outputs. The names become the wire names in gRPC requests.
- `input_polymorphic_shapes={'features': 'b, _'}` and
  `ShapeDtypeStruct(('b', 16), ...)` together say "the batch dimension is
  symbolic". One StableHLO module serves any batch size.
- `platforms=['cpu', 'cuda', 'tpu']` lowers the function three times. Drop
  what you do not need.

The result is a directory:

```text
/tmp/classifier
├── checkpoint/
├── history.txtpb
├── manifest.pb
├── orbax_model_version.txt
├── orchestration.pb
└── serving_default_jax_specific_info_supplemental.pb
```

## 2. Look inside

`xsm-cli` ships with the export wheel.

```bash
$ xsm-cli show /tmp/classifier
Manifest: 7 KiB, 2 objects, 1 supplementals
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Object          ┃ Type                ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ serving_default │ MLIR StableHLO v1.0 │
│                 │ inlined: 3861 bytes │
│ weights         │ Orbax Checkpoint    │
│                 │ file: checkpoint    │
└─────────────────┴─────────────────────┘
```

Add `--details serving_default` to see the input and output signature, the
lowering platforms, and the weight names. This is the first thing to check
when a server rejects a request: the names and dtypes it expects are printed
right there.

## 3. Run it in-process

`xla_saved_model_runner` loads a bundle through the same C++ `Interpreter`
the server uses. It is the quickest way to confirm an export is sound.

```python
import numpy as np
from xla_saved_model_runner import model_runner

runner = model_runner.ModelRunner(model_path='/tmp/classifier')
print(runner.signature_names())           # ['serving_default']

signature = runner.signature('serving_default')
outputs = signature.run_with_dict_return(
    features=np.random.rand(3, 16).astype(np.float32))
print(outputs['score'].shape)             # (3, 1)
```

Inputs are passed as keyword arguments named after the input signature;
outputs come back as a dict of NumPy arrays. The open-source runner executes
on CPU regardless of how the function was lowered, which is fine for
validation.

## 4. Serve it

Mount the bundle into a server image and point the server at it:

```bash
docker run \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=cpu \
  --port=8500 \
  --use_insecure_grpc_credentials=true
```

The server compiles the StableHLO for the host CPU, loads the checkpoint, and
begins listening on `[::]:8500`.

`--model_name` is the name clients put in `model_spec.name`. It does not have
to match the directory. `--use_insecure_grpc_credentials=true` is for local
use; the TLS flags are `--server_private_key_path`,
`--server_cert_chain_path`, and `--client_cert_chain_path`.

::::{tab-set}
:::{tab-item} GPU
Use a `gpu-jax-cuda12` or `gpu-jax-cuda13` image, pass the GPU through, and
set the platform to `cuda`:

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
  --port=8500 \
  --use_insecure_grpc_credentials=true
```

To run on one MIG partition instead of the whole card, replace `--gpus all`
with `--gpus '"device=<mig-id>"'`.
:::
:::{tab-item} TPU
On a Cloud TPU VM, use the `tpu-jax-only` image (which bundles `libtpu.so`
inside the container) with `--privileged` so the container can access the TPU
device nodes, and set the platform to `tpu`:

```bash
docker run \
  --privileged \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=tpu \
  --port=8500 \
  --use_insecure_grpc_credentials=true
```
:::
:::{tab-item} Without Docker
A locally built binary takes the same flags:

```bash
bazel run \
  -c opt \
  //runtime/serving/gcp/server:xla_serving_main_cpu \
  -- \
  --model_name=classifier \
  --model_path=/tmp/classifier \
  --orbax_grpc_server_platform=cpu \
  --port=8500 \
  --use_insecure_grpc_credentials=true
```
:::
::::

## 5. Query it

The server speaks `PredictionService/Predict`. A request names the model and
signature and carries one `HostBufferProto` per input: a dtype, a shape, and
the raw bytes. The generated protobuf modules are part of the
`xla_saved_model` wheel, so a client needs nothing beyond `grpcio`.

```python
import grpc
import numpy as np
from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2


def to_host_buffer(array: np.ndarray) -> host_buffer_pb2.HostBufferProto:
  buf = host_buffer_pb2.HostBufferProto()
  buf.shape.dims.extend(array.shape)
  buf.dtype.kind = dtype_pb2.DTypeProto.KIND_F32
  buf.raw_data = array.tobytes()
  return buf


def from_host_buffer(buf: host_buffer_pb2.HostBufferProto) -> np.ndarray:
  return np.frombuffer(buf.raw_data, dtype=np.float32).reshape(buf.shape.dims)


channel = grpc.insecure_channel('localhost:8500')
predict = channel.unary_unary(
    f'/{predict_pb2.DESCRIPTOR.package}.PredictionService/Predict',
    request_serializer=predict_pb2.PredictRequest.SerializeToString,
    response_deserializer=predict_pb2.PredictResponse.FromString,
)

request = predict_pb2.PredictRequest()
request.model_spec.name = 'classifier'
request.model_spec.signature_name = 'serving_default'
request.inputs['features'].CopyFrom(
    to_host_buffer(np.random.rand(4, 16).astype(np.float32)))

response = predict(request, timeout=30)
print(from_host_buffer(response.outputs['score']))
```

```text
[[0.41297507]
 [0.4112966 ]
 [0.37193033]
 [0.52405393]]
```

`DTypeProto.KIND_*` follows IFRT's naming: `KIND_S32`, `KIND_F32`,
`KIND_BF16`, `KIND_STRING`, and so on. String inputs use the `string_data`
field instead of `raw_data`. For a C++ client see `examples/nnx/client.cc`.

## Where this leaves you

You have a bundle that is already multi-platform, a server that will
JIT-compile it wherever it lands, and a client. Next steps:

- Exporting other frameworks (Flax Linen, Keras 3, PyTorch via `torchax`),
  existing Orbax checkpoints, multi-device sharding, and multiple signatures:
  [Exporting models](guides/exporting.md).
- Server flags, multi-model serving, TLS, `GetModelMetadata`, and embedding the
  C++ `Interpreter`: [Serving and runtime](guides/serving.md).
- Batching, so the accelerator sees 32 requests at once instead of one:
  [Batching](guides/batching.md).
- Host-side feature processing in the same pipeline:
  [Pre- and post-processing](guides/processing.md).
- Predictable first-request latency: [Warmup](guides/warmup.md) and
  [Ahead-of-time compilation](guides/aot.md).

For larger end-to-end examples, the repository's `examples/` directory has a
ViT classifier, a DLRM recommender with embeddings, a PyTorch model via
`torchax`, C++ composite preprocessors, and a Qwen3 beam-search exporter.

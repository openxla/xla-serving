# Warmup

The first request to a freshly started server pays for XLA compilation and
for whatever lazy initialization the pipeline does. On CPU that is a second
or two; on TPU with a few batch sizes it can be tens of seconds, during which
the server is already reporting healthy and taking traffic. Warmup moves that
cost to startup: the server replays a set of recorded requests before it
starts listening.

If you want to remove compilation from startup entirely rather than just
front-load it, see [Ahead-of-time compilation](aot.md). The two combine
well; warmup still exercises the host-side stages and device allocations.

## Recording warmup requests at export

Pass `warmup_examples` to `save()`. It is a mapping from signature name to a
list of example inputs, each of which is a dict of input name to array:

```python
import numpy as np
import xla_saved_model

warmup_examples = {
    'serving_default': [
        {'features': np.zeros((1, 16), np.float32)},
        {'features': np.zeros((8, 16), np.float32)},
        {'features': np.zeros((32, 16), np.float32)},
    ],
}

xla_saved_model.save(
    jax_model,
    export_path='/tmp/classifier',
    serving_configs=[serving_config],
    warmup_examples=warmup_examples,
)
```

The exporter converts each example to a `PredictRequest`, wraps it in a
`PredictionLog`, and writes the records length-prefixed to
`assets.extra/xla_serving_warmup_requests` inside the bundle. Values can be
NumPy arrays, anything with a `.numpy()` method, or plain lists; strings are
stored as string buffers.

Choose examples that cover every shape you expect to compile. With
[batching](batching.md) enabled, that means one example per entry in
`allowed_batch_sizes`. Without it, cover the batch sizes your clients
actually send. The contents of the arrays do not matter for compilation,
but they do flow through your preprocessors, so use values that will not
trip an assertion there.

## Replaying at startup

Warmup is off by default. Turn it on with a server flag:

```bash
docker run \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=cpu \
  --warmup_server=true
```

The server runs every recorded request through the loaded model and only
then binds the port and starts accepting requests. If `--warmup_server`
is set and the bundle has no warmup file, startup fails. That is
deliberate: a deployment that expects warmup and silently skips it is
harder to debug than one that refuses to start.

## Writing the file yourself

The file format is the same one TensorFlow Serving uses for
`tf_serving_warmup_requests`, with XSE's `PredictRequest` inside. If you
want to record real traffic rather than synthetic examples, write
`PredictionLog` messages with
`xla_saved_model.protos.grpc.prediction_log_pb2` and
`google.protobuf.proto.serialize_length_prefixed`, and drop the file into
`assets.extra/` in the bundle. The server does not care how it got there.

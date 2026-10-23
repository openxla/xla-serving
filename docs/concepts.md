# Concepts

Five terms come up everywhere: the *bundle*, *models*, *signatures*,
*orchestration*, and *platforms*. This page defines them once so the guides
do not have to.

## The bundle

`xla_saved_model.save()` writes a directory. A minimal one, for a single JAX
model with no host-side processing, looks like this:

```text
classifier/
├── manifest.pb                 # functions, values, supplementals; the index
├── orchestration.pb            # one Pipeline per serving signature
├── checkpoint/                 # Orbax checkpoint with the parameters
├── history.txtpb               # append-only log: export, xsm-cli updates and rollbacks
└── orbax_model_version.txt     # written last; its presence means "complete"
```

The **manifest** is the index. It lists every *function* (StableHLO bytecode,
inlined by default) and every *value* (the weights, as a reference to the
checkpoint), plus *supplementals*: side files such as the orchestration graph,
a nested TF SavedModel, or XLA compile options.

Other files appear when you use the corresponding feature:

| Path | Appears when |
|---|---|
| `tf_saved_model/` | A `TensorflowModel` is part of the export. Standard TF SavedModel with the pre/post-processing functions and their resources. |
| `assets.extra/xla_serving_warmup_requests` | You pass `warmup_examples=` to `save()`. See [Warmup](guides/warmup.md). |
| `<function>_<label>_pjrt_executable.bin` | You run `xsm-cli recompile`. See [Ahead-of-time compilation](guides/aot.md). |
| `<function>.shlo` | You pass `save_shlo_to_file=True`; StableHLO goes to a file instead of inline in the manifest. |
| `<composite>/` | A `CompositeModel` of C++ transforms is exported; holds its config and assets. |

`xsm-cli show <dir>` prints the manifest in a readable form. Reach for it
before reading protos by hand.

The bundle is complete when `orbax_model_version.txt` exists. It is written
last, so a loader that polls a directory can key off that file.

## Models and method keys

A **model** is a named container of callable functions plus whatever state
they need:

- `JaxModel` holds one or more `ApplyFnBundle`s and a PyTree of parameters.
  Each bundle wraps a function with signature `(params, inputs) -> outputs`.
- `TensorflowModel` holds `tf.function`s and their trackable resources
  (lookup tables, embedding tables, tokenizer assets).
- `CompositeModel` holds a sequence of C++ transforms from
  `xla_serving_data`.

Within a model, each function has a **method key**. If you pass a single
`ApplyFnBundle` it is keyed `serving_default`; pass a mapping to name them
yourself. The pair `(model name, method key)` is how orchestration refers to a
function.

Model names default to `jax_model` and `tensorflow_model`. Name them
explicitly whenever you export more than one.

## Serving signatures

A **serving signature** is what a client calls. It has a name
(`serving_default` unless you say otherwise), an input signature, and
optionally batching options and an orchestration. `ServingConfig` is the
Python object; a `Pipeline` message in `orchestration.pb` is the serialized
form.

The input signature is a PyTree of specs. For JAX inputs use
`jax.ShapeDtypeStruct` with a string in the batch dimension:

```python
input_signature = {
    'features': jax.ShapeDtypeStruct(('b', 16), jnp.float32),
}
```

`'b'` is a symbolic dimension. It must agree with
`ApplyFnBundle.input_polymorphic_shapes`, which uses the same letters in
`jax.export` shape-spec syntax (`'b, _'` means "symbolic batch, concrete
second dim"). For TF-fronted pipelines use `tf.TensorSpec`,
`tf.RaggedTensorSpec`, or `tf.SparseTensorSpec` with `None` for the batch
dimension. For pipelines that start with a C++ stage use either
`jax.ShapeDtypeStruct` or `ShloTensorSpec` (see
[Pre- and post-processing](guides/processing.md)).

One bundle can hold several signatures: a `score` signature that batches and
an `embed` signature that does not, say, or the same model with different
preprocessors.

## Orchestration

**Orchestration** is the list of model calls that implements a signature. Each
`ModelCall` names a model, a method key, and a *role*:

| Role | Runs on | Typical use |
|---|---|---|
| `ROLE_PREPROCESSOR` | Host | Tokenization, feature hashing, bucketizing, embedding lookup |
| `ROLE_MODEL` | Accelerator (or CPU) | The JAX function |
| `ROLE_POSTPROCESSOR` | Host | Detokenization, top-k, formatting |

```python
from xla_saved_model import serving_config
from xla_saved_model.protos import orchestration_pb2

orchestration = serving_config.Orchestration(model_calls=[
    serving_config.ModelCall(model_name='tf_model', method_key='preprocess',
                             role=orchestration_pb2.ROLE_PREPROCESSOR),
    serving_config.ModelCall(model_name='jax_model', method_key='serving_default',
                             role=orchestration_pb2.ROLE_MODEL),
    serving_config.ModelCall(model_name='tf_model', method_key='postprocess',
                             role=orchestration_pb2.ROLE_POSTPROCESSOR),
])
```

By default calls run in the order listed and each call's outputs become the
next call's inputs. With `Orchestration(graph_based=True)` you name
`input_keys` and `output_keys` on each call and the exporter sorts them
topologically, which lets two preprocessors feed one model.

`role` defaults to `ROLE_MODEL`. Set it explicitly on host-side calls: the
batch scheduler uses roles to work out which stages a
`batch_component` such as `PRE_PROCESSOR_AND_MODEL_FUNCTION` covers.

A single `JaxModel` with one function needs no orchestration; the exporter
synthesizes a one-call pipeline. Anything with two or more models requires
one.

## Platforms

A JAX function is **lowered** for a set of platforms at export:

```python
jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']}
```

The manifest records the list. At load time the runtime picks the lowering
that matches the device it is running on and compiles it with PJRT. Loading a
bundle on a platform it was not lowered for fails, with a message that says
so. `xsm-cli show <dir> --details serving_default` prints the list under
"Lowering platforms".

The server flag `--orbax_grpc_server_platform` takes `cpu`, `cuda`, or `tpu`.
The Python `TargetConfig` and `xsm-cli recompile --platform` use the same
spelling; `gpu` is not accepted anywhere.

JAX functions lowered for `cpu` can also act as host-side preprocessors in a
pipeline whose main model runs on an accelerator. `examples/nnx_jax_preprocessor`
does this with an embedding lookup.

## Naming you will see in logs

Internally each function in the manifest is named
`<signature>_<model>_<method_key>`, so a default export produces
`serving_default_jax_model_serving_default`. A single-model export stores
weights under the key `weights`; multi-model exports use `<model>_weights`.
These names surface in `xsm-cli show` and in runtime error messages.

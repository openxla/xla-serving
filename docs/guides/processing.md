# Pre- and post-processing

Most production models do not take a dense float tensor. They take ids,
strings, variable-length lists, and they return something that needs
decoding. XSE runs that work on the host, in the same process as the
accelerator call, as extra stages in the serving signature. There are three
ways to write a stage; pick by what you already have.

| Stage type | Written in | Needs at serving time | Good for |
|---|---|---|---|
| `TensorflowModel` | `tf.function` | A `-with-tf` image or `--define ENABLE_TF=1` | Existing TF preprocessing, tokenizers, `tf.lookup` tables, TPU embedding layers, ragged and sparse inputs |
| `JaxModel` lowered for `cpu` | JAX | Nothing extra | Numeric transforms, embedding lookups, anything you can write with `jnp` |
| `CompositeModel` | C++ (`xla_serving_data`) | `xla_serving_data` at export; nothing extra at serving | Latency-critical transforms, large static resources, no TF dependency |

All three plug into the same place: `ServingConfig.orchestration`. The
[concepts page](../concepts.md#orchestration) explains the `ModelCall` and
role vocabulary; this page shows each stage type in use. The examples all
come from the repository's `examples/` directory, trimmed.

## TensorFlow stages

Wrap your `tf.function`s in a `TensorflowModel`. Each function takes a dict
of tensors and returns a dict of tensors. Resources the functions close over,
such as embedding or lookup tables, go in `trackable_resources` so that they
are saved with the TF SavedModel.

```python
import tensorflow as tf
import xla_saved_model
from xla_saved_model import serving_config
from xla_saved_model.protos import orchestration_pb2


@tf.function
def postprocess(inputs):
  return {'pred': inputs['pred'] + 10.0}


tf_model = xla_saved_model.TensorflowModel(
    name='tf_model',
    tf_functions={
        'preprocess': embedding_layer,     # a tf.Module with a __call__
        'postprocess': postprocess,
    },
    trackable_resources=[embedding_layer],
)

jax_model = xla_saved_model.JaxModel(
    name='jax_model',
    apply_fn_bundle=xla_saved_model.ApplyFnBundle(
        apply_fn=apply_fn,
        input_polymorphic_shapes={f'feature_{i}': 'b, _, _' for i in range(1, 5)},
        jax_export_kwargs={'platforms': ['cpu', 'tpu']},
    ),
    params=params,
)

xla_saved_model.save(
    [tf_model, jax_model],
    export_path=export_path,
    serving_configs=[
        xla_saved_model.ServingConfig(
            input_signature={
                f'feature_{i}': tf.TensorSpec((None, 1), tf.int32)
                for i in range(1, 5)
            },
            orchestration=serving_config.Orchestration(model_calls=[
                serving_config.ModelCall('tf_model', 'preprocess',
                                         role=orchestration_pb2.ROLE_PREPROCESSOR),
                serving_config.ModelCall('jax_model', 'serving_default',
                                         role=orchestration_pb2.ROLE_MODEL),
                serving_config.ModelCall('tf_model', 'postprocess',
                                         role=orchestration_pb2.ROLE_POSTPROCESSOR),
            ]),
        )
    ],
)
```

Things to know:

- The signature's `input_signature` describes what the *first* stage
  receives, so with a TF preprocessor it is written in `tf.TensorSpec`s with
  `None` for the batch dimension. The JAX model's
  `input_polymorphic_shapes` describes what *it* receives, which is the
  preprocessor's output. Here the embedding layer turns `(b, 1)` ids into
  `(b, 1, dim)` vectors, hence `'b, _, _'`.
- Ragged and sparse inputs work: use `tf.RaggedTensorSpec` or
  `tf.SparseTensorSpec` in the input signature. `examples/nnx_tf_ragged`
  shows two TF preprocessors passing ragged components between them.
- Only one `TensorflowModel` is allowed per bundle. Put all your TF functions
  in it under different method keys.
- The functions run under TFRT inside the server. The export writes a
  standard SavedModel under `tf_saved_model/`, so `saved_model_cli` works on
  it if you need to debug the TF side on its own.
- Serving a bundle with a `TensorflowModel` needs a server with TF compiled
  in. The images with `-with-tf` in the name have it; for a source build
  pass `--define ENABLE_TF=1`. A plain server cannot load the TF stages and
  refuses the bundle.

The ViT and DLRM examples use this path for image decoding and for
`TPUEmbeddingForServing` lookups respectively.

## JAX stages on the CPU

If your preprocessing is numeric, you do not need TensorFlow. Export a second
`JaxModel` lowered only for `cpu` and give it the preprocessor role. The
runtime executes it as a StableHLO kernel on the host and feeds its outputs
to the accelerator stage.

```python
embedding = xla_saved_model.JaxModel(
    name='embedding_layer',
    apply_fn_bundle=xla_saved_model.ApplyFnBundle(
        apply_fn=embedding_apply_fn,
        input_polymorphic_shapes={f'feature_{i}': 'b, _' for i in range(1, 5)},
        jax_export_kwargs={'platforms': ['cpu']},
    ),
    params=embedding_params,
)

main = xla_saved_model.JaxModel(
    name='jax_model',
    apply_fn_bundle=xla_saved_model.ApplyFnBundle(
        apply_fn=apply_fn,
        input_polymorphic_shapes={f'feature_{i}': 'b, _, _' for i in range(1, 5)},
        jax_export_kwargs={'platforms': ['cpu', 'tpu']},
    ),
    params=params,
)

xla_saved_model.save(
    [embedding, main],
    export_path=export_path,
    serving_configs=[
        xla_saved_model.ServingConfig(
            input_signature={
                f'feature_{i}': jax.ShapeDtypeStruct(('b', 1), jnp.int32)
                for i in range(1, 5)
            },
            orchestration=serving_config.Orchestration(model_calls=[
                serving_config.ModelCall('embedding_layer',
                                         role=orchestration_pb2.ROLE_PREPROCESSOR),
                serving_config.ModelCall('jax_model',
                                         role=orchestration_pb2.ROLE_MODEL),
            ]),
        )
    ],
)
```

Each `JaxModel` gets its own checkpoint under `checkpoint/<name>/`, so a
large embedding table in the preprocessor stays on the host and never
occupies accelerator memory. This is the whole point of the pattern for
recommenders: the lookup is memory-bound and the MLP is compute-bound, and
they belong on different devices.

A detail: preprocessors exported this way have 64-bit outputs demoted to
32-bit unless `jax_enable_x64` is on, so that the handoff to the model stage
matches what the model was traced with.

## C++ stages

For transforms where TF is too heavy or too slow, `xla_serving_data`
provides a C++ `TransformBase` interface, a `ResourceManager` for assets
such as vocabulary files, and a `CompositeModel` that chains transforms into
one pipeline stage. At serving time the runtime executes the chain as a
single kernel with `HostBuffer` inputs and outputs; there is no TF and no
Python involved.

The export side needs the `xla_serving_data` wheel and your transforms'
Python bindings. Each transform is exposed to the exporter through a small
module object with three members:

```python
class AddFromResourceModule:

  def __init__(self, resource):
    self._transform = my_transforms_pybind.AddIntFromResourceTransform(resource)

  @property
  def transform(self):
    return self._transform

  def export_assets(self) -> dict[str, str]:
    return self.transform.assets()          # name -> file path, copied into the bundle

  def get_output_signature(self, input_signature):
    return input_signature                  # shape/dtype after this transform
```

Compose them and use the result like any other model:

```python
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.export import main_lib

preprocessor = main_lib.CompositeModel(
    name='transform_inputs',
    transforms=[AddTwoInputBuffersModule(), AddFromResourceModule(vocab)],
)

xla_saved_model.save(
    [preprocessor, jax_model],
    export_path=export_path,
    serving_configs=[
        xla_saved_model.ServingConfig(
            input_signature={
                'x': shlo_type.ShloTensorSpec(shape=(None, 4), dtype=shlo_type.ShloDType.i32),
                'y': shlo_type.ShloTensorSpec(shape=(None, 4), dtype=shlo_type.ShloDType.i32),
            },
            orchestration=serving_config.Orchestration(model_calls=[
                serving_config.ModelCall('transform_inputs',
                                         role=orchestration_pb2.ROLE_PREPROCESSOR),
                serving_config.ModelCall('jax_model',
                                         role=orchestration_pb2.ROLE_MODEL),
            ]),
        )
    ],
)
```

The input signature of a C++ stage can be written with either
`jax.ShapeDtypeStruct` (standard in newer code) or `ShloTensorSpec`, the
exporter's internal tensor type.

`examples/composite_processor` is a complete, buildable version of a custom
linear chain, including the C++ transforms, their pybind wrappers, and the
Bazel rules. You do not have to write every transform yourself:
`xla_serving_data.ops` ships ready-made ones, from `bucket_pad` and
`cpu_embedding` (which `examples/dlrm/export_jce.py` uses in place of TF
embedding lookups) to the elementwise `math` and `bitwise` families,
reductions, scans, and packing. For multi-input / multi-output DAGs built out
of those ops, `CompositeModel.from_graph` wires a `processor.GraphConfig`
directly (`examples/math_preprocessor` shows the pattern).

`xsm-cli show <dir>` lists a composite stage as `CompositeProcessor`, and
`--details <name>` prints its transforms, their edges, and the assets it
carries. Decoding the stage needs the `xla-serving-data` wheel; without it the
CLI says so and skips that part.

## Graph-based orchestration

The sequential form above feeds each stage's whole output dict to the next
stage. When that is not the shape of your pipeline, for instance two
independent preprocessors whose outputs are both consumed by the model, set
`graph_based=True` and name the keys:

```python
serving_config.Orchestration(
    graph_based=True,
    model_calls=[
        serving_config.ModelCall('tokenizer', input_keys=['text'],
                                 output_keys=['input_ids'],
                                 role=orchestration_pb2.ROLE_PREPROCESSOR),
        serving_config.ModelCall('image_decoder', input_keys=['image_bytes'],
                                 output_keys=['pixels'],
                                 role=orchestration_pb2.ROLE_PREPROCESSOR),
        serving_config.ModelCall('jax_model', input_keys=['input_ids', 'pixels'],
                                 output_keys=['logits'],
                                 role=orchestration_pb2.ROLE_MODEL),
    ],
)
```

The exporter sorts calls topologically by key and the runtime routes buffers
by name. `input_signature` must be a mapping in this mode. Batched stages
must still be adjacent in the sorted order; see [Batching](batching.md).

## Choosing

If you already have TF preprocessing, keep it; the cost is a larger server
image. If you are starting fresh and the transform is arithmetic, write it
in JAX and lower for CPU; it is the least machinery. Reach for C++ when a
profile tells you the host stage is on the critical path, or when you need
to ship a server without TensorFlow in it.

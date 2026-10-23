# Exporting models

`xla_saved_model.save()` packages a model's computation, its weights, and its
serving signatures into a directory on disk. This guide covers the export API
for single-stage models across JAX, Flax, Keras 3, and PyTorch (`torchax`). For
multi-stage pipelines that add host-side pre- or post-processing, see
[Pre- and post-processing](processing.md).

## The export contract

Every JAX model export revolves around three objects:

1. **`ApplyFnBundle`** — wraps a pure function `apply_fn(params, inputs) -> outputs`
   together with its shape-polymorphism rules and target hardware platforms.
2. **`JaxModel`** — pairs one or more `ApplyFnBundle`s with a parameter PyTree
   (or a path to an existing Orbax checkpoint).
3. **`ServingConfig`** — declares a client-facing signature name, the expected
   input tensor shapes and dtypes, and optional [batching](batching.md) and
   [orchestration](processing.md) rules.

```python
import jax
import jax.numpy as jnp
import xla_saved_model

xla_saved_model.save(
    xla_saved_model.JaxModel(
        apply_fn_bundle=xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,
            input_polymorphic_shapes={'features': 'b, _'},
            jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
        ),
        params=params,
    ),
    export_path='/tmp/my_model',
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

### How `apply_fn` is structured

`apply_fn` takes two positional arguments—`(params, inputs)`—and returns
`outputs`:

- **`params`** is a PyTree of model weights matching `JaxModel(params=...)`. If
  your model is stateless (`params=None`), the first argument is `None`.
- **`inputs`** is a dict mapping input tensor names to JAX arrays, matching
  `ServingConfig.input_signature`.
- **`outputs`** is a dict mapping output tensor names to JAX arrays.

The keys of `inputs` and `outputs` become the wire names in gRPC
`PredictRequest.inputs` and `PredictResponse.outputs` (and the keyword
arguments/return dict in `ModelRunner`).

## Exporting from different frameworks

::::{tab-set}
:::{tab-item} Flax NNX
Split the `nnx.Module` into its static graph definition and parameter state
with `nnx.split`, and merge them inside `apply_fn`:

```python
from flax import nnx
import jax
import jax.numpy as jnp
import xla_saved_model

model = MyNnxModel(rngs=nnx.Rngs(0))
graphdef, params, rest_state = nnx.split(
    model, nnx.filterlib.Not(nnx.RngState), ...
)


def apply_fn(params, inputs):
  module = nnx.merge(graphdef, params, rest_state, copy=True)
  return {'logits': module(inputs['features'])}
```

See `examples/nnx/export.py` for a complete recommender example, including
`bfloat16` parameter casting.
:::

:::{tab-item} Flax Linen / plain JAX
For Flax Linen modules (or plain JAX functions), pass `params` straight into
`module.apply`:

```python
import jax
import jax.numpy as jnp
import xla_saved_model

module = MyLinenModule()
variables = module.init(jax.random.key(0), jnp.zeros((1, 16), jnp.float32))
params = variables['params']


def apply_fn(params, inputs):
  logits = module.apply({'params': params}, inputs['features'])
  return {'logits': logits}
```
:::

:::{tab-item} Keras 3 (JAX backend)
With `KERAS_BACKEND=jax`, use Keras 3's public `stateless_call` to run the
model as a pure function of its trainable and non-trainable variables:

```python
import os
os.environ['KERAS_BACKEND'] = 'jax'

import jax.numpy as jnp
import keras

keras_model = build_or_load_keras_model()
params = {
    'trainable': [v.value for v in keras_model.trainable_variables],
    'non_trainable': [v.value for v in keras_model.non_trainable_variables],
}


def apply_fn(params, inputs):
  logits, _ = keras_model.stateless_call(
      params['trainable'],
      params['non_trainable'],
      inputs['images'],
      training=False,
  )
  return {'logits': jnp.asarray(logits, dtype=jnp.float32)}
```

See `examples/vit/export.py` for a full Vision Transformer (ViT) export using
`keras_hub`.
:::

:::{tab-item} PyTorch (via torchax)
Install the `torch` extra (`pip install "xla-saved-model[torch]"`). The
`torch2xsm.convert_to_jax` helper uses
[torchax](https://github.com/pytorch/xla/tree/master/torchax) to extract a
PyTorch module's state dict and wrap its forward pass as a JAX callable:

```python
import jax
import jax.numpy as jnp
import torch
import xla_saved_model
from xla_saved_model._src.torch2xsm import main_lib as torch2xsm

pt_model = MyTorchModule().eval()
weights, jfunc = torch2xsm.convert_to_jax(pt_model)


def apply_fn(params, inputs):
  out = jfunc(params, (inputs['x'],))
  return {'y': out}


xla_saved_model.save(
    xla_saved_model.JaxModel(
        apply_fn_bundle=xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,
            input_polymorphic_shapes={'x': 'b, _'},
            jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
        ),
        params=weights,
    ),
    export_path='/tmp/torch_model',
    serving_configs=[
        xla_saved_model.ServingConfig(
            input_signature={'x': jax.ShapeDtypeStruct(('b', 10), jnp.float32)},
        )
    ],
)
```

See `examples/torchax/export.py`.
:::
::::

## Shapes: fixed, symbolic, and bucketed

### Fixed shapes

If every request has the exact same shape, use integer dimensions in
`input_signature` and omit `input_polymorphic_shapes`:

```python
input_signature = {'features': jax.ShapeDtypeStruct((1, 16), jnp.float32)}
```

The runtime rejects requests with any other shape.

### Symbolic batch dimensions

To serve variable batch sizes from a single exported StableHLO module, put a
dimension variable name (such as `'b'`) in `input_signature` and provide a
matching [`jax.export` shape spec](https://docs.jax.dev/en/latest/export/shape_poly.html)
in `input_polymorphic_shapes`:

```python
apply_fn_bundle = xla_saved_model.ApplyFnBundle(
    apply_fn=apply_fn,
    input_polymorphic_shapes={
        'user_features': 'b, _',
        'item_features': 'b, _',
    },
    jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
)

input_signature = {
    'user_features': jax.ShapeDtypeStruct(('b', 64), jnp.float32),
    'item_features': jax.ShapeDtypeStruct(('b', 64), jnp.float32),
}
```

Using the same symbol `'b'` across multiple inputs enforces that they share the
same batch size in every request. In `input_polymorphic_shapes`, `_` is a
placeholder for a dimension that stays concrete (taken from `input_signature`).
If your function requires algebraic constraints on a symbolic dimension (for
example `'b >= 1'`), pass them in `polymorphic_constraints=('b >= 1',)`.

### Concrete batch buckets (for AOT compilation)

If you also pass `input_polymorphic_shape_symbol_values`, the exporter traces
and saves one concrete StableHLO function per value instead of a single
symbolic function:

```python
apply_fn_bundle = xla_saved_model.ApplyFnBundle(
    apply_fn=apply_fn,
    input_polymorphic_shapes={'features': 'b, _'},
    input_polymorphic_shape_symbol_values={'b': [1, 8, 32]},
    jax_export_kwargs={'platforms': ['cuda']},
)
```

This is required for [ahead-of-time compilation](aot.md), because XLA's offline
compiler needs concrete tensor shapes. Pair it with matching
`allowed_batch_sizes=[1, 8, 32]` in [batching](batching.md) so incoming
requests are padded to one of the compiled buckets.

## Target platforms

`jax_export_kwargs={'platforms': [...]}` controls which hardware backends the
StableHLO is lowered for. Valid platform names are `'cpu'`, `'cuda'`, and
`'tpu'`.

:::{important}
Always set `jax_export_kwargs={'platforms': [...]}` explicitly. When omitted,
`platforms` defaults to `['tpu']`, and loading the resulting bundle on a CPU or
CUDA runtime fails because no lowering for that platform is present in the
manifest.
:::

Lowering is completely deviceless: you can export with
`platforms=['cpu', 'cuda', 'tpu']` on a laptop or CI runner that has no GPU or
TPU attached.

## Weights and existing checkpoints

By default `JaxModel(apply_fn_bundle=..., params=params)` writes `params` into
`<export_path>/checkpoint/` as an Orbax checkpoint.

If your training job already saved an Orbax `StandardCheckpointHandler`
checkpoint and you want to reference it in the manifest without re-saving, pass
`checkpoint_path` (and still provide `params`—or a tree of `jax.ShapeDtypeStruct`s
with the same structure—so the exporter can trace `apply_fn`):

```python
model = xla_saved_model.JaxModel(
    apply_fn_bundle=apply_fn_bundle,
    params=params_spec,
    checkpoint_path='checkpoint',
)
```

## Multi-device sharding

To export a model sharded across multiple devices:

1. Wrap `apply_fn` in `jax.jit` with explicit `in_shardings` and
   `out_shardings` built from a `jax.sharding.Mesh` and `NamedSharding`.
2. Attach matching `NamedSharding`s to the `params` leaves and
   `input_signature` `ShapeDtypeStruct`s.
3. Pass `jax_mesh=mesh` to `JaxModel` so the exporter records the device mesh
   in the bundle's XLA compile options.

```python
from jax.experimental import mesh_utils
from jax.sharding import Mesh, NamedSharding, PartitionSpec as P
import xla_saved_model

devices = mesh_utils.create_device_mesh((4,))
mesh = Mesh(devices, axis_names=('tp',))

replicated = NamedSharding(mesh, P())
tp_sharded = NamedSharding(mesh, P('tp', None))

sharded_apply = jax.jit(
    apply_fn,
    in_shardings=({'w': tp_sharded}, {'x': replicated}),
    out_shardings={'y': replicated},
)

model = xla_saved_model.JaxModel(
    apply_fn_bundle=xla_saved_model.ApplyFnBundle(
        apply_fn=sharded_apply,
        jax_export_kwargs={'platforms': ['tpu']},
    ),
    params=params,
    jax_mesh=mesh,
)
```

If you do not have the target multi-device slice on the export machine, create
an abstract mesh or virtual CPU devices (`os.environ['XLA_FLAGS'] =
'--xla_force_host_platform_device_count=4'`) before importing `jax`.

## Multiple serving signatures

A single bundle can expose several serving signatures—for example a `score`
signature and an `embed` signature that share the same weight checkpoint. Pass
a dict of `ApplyFnBundle`s to `JaxModel` and one `ServingConfig` per signature:

```python
from xla_saved_model import serving_config

model = xla_saved_model.JaxModel(
    name='encoder',
    apply_fn_bundle={
        'score': xla_saved_model.ApplyFnBundle(
            apply_fn=score_fn,
            input_polymorphic_shapes={'features': 'b, _'},
            jax_export_kwargs={'platforms': ['cpu', 'cuda']},
        ),
        'embed': xla_saved_model.ApplyFnBundle(
            apply_fn=embed_fn,
            input_polymorphic_shapes={'features': 'b, _'},
            jax_export_kwargs={'platforms': ['cpu', 'cuda']},
        ),
    },
    params=params,
)

xla_saved_model.save(
    model,
    export_path='/tmp/encoder',
    serving_configs=[
        xla_saved_model.ServingConfig(
            serving_signature='score',
            input_signature={'features': jax.ShapeDtypeStruct(('b', 16), jnp.float32)},
            orchestration=serving_config.Orchestration(
                model_calls=[serving_config.ModelCall('encoder', 'score')]
            ),
        ),
        xla_saved_model.ServingConfig(
            serving_signature='embed',
            input_signature={'features': jax.ShapeDtypeStruct(('b', 16), jnp.float32)},
            orchestration=serving_config.Orchestration(
                model_calls=[serving_config.ModelCall('encoder', 'embed')]
            ),
        ),
    ],
)
```

Clients select which signature to run via `model_spec.signature_name` in
`PredictRequest` (or `runner.signature('embed')` in Python).

## `xla_saved_model.save()` reference

```python
xla_saved_model.save(
    models,                  # JaxModel, or a sequence of [JaxModel, TensorflowModel, CompositeModel]
    export_path,             # destination directory path
    serving_configs,         # sequence of ServingConfig (at least one)
    *,
    warmup_examples=None,    # optional dict[signature_name, Sequence[Mapping[str, ArrayLike]]]
    save_shlo_to_file=False, # write StableHLO to <fn>.shlo instead of inlining in manifest.pb
)
```

Set `save_shlo_to_file=True` when a model's StableHLO bytecode is large enough
to approach protobuf's 2 GiB message limit.

# Ahead-of-time compilation

By default a server compiles StableHLO to device code when it loads a bundle,
and again for each new input shape it meets. For a small CPU model that is
sub-second. For a large model on TPU it is tens of seconds of startup, a
compiler heap spike that can be several times the size of the model, and a
first-request latency cliff after every restart or autoscale event.

Ahead-of-time (AOT) compilation moves that step into your build pipeline.
`xsm-cli recompile` takes a bundle, compiles every function for a platform
and topology you name, and writes a new bundle with the executables inside.
A server loading it maps the executables read-only and never invokes the
compiler. The `examples/aot_benchmark` script measures the difference on an
8-layer MLP: cold start drops by two orders of magnitude on CPU and the
compiler's memory spike is gone.

## Two things the export needs

**Concrete shapes.** The AOT compiler compiles the StableHLO exactly as
exported, and XLA cannot compile a module with a symbolic dimension. Give
the exporter the list of batch sizes you will serve, and it writes one
concrete function per value:

```python
xla_saved_model.ApplyFnBundle(
    apply_fn=apply_fn,
    input_polymorphic_shapes={'features': 'b, _'},
    input_polymorphic_shape_symbol_values={'b': [1, 8, 32]},
    jax_export_kwargs={'platforms': ['cpu']},
)
```

`xsm-cli show` then reports the function as a "polymorphic function" with
three variants, and `recompile` produces three executables. Keep the list
in step with `allowed_batch_sizes` in your [batch options](batching.md).
Without this, `recompile` fails with an opaque XLA error mentioning
`dynamic_broadcast_in_dim` or similar.

**A single lowering platform.** Lower the function for the one platform
you are compiling for. A multi-platform export (`platforms=['cpu', 'cuda',
'tpu']`) adds a hidden platform-index argument to the module; the JIT path
handles it, but an executable compiled from it currently fails at run time
with `Execution supplied N arguments but compiled program expected N+1`.
If you need the same model on several platforms, export once per platform.

## Compiling

```bash
xsm-cli recompile \
    --input-model-path=/tmp/classifier \
    --output-model-path=/tmp/classifier_aot \
    --platform=cpu \
    --target-topology-label=cpu_target
```

The output is a complete bundle: the input is copied, the executables are
added, and the manifest gains a `compiled_executables` entry keyed by the
label. By default it lands in a timestamped subdirectory,
`/tmp/classifier_aot/1725890000/`, which suits loaders that pick the newest
version. Pass `--version=3` to name the subdirectory yourself, or
`--no-auto-timestamp` to write straight into the output path.

::::{tab-set}
:::{tab-item} CPU
No extra inputs; compilation runs wherever you invoke the CLI.

```bash
xsm-cli recompile \
  --input-model-path=/tmp/m \
  --output-model-path=/tmp/m_aot \
  --platform=cpu \
  --target-topology-label=cpu_target
```
:::
:::{tab-item} CUDA
Compilation is deviceless, but XLA's GPU autotuner wants to run candidate
kernels. Without a GPU present it falls back to defaults, and the first
requests on the real GPU pay 10–30 s of autotuning. To avoid that, dump an
autotune database from a warm JIT server on the target GPU and pass it in:

```bash
# On a machine with the target GPU, while serving JIT:
export XLA_FLAGS="--xla_dump_autotune_cache_to=/tmp/autotune.pb"

# Then:
xsm-cli recompile \
  --input-model-path=/tmp/m \
  --output-model-path=/tmp/m_aot \
  --platform=cuda \
  --target-topology-label=cuda_h100 \
  --autotune-database-path=/tmp/autotune.pb
```

`--autotune-database-path` accepts a `.pb` or `.textproto` file (read
only) or a directory (read-write per-fusion cache).
:::
:::{tab-item} TPU
Compilation targets a specific chip generation and slice topology. Name the
topology:

```bash
xsm-cli recompile \
  --input-model-path=/tmp/m \
  --output-model-path=/tmp/m_aot \
  --platform=tpu \
  --target-topology-label=tpu_2x2x1 \
  --target-topology=2x2x1
```

or pass `--serialized-topology=<file>` with a serialized
`TopologyDescription` captured from the target slice when mesh dimensions
alone are not enough. The executable is bound to that topology; see the
note on stripping below before you drop the StableHLO.
:::
::::

The label (`-l`, `--target-topology-label`) is a free-form string stored in
the manifest. The runtime looks executables up by it, so pick something
that identifies the hardware: `cpu_target`, `cuda_h100`, `tpu_v6e_2x2`. A
bundle can carry executables for several labels; run `recompile` once per
label with the previous output as input.

### Keeping or stripping the StableHLO

By default the StableHLO stays in the bundle next to the executable. That
costs disk and host memory, often tens of megabytes, but it means a server
on unexpected hardware can fall back to JIT if you let it. Pass
`--strip-stablehlo` to remove it; the bundle then only runs on hardware
matching one of its labels, and fails fast otherwise. Strip once you trust
the deployment target is fixed, and not before. On TPU in particular, a
stripped bundle compiled for `2x2x1` cannot be rescheduled onto a different
slice shape.

### All flags

| Flag | Default | Meaning |
|---|---|---|
| `-i`, `--input-model-path` | required | Bundle to compile |
| `-o`, `--output-model-path` | required | Where to write the result |
| `-p`, `--platform` | required | `cpu`, `cuda`, or `tpu` |
| `-l`, `--target-topology-label` | required | Manifest key for the executables |
| `--target-topology` | | TPU mesh, e.g. `2x2x1`. Exclusive with the next flag |
| `--serialized-topology` | | Path to a serialized `TopologyDescription` |
| `--autotune-database-path` | | CUDA autotune results, file or directory |
| `--include-stablehlo` / `--strip-stablehlo` | include | Keep or drop the portable IR |
| `-v`, `--version` | | Integer version subdirectory |
| `--auto-timestamp` / `--no-auto-timestamp` | on | Timestamped subdirectory when `--version` is not given |
| `--overwrite` / `--no-overwrite` | off | Replace an existing output directory |

The same operation is available from Python as `xla_saved_model.recompile`
with a `TargetConfig` per label, which is convenient inside an export
script:

```python
import xla_saved_model

xla_saved_model.recompile(
    '/tmp/classifier', '/tmp/classifier_aot',
    xla_saved_model.TargetConfig(platform='cpu', target_topology_label='cpu_target'),
    auto_timestamp=False,
)
```

## Serving an AOT bundle

The runtime needs to know which label to look for, and whether JIT is an
acceptable fallback when it finds no match.

::::{tab-set}
:::{tab-item} ModelRunner
```python
from xla_saved_model_runner import model_runner

runner = model_runner.ModelRunner(
    model_path='/tmp/classifier_aot',
    platform='cpu',
    target_topology_label='cpu_target',
    allow_jit_fallback_for_aot=False,
)
```
:::
:::{tab-item} C++
```cpp
#include "runtime/interpreter/interpreter.h"
#include "runtime/interpreter/interpreter_create_options.h"

auto options = std::make_unique<xla_serving::InterpreterCreateOptions>();
options->model_path = "/tmp/classifier_aot";
options->model_name = "classifier";
options->platform_name = "cpu";
options->target_topology_label = "cpu_target";
options->allow_jit_fallback_for_aot = false;

TF_ASSIGN_OR_RETURN(auto interpreter,
                    xla_serving::Interpreter::Create(std::move(options)));
```
:::
:::{tab-item} gRPC server
The standalone server does not yet expose `target_topology_label` or
`allow_jit_fallback_for_aot` as flags. Until it does, serve AOT bundles
through an embedding of the C++ `Interpreter` or use `ModelRunner`.
:::
::::

Requests whose shape has no matching executable fail with
`AOT cache miss: unseen input shape` when fallback is off. With
`allow_jit_fallback_for_aot=True` and the StableHLO still present, the
runtime compiles that shape on the spot instead, which turns a hard error
into a latency spike; choose based on which you would rather see in
production.

## Reusing output memory

An executable normally allocates fresh device buffers for its outputs on
every call. For large outputs on an accelerator that allocation is on the
request path. XLA can instead write results into memory the caller hands in,
and the exporter can arrange for that:

```python
xla_saved_model.ApplyFnBundle(
    apply_fn=apply_fn,
    use_input_aliasing=True,
    jax_export_kwargs={'platforms': ['cuda']},
)
```

With `use_input_aliasing=True` the exported function gets one hidden extra
argument per output, aliased to that output. The caller's signature does not
change; the extra arguments are the runtime's business. Three conditions
apply:

- Shapes must be concrete (the usual AOT requirement; use
  `input_polymorphic_shape_symbol_values` or fixed shapes).
- The pipeline must be graph-based, with `output_keys` on the `ModelCall`.
  The runtime addresses the buffers by component output name, and only a
  graph-based export records those names.
- The bundle must be served from AOT executables. The JIT path refuses to
  load a function exported for input aliasing, so `recompile` is not
  optional here and neither is `allow_jit_fallback_for_aot=False`.

On the serving side, embedders of the C++ `Interpreter` can supply pre-allocated
output memory through `RunOptions` in two ways:

- **`RunOptions::preallocated_output_arrays`** (device memory): a map from
  component output name to pre-allocated `xla::ifrt::ArrayRef`. Requires an AOT
  component exported with `use_input_aliasing=True`. Each array must match the
  output's dtype, shape, sharding, and layout, and is donated to the execution;
  the result comes back backed by that device memory.
- **`RunOptions::preallocated_output_host_buffers`** (host memory): a map from
  component output name to pre-allocated `HostBuffer` (typically pinned host
  memory for fast DMA). Works with any component and avoids host-side output
  allocation when transferring results from device to host.

You can pre-allocate any subset of a component's outputs (for unmapped outputs,
the runtime allocates memory as usual), or provide both a pre-allocated device
array and a pre-allocated host buffer for the same output to eliminate both
device-side and host-side allocations. Neither `ModelRunner` nor the gRPC
server exposes `RunOptions` today, so this is a feature for C++ embedders.

# Ahead-Of-Time (AOT) Model Recompilation & Serving Guide

This guide walks through precompiling and serving exported XLA Serving models
Ahead-Of-Time (AOT) to eliminate Just-In-Time (JIT) compilation latency and
stabilize host container memory in production.

---

## 1. Exporting the Model

Export your trained JAX or TensorFlow model using `xla_saved_model.save`:

```python
import jax
import jax.numpy as jnp
import xla_saved_model

class SimpleModel:
  def __call__(self, x):
    return jnp.sin(x) + 1.0

model = SimpleModel()
bundle = xla_saved_model.ApplyFnBundle(apply_fn=model)
jax_model = xla_saved_model.JaxModel(apply_fn_bundle=bundle, name="simple_model")
serving_config = xla_saved_model.ServingConfig(
    serving_signature="serving_default",
    input_signature={
        "x": jax.ShapeDtypeStruct((1, 32), jnp.float32)
    },
)

xla_saved_model.save(
    models=jax_model,
    export_path="/path/to/exported_model",
    serving_configs=[serving_config],
)
```

---

## 2. Recompiling Offline via `xsm-cli recompile`

Precompile the model Ahead-Of-Time using the `xsm-cli` command-line tool.

### StableHLO Bytecode: Preservation vs. Stripping

By default (`--include-stablehlo`), the compiler retains portable StableHLO
bytecode inside the model bundle alongside the compiled hardware executable.
Passing `--strip-stablehlo` is an optional optimization with key trade-offs:

* **Preserving StableHLO (Default)**:
  * *Benefits*: Enables runtime JIT fallback (allow_jit_fallback_for_aot=True)
    if deployed onto different hardware or unexpected accelerator topologies.
  * *Downside*: Retains intermediate bytecode in the bundle, increasing storage
    and host resident memory (~10–100+ MB depending on graph complexity).
* **Stripping StableHLO (`--strip-stablehlo`)**:
  * *Benefits*: Minimizes artifact size and reduces host memory footprint by
    ensuring only the read-only memory-mapped executable is loaded.
  * *Downside*: Removes runtime JIT fallback. If hardware or topology
    does not match the compiled target, execution fails fast.
* **TPU Mesh Considerations**:
  * On TPU, compilation targets specific slice topologies (e.g. `2x2x1` for
    TPU v7x 3D meshes). Stripping StableHLO locks the bundle to that exact
    hardware slice; scheduling onto a different slice dimension prevents JIT
    re-lowering and triggers a fatal topology mismatch.

### CPU Target
Compile for host CPU execution:

```bash
xsm-cli recompile \
    --input-model-path=/path/to/exported_model \
    --output-model-path=/path/to/aot_model_cpu \
    --platform=cpu \
    --target-topology-label="cpu_target"

# Optional: Pass --strip-stablehlo to minimize bundle size if JIT fallback
# is not required.
```

### CUDA GPU Target (Enterprise & Datacenter Serving)
For CUDA GPU targets (e.g. NVIDIA A100/H100), online autotuning can delay the
first request by 10–30 seconds. Providing an offline autotune database locks in
optimal GEMM and convolution fusions:

```bash
# Optional: Dump autotune results during export or warmup
export XLA_FLAGS="--xla_gpu_dump_autotune_results_to=/path/to/autotune.pb"

# Recompile using the offline autotune database
xsm-cli recompile \
    --input-model-path=/path/to/exported_model \
    --output-model-path=/path/to/aot_model_cuda \
    --platform=cuda \
    --target-topology-label="cuda_target" \
    --autotune-database-path=/path/to/autotune.pb
```

---

## 3. Comprehensive AOT Parameter Reference

### A. `xsm-cli recompile` CLI Flags

Deviceless compilation is active by default. Target topologies are
automatically deduced from `--target-topology` or the platform backend
without requiring attached hardware.

#### Quick Reference Table

| Flag | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `--input-model-path`, `-i` | String | *Required* | Input model directory. |
| `--output-model-path`, `-o` | String | *Required* | Output model directory. |
| `--platform`, `-p` | String | *Required* | Target: cpu, cuda, or tpu. |
| `--target-topology-label`, `-l` | String | *Required* | Manifest label. |
| `--target-topology` | String | `None` | Mesh topology (e.g. 2x2x1). |
| `--serialized-topology` | String | `None` | Path to topology proto. |
| `--autotune-database-path` | String | `None` | CUDA autotune file or dir. |
| `--include-stablehlo` | Flag | `True` | Retain bytecode in manifest. |
| `--strip-stablehlo` | Flag | `False` | Strip bytecode to save RAM. |
| `--version`, `-v` | Int | `None` | Version subdirectory. |
| `--auto-timestamp` | Bool | `True` | Timestamped output dir. |
| `--overwrite` | Bool | `False` | Overwrite existing output. |

#### Detailed Parameter Descriptions & Possible Values

* **`--input-model-path`, `-i`**:
  * *Type*: String | *Default*: *(Required)*
  * *Possible Values*: Any valid directory path to an exported XLA Serving
    model bundle directory (containing `manifest.jsonpb` or `manifest.pb`).
  * *Description*: Source model bundle to be recompiled Ahead-Of-Time.
* **`--output-model-path`, `-o`**:
  * *Type*: String | *Default*: *(Required)*
  * *Possible Values*: Any valid writable directory path.
  * *Description*: Destination path where the recompiled model bundle with
    compiled executables will be written.
* **`--platform`, `-p`**:
  * *Type*: String | *Default*: *(Required)*
  * *Possible Values*: `"cpu"`, `"cuda"`, `"tpu"` (case-insensitive).
    Note that `"gpu"` is not accepted; pass `"cuda"` directly.
  * *Description*: Target hardware backend for code generation.
* **`--target-topology-label`, `-l`**:
  * *Type*: String | *Default*: *(Required)*
  * *Possible Values*: Non-empty alphanumeric/underscore identifier string
    (e.g. `"cpu_target"`, `"cuda_sm90"`, `"tpu_2x2x1"`).
  * *Description*: Key stored in the manifest under `aot_executables` mapping.
    Runtime loaders (`ModelRunner` and `Interpreter`) query this label to
    select the matching precompiled binary.
* **`--target-topology`**:
  * *Type*: String | *Default*: `None`
  * *Possible Values*: Hardware mesh dimensions string (e.g. `"1x1x1"`,
    `"2x2x1"`, `"2x4x1"`).
  * *Description*: Target accelerator topology mesh. Mutually exclusive with
    `--serialized-topology`. Used primarily for TPU SPMD meshes.
* **`--serialized-topology`**:
  * *Type*: String | *Default*: `None`
  * *Possible Values*: File path pointing to serialized `TopologyDescription`
    protobuf bytes.
  * *Description*: Path to serialized topology description. Mutually exclusive
    with `--target-topology`. Required for deviceless TPU compilation when
    mesh dimensions alone are insufficient.
* **`--autotune-database-path`**:
  * *Type*: String | *Default*: `None`
  * *Possible Values*: File path ending in `.pb` or `.textproto` (read-only
    profile), or directory path (read-write per-fusion cache).
  * *Description*: CUDA autotune cache repository. Providing a pre-populated
    database locks optimal GEMM/convolution fusions and eliminates online
    tuning pauses during compilation.
* **`--include-stablehlo` / `--strip-stablehlo`**:
  * *Type*: Flag pair | *Default*: `--include-stablehlo`
  * *Possible Values*: Pass `--include-stablehlo` or `--strip-stablehlo`.
  * *Description*: Whether to preserve portable StableHLO IR inside the
    manifest. Preserving bytecode enables runtime JIT fallback if deployed to
    unexpected hardware; stripping bytecode minimizes bundle size and host
    resident memory.
* **`--version`, `-v`**:
  * *Type*: Integer | *Default*: `None`
  * *Possible Values*: Any non-negative integer (e.g. `1`, `2`, `101`).
  * *Description*: Explicit version number. When provided, output is saved into
    a versioned subdirectory: `<output_model_path>/<version>/`.
* **`--auto-timestamp` / `--no-auto-timestamp`**:
  * *Type*: Bool flag | *Default*: `True` (`--auto-timestamp`)
  * *Possible Values*: Flag toggle.
  * *Description*: When `--version` is omitted and `--auto-timestamp` is
    enabled, creates a timestamp-based subdirectory (e.g.
    `<output_model_path>/1725890000/`). If set to `--no-auto-timestamp`,
    writes directly to `<output_model_path>`.
* **`--overwrite` / `--no-overwrite`**:
  * *Type*: Bool flag | *Default*: `False` (`--no-overwrite`)
  * *Possible Values*: Flag toggle.
  * *Description*: Whether to overwrite the destination directory if it
    already exists.

### B. Serving Runtime Options Reference

| Parameter | Type | Default | Description |
| :--- | :--- | :--- | :--- |
| `target_topology_label` | String | `""` | Topology label in manifest. |
| `allow_jit_fallback_for_aot` | Bool | `False` | Fallback to JIT on mismatch. |
| `platform` | String | `""` | Target device platform. |

### C. Supplying Serving Runtime Options to the XLA Serving Runtime

To load and execute AOT precompiled models, supply `target_topology_label` and
`allow_jit_fallback_for_aot` to your chosen runtime interface:

#### 1. Python API (`ModelRunner`)
In Python environments (e.g. notebooks, offline evaluation, or services),
use `ModelRunner` from the `xla_saved_model_runner` package:

```python
from xla_saved_model_runner import model_runner

runner = model_runner.ModelRunner(
    model_name="my_model",
    model_path="/path/to/aot_model",
    platform="cuda",
    target_topology_label="cuda_target",
    allow_jit_fallback_for_aot=False,
)
sig = runner.signature("serving_default")
outputs = sig.run_with_dict_return(**sample_input)
```

#### 2. C++ In-Process Integration (`InterpreterCreateOptions`)
To integrate AOT execution directly into C++ applications or serving binaries,
pass the target topology label and fallback policy to
`InterpreterCreateOptions`:

```cpp
#include "runtime/interpreter/interpreter.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving

auto options = std::make_unique<xla_serving::InterpreterCreateOptions>();
options->model_path = "/path/to/aot_model";
options->model_name = "my_model";
options->platform_name = "cuda";
options->target_topology_label = "cuda_target";
options->allow_jit_fallback_for_aot = false;

TF_ASSIGN_OR_RETURN(std::unique_ptr<xla_serving::Interpreter> interpreter,
                    xla_serving::Interpreter::Create(std::move(options)));
```

#### 3. Containerized gRPC Serving (Docker)
For standalone microservice deployments, launch the XLA Serving Engine
Docker container on gRPC port 8500:

```bash
docker run -p 8500:8500 \
    -v /path/to/aot_model:/models/my_model \
    gcr.io/neptune-serving-engine/gpu-jax-cuda12:latest \
    --model_path=/models/my_model \
    --model_name=my_model \
    --orbax_grpc_server_platform=cuda \
    --port=8500
```

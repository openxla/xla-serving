# XLA SavedModel CLI (`xsm-cli`)

`xsm-cli` is a command-line tool for inspecting, debugging, and exporting
artifacts from XLA SavedModels. It is installed automatically along with the
`xla_saved_model` package.

## Commands & Common Use Cases

### 1. Show Model Summary

Displays high-level manifest metadata, objects (functions, values),
supplementals, and device assignments:

```bash
xsm-cli show /path/to/model
```

### 2. Inspect Details of Objects and Pipelines

Use `--details` (`-d`) to inspect specific functions, external values, or
supplementals (supports comma-separated names or multiple flags).

The `serving_default` below is a default function name for many XLA SavedModel objects
but it can be changed during the export, examine your model with
`xsm-cli show /path/to/model` first to learn what objects it exports.

```bash
# Inspect a specific function (signatures, lowering platforms, weights)
xsm-cli show /path/to/model --details serving_default

# Inspect orchestration pipeline (inputs/outputs, stage components, etc.)
xsm-cli show /path/to/model --details orchestration

# Inspect multiple objects
xsm-cli show /path/to/model --details serving_default,orchestration
```

Add `--verbose` (`-v`) to expand full tensor shapes, signatures, and weight
names:

```bash
xsm-cli show /path/to/model --details serving_default -v
```

### 3. Dump StableHLO MLIR to File

Extract and deserialize the StableHLO bytecode embedded in the manifest to an
MLIR text file:

```bash
xsm-cli show /path/to/model --details serving_default --savehlo /tmp/model.mlir
```

### 4. Show XLA Compile Options

Inspect platform-specific XLA compilation options (e.g., TPU, CPU, CUDA)
configured for a function:

```bash
xsm-cli show /path/to/model --details serving_default --xla_compile_options
```

### 5. Check CLI Version

```bash
xsm-cli version
```

### 6. Recompile Models Ahead-Of-Time (AOT)

Precompile model functions for a target platform ahead of serving time:

```bash
xsm-cli recompile \
    --input-model-path=/path/to/exported_model \
    --output-model-path=/path/to/aot_model \
    --platform=cpu \
    --target-topology-label="cpu_target"

# Optional: Pass --strip-stablehlo to minimize artifact size.
```

See the [AOT User Guide](../../docs/aot_user_guide.md) for full details on
multi-platform recompilation, Docker serving, and autotune databases.

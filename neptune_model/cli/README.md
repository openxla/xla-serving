# Neptune Model CLI (`npt-cli`)

`npt-cli` is a command-line tool for inspecting, debugging, and exporting
artifacts from Neptune Models. It is installed automatically along with the
`neptune_model` package.

## Commands & Common Use Cases

### 1. Show Model Summary

Displays high-level manifest metadata, objects (functions, values),
supplementals, and device assignments:

```bash
npt-cli show /path/to/model
```

### 2. Inspect Details of Objects and Pipelines

Use `--details` (`-d`) to inspect specific functions, external values, or
supplementals (supports comma-separated names or multiple flags).

The `serving_default` below is a default function name for many Neptune objects
but it can be changed during the export, examine your model with
`npt-cli show /path/to/model` first to learn what objects it exports.

```bash
# Inspect a specific function (signatures, lowering platforms, weights)
npt-cli show /path/to/model --details serving_default

# Inspect orchestration pipeline (inputs/outputs, stage components, etc.)
npt-cli show /path/to/model --details orchestration

# Inspect multiple objects
npt-cli show /path/to/model --details serving_default,orchestration
```

Add `--verbose` (`-v`) to expand full tensor shapes, signatures, and weight
names:

```bash
npt-cli show /path/to/model --details serving_default -v
```

### 3. Dump StableHLO MLIR to File

Extract and deserialize the StableHLO bytecode embedded in the manifest to an
MLIR text file:

```bash
npt-cli show /path/to/model --details serving_default --savehlo /tmp/model.mlir
```

### 4. Show XLA Compile Options

Inspect platform-specific XLA compilation options (e.g., TPU, CPU, CUDA)
configured for a function:

```bash
npt-cli show /path/to/model --details serving_default --xla_compile_options
```

### 5. Check CLI Version

```bash
npt-cli version
```

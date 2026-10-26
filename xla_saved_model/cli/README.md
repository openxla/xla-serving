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
xsm-cli show /path/to/model \
  --details serving_default

# Inspect orchestration pipeline (inputs/outputs, stage components, etc.)
xsm-cli show /path/to/model \
  --details orchestration

# Inspect multiple objects
xsm-cli show /path/to/model \
  --details serving_default,orchestration
```

Add `--verbose` (`-v`) to expand full tensor shapes, signatures, and weight
names:

```bash
xsm-cli show /path/to/model \
  --details serving_default \
  --verbose
```

### 3. Dump StableHLO MLIR to File

Extract and deserialize the StableHLO bytecode embedded in the manifest to an
MLIR text file:

```bash
xsm-cli show /path/to/model \
  --details serving_default \
  --savehlo /tmp/model.mlir
```

### 4. Show XLA Compile Options

Inspect platform-specific XLA compilation options (e.g., TPU, CPU, CUDA)
configured for a function:

```bash
xsm-cli show /path/to/model \
  --details serving_default \
  --xla_compile_options
```

### 5. Update, Inspect History, and Rollback Pipeline Configurations

Update batching configuration options in an exported model's orchestration
pipelines:

```bash
# Update maximum batch size in-place (backs up the modified config proto to .orig and .bak)
xsm-cli update /path/to/model \
  --max_batch_size 32

# Update multiple batching parameters and target a specific pipeline
xsm-cli update /path/to/model \
    --pipeline_name serving_default \
    --max_batch_size 64 \
    --batch_timeout_micros 5000 \
    --allowed_batch_sizes 8,16,32,64 \
    --num_batch_threads 4 \
    --max_enqueued_batches 16 \
    --disable_large_batch_splitting \
    --pad_variable_length_inputs

# Clone and update to a new directory (leaves original model untouched)
xsm-cli update /path/to/model \
  --max_batch_size 64 \
  --output_dir /path/to/updated_model

# View the model's modification audit log (history.txtpb)
xsm-cli history /path/to/model

# Undo the most recent in-place update (restores config proto from .bak)
xsm-cli rollback /path/to/model \
  --to=bak

# Reset configuration to the original exported state (restores config proto from .orig)
xsm-cli rollback /path/to/model \
  --to=orig
```

In-place updates modify the configuration protobuf atomically and create
lightweight binary protobuf backups alongside the modified file (e.g.,
`orchestration.pb.orig` for the initial exported state and
`orchestration.pb.bak` for the immediately preceding state, rather than copying
the entire model directory). Only a single previous state (`.bak`) and the
original state (`.orig`) are maintained, so running the same `rollback` command
repeatedly when the configuration already matches the backup is a no-op. Each
export, update, and rollback is also appended as a `HistoryRecord` entry in the
model's `history.txtpb` textproto audit log.

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

See the [AOT guide](../../docs/guides/aot.md) for full details on
multi-platform recompilation, Docker serving, and autotune databases.

### 7. Check CLI Version

```bash
xsm-cli version
```


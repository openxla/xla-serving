# xsm-cli

`xsm-cli` is the command-line companion to `xla_saved_model`, installed with
the wheel. It reads and edits bundles without loading them onto a device, so
it works anywhere Python runs.

| Command | What it does |
|---|---|
| `show` | Print the manifest, a function's signature, or the pipelines |
| `update` | Change batching options in place or into a copy |
| `history` | Print the bundle's modification log |
| `rollback` | Undo the last `update`, or all of them |
| `recompile` | Compile ahead of time for a platform ([own page](aot.md)) |
| `version` | Print the CLI's version string |

## show

Prints the manifest: every function and value, every supplemental, and the
pipelines.

```bash
xsm-cli show /tmp/classifier
```

```text
Manifest: 7 KiB, 2 objects, 1 supplementals
┏━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┓
┃ Object          ┃ Type                ┃
┡━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━┩
│ serving_default │ MLIR StableHLO v1.0 │
│                 │ inlined: 3861 bytes │
│ weights         │ Orbax Checkpoint    │
│                 │ file: checkpoint    │
└─────────────────┴─────────────────────┘
┏━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┓
┃ Supplemental  ┃ Type                           ┃
┡━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┩
│ orchestration │ Orchestration Pipelines v0.0.1 │
│               │ file: orchestration.pb         │
└───────────────┴────────────────────────────────┘
```

`--details` (`-d`) drills into one or more objects by name. Names are the
ones in the first column, plus `orchestration` for the pipelines:

```bash
# Signature, platforms, weights
xsm-cli show /tmp/classifier \
  --details serving_default

# Pipelines, roles, batch options
xsm-cli show /tmp/classifier \
  --details orchestration

# Inspect multiple objects
xsm-cli show /tmp/classifier \
  --details serving_default,orchestration
```

For a function, the details include the input and output signature with
dtypes, the lowering platforms, the calling-convention version, and a count
of weight tensors. Add `--verbose` (`-v`) to list the weights by name and
shape. For a C++ `CompositeProcessor` stage the details are its transforms,
their edges, and its assets (this part needs the `xla-serving-data` wheel
installed; otherwise the CLI prints a hint and moves on).

Two more flags pull specific things out of a function:

```bash
# Deserialize the StableHLO bytecode to MLIR text.
xsm-cli show /tmp/classifier \
  --details serving_default \
  --savehlo /tmp/classifier.mlir

# Print the XLA compile options recorded at export, per platform.
xsm-cli show /tmp/classifier \
  --details serving_default \
  --xla_compile_options
```

The MLIR dump is the thing to attach to a bug report about a compile
failure. It is also a quick way to check what `jax.export` actually
captured, for instance whether a custom call survived.

## update

Edits batching options in `orchestration.pb`.

```bash
# In place; keeps .orig and .bak copies of the file.
xsm-cli update /tmp/classifier \
  --max_batch_size 64

# Into a copy, leaving the original untouched.
xsm-cli update /tmp/classifier \
  --max_batch_size 64 \
  --output_dir /tmp/classifier_b64
```

The in-place form writes the new proto atomically and keeps two backups next
to it: `orchestration.pb.orig`, the state at export, written once and never
touched again; and `orchestration.pb.bak`, the state before the most recent
update. Only those two exist, so there is a one-step undo and a full reset,
nothing in between. If the pipelines are inlined in `manifest.pb` rather than
split out, the backups are of `manifest.pb` instead.

`max_batch_size` is the only field the CLI edits today. Other
`BatchOptions` fields can be changed with a few lines of Python against
`xla_saved_model.protos.orchestration_pb2.Pipelines`; if you do that, append
your own record to `history.txtpb` so the next person can see what changed.

## history

Every export, `update`, and `rollback` appends a record to `history.txtpb` in
the bundle. `history` prints them as a table:

```bash
xsm-cli history /tmp/classifier
```

```text
                  Model Modification History (version: 0.0.1)
┏━━━┳━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━┓
┃ # ┃ Timestamp       ┃ User  ┃ Action   ┃ Description      ┃ Changes         ┃
┡━━━╇━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━┩
│ 1 │ 2026-10-15T20:… │ alice │ EXPORT   │ Initial model    │ -               │
│   │                 │       │          │ export.          │                 │
│ 2 │ 2026-10-15T20:… │ alice │ UPDATE   │ Updated          │ serving_defaul… │
│   │                 │       │          │ max_batch_size   │ 0 -> 32         │
│   │                 │       │          │ via CLI.         │                 │
│ 3 │ 2026-10-15T20:… │ alice │ UPDATE   │ Updated          │ serving_defaul… │
│   │                 │       │          │ max_batch_size   │ 32 -> 64        │
│   │                 │       │          │ via CLI.         │                 │
│ 4 │ 2026-10-15T20:… │ alice │ ROLLBACK │ Rolled back      │ serving_defaul… │
│   │                 │       │          │ model            │ 64 -> 32        │
│   │                 │       │          │ configuration    │                 │
│   │                 │       │          │ from .bak        │                 │
│   │                 │       │          │ backup.          │                 │
└───┴─────────────────┴───────┴──────────┴──────────────────┴─────────────────┘
```

The file itself is a text proto and is fine to read directly when the table
is too narrow, or to append to from a script; the CLI is just a view of it:

```text
records {
  action: "UPDATE"
  timestamp: "2026-10-15T20:48:41.159924+00:00"
  user: "alice"
  description: "Updated max_batch_size via CLI."
  changes {
    key: "serving_default.max_batch_size"
    value { previous_value: "0" new_value: "32" }
  }
}
```

## rollback

Restores the configuration proto from one of the backups that `update`
keeps:

```bash
# Undo the last update (restores from .bak)
xsm-cli rollback /tmp/classifier \
  --to=bak

# Reset to the state at export (restores from .orig)
xsm-cli rollback /tmp/classifier \
  --to=orig
```

```text
                   Model Rollback Applied (.bak)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┓
┃ Target / Parameter             ┃ Previous Value ┃ Restored Value ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━┩
│ serving_default.max_batch_size │ 64             │ 32             │
└────────────────────────────────┴────────────────┴────────────────┘
Successfully rolled back model from .bak backup at `/tmp/classifier`
```

A rollback appends its own `ROLLBACK` record to `history.txtpb`. Running it
again when the configuration already matches the backup prints
`nothing to rollback` and adds no record. There is no rollback for
`--output_dir` copies; delete the copy.

## recompile

Compiles a bundle ahead of time for a platform and topology. It has its own
page: [Ahead-of-time compilation](aot.md).

```bash
xsm-cli recompile \
  --input-model-path=/tmp/classifier \
  --output-model-path=/tmp/classifier_aot \
  --platform=cpu \
  --target-topology-label=cpu_target
```

## version

```bash
$ xsm-cli version
XSM CLI version: 0.0.1
```

This is the CLI's own version string, not the package's. For the package,
use `pip show xla-saved-model`. Bundles record the format version separately
in `orbax_model_version.txt`; when a server rejects a bundle, compare that
file with the server's startup log.

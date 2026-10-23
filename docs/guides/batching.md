# Batching

Accelerators are efficient when they process many examples per program
launch, and gRPC clients send one example at a time. The server bridges the
two with a batch scheduler: requests arriving within a short window are
concatenated along the batch dimension, run as one call, and the results are
split back out. Clients never know.

Batching is configured in the bundle, not on the server. You declare it in
`ServingConfig.batch_options` at export, and you can change the numbers
afterwards with `xsm-cli update` without re-exporting. A server loads whatever
the bundle says.

## Turning it on

```python
import xla_saved_model
from xla_saved_model.protos import orchestration_pb2

batch_options = orchestration_pb2.BatchOptions(
    batch_component=orchestration_pb2.BatchOptions.MODEL_FUNCTION,
    allowed_batch_sizes=[8, 32],
    batch_timeout_micros=2000,
    num_batch_threads=2,
    max_enqueued_batches=16,
)

xla_saved_model.save(
    jax_model,
    export_path='/tmp/classifier',
    serving_configs=[
        xla_saved_model.ServingConfig(
            input_signature=input_signature,
            batch_options=batch_options,
        )
    ],
)
```

With this, a request carrying 5 rows is padded to 8 and run; a burst of
twenty single-row requests arriving within 2 ms becomes one batch of 32 with
12 rows of padding. A request carrying more rows than `max_batch_size` is
rejected with `INVALID_ARGUMENT: Task size 40 is larger than maximum input
batch size 32`, so size your client batches accordingly.

`xsm-cli show <dir> --details orchestration` prints the options a bundle
carries.

## The fields that matter

`batch_component`
: Which stages of the pipeline run inside the batch. `MODEL_FUNCTION` batches
  only the accelerator call and runs pre- and post-processing per request.
  `WHOLE_PIPELINE` batches everything. `PRE_PROCESSOR_AND_MODEL_FUNCTION` and
  `MODEL_FUNCTION_AND_POST_PROCESSOR` are the two halves. Setting this is
  what enables batching; `NO_BATCHING` or leaving it unset disables it.

`allowed_batch_sizes` and `max_batch_size`
: The shapes the model will actually be run with. Batches are padded up to
  the next allowed size, so each entry is one compiled executable on the
  device. A short list like `[8, 32, 128]` is typical. If you give
  `allowed_batch_sizes`, `max_batch_size` defaults to the last entry; if you
  omit it, every size up to `max_batch_size` is allowed and you will compile
  a new executable the first time each size appears.

`batch_timeout_micros`
: How long the scheduler waits for more requests before running an
  incomplete batch. This is a direct trade of tail latency for throughput.
  Values from a few hundred microseconds to a few milliseconds are common;
  zero means "run whatever is queued as soon as a thread is free".

`num_batch_threads`
: How many batches can be in flight at once. Match it to the number of
  device cores you want busy. Zero is treated as one.

`max_enqueued_batches`
: Back-pressure. Once this many batches are waiting, new requests fail fast
  instead of queueing. Set it low enough that a stalled device turns into
  errors your load balancer can see, rather than a growing queue.

`batch_padding_policy`
: What to do with 18 requests when the allowed sizes are 16 and 32. `PAD_UP`
  (the default) runs a batch of 32. `BATCH_DOWN` runs 16 now and leaves 2
  waiting. `MINIMIZE_TPU_COST_PER_REQUEST` picks whichever gives the lower
  device cost per real request, using per-size costs it measures at runtime.

`pad_variable_length_inputs`
: Pads a non-batch dimension, such as sequence length, to the longest in the
  batch so that ragged requests can share a batch. Requires
  `disable_large_batch_splitting=True`.

The remaining fields (`low_priority_batch_options`,
`priority_batching_policy`, `mixed_priority_batching_policy`,
`scheduler_config`) implement two-tier priority queues for mixing latency
sensitive and sheddable traffic on one model. They are documented in
`protos/orchestration.proto` and are not needed to get started.

## Which stages to batch

Host-side stages are not free to batch. A TF tokenizer batched across 32
requests has to pad every string to the longest one; a per-request
tokenizer does not. On the other hand, a C++ bucketizer that is cheap per
row benefits from running once per batch. The rule of thumb:

- Start with `MODEL_FUNCTION`. The accelerator call is what you are
  amortizing.
- Move to `PRE_PROCESSOR_AND_MODEL_FUNCTION` when the preprocessor is cheap
  per row and its output is what you would otherwise be padding anyway
  (embedding lookups, feature crosses).
- Use `WHOLE_PIPELINE` for pipelines that are shape-stable end to end, like
  the LLM exporter's tokenizer–model–detokenizer chain with a fixed prompt
  length.

Batched stages have to be adjacent in the pipeline. The exporter checks
this and refuses an orchestration where a non-batched call sits between two
batched ones.

## Batching and symbolic shapes

A function exported with a symbolic batch dimension compiles lazily, once
per distinct batch size the scheduler produces. With `allowed_batch_sizes`
set, that is a handful of compilations and then steady state.

If you also set `input_polymorphic_shape_symbol_values={'b': [8, 32]}` on
the `ApplyFnBundle`, the exporter emits one concrete StableHLO module per
value instead of one polymorphic module. Keep the two lists identical. You
want this when you plan to [compile ahead of time](aot.md), since the AOT
compiler needs concrete shapes, and it costs nothing otherwise. Without
batching, a concrete-shape export rejects any request whose batch size is
not in the list.

## Changing the numbers later

`xsm-cli update` rewrites `orchestration.pb` in place or into a copy, and
records the change in `history.txtpb`:

```bash
xsm-cli update /tmp/classifier \
  --max_batch_size 64

xsm-cli update /tmp/classifier \
  --max_batch_size 64 \
  --output_dir /tmp/classifier_b64
```

The in-place form keeps `.orig` and `.bak` copies next to the file, and
`xsm-cli rollback` restores either one; `xsm-cli history` shows what was
changed and when. Today the CLI exposes `max_batch_size`; for other fields,
edit the proto with a short Python script against
`orchestration_pb2.Pipelines`. Details in [xsm-cli](xsm_cli.md).

## Observing it

The server logs the batch options it loaded at startup:

```text
interpreter.cc:446] Batch options:
batch_component: MODEL_FUNCTION
max_batch_size: 32
batch_timeout_micros: 2000
allowed_batch_sizes: 8
allowed_batch_sizes: 32
```

The server also runs a profiler endpoint on `--xprof_port` (default 8888).
Capturing a trace with [XProf](https://github.com/openxla/xprof) while
under load shows the executable launches and their shapes, which is the
most direct way to see whether `batch_timeout_micros` is filling batches or
just adding latency.

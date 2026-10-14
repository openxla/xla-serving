# XSE vs. pure-Python JAX serving: DLRM-DCNv2 benchmark

A reproducible, single-node, apples-to-apples comparison of
[XLA Serving Engine (XSE)](../../README.md) against a hand-written
pure-Python JAX gRPC server, on the same GCP GPU or TPU VM, serving the same
model with the same weights — and, with the `xse-tf-*` arms, of XSE's
`CompositeProcessor` CPU preprocessor against a production-grade
**TensorFlow** preprocessor serving the same model in the same XSE binary (see
["TF preprocessor vs. CompositeProcessor"](#tf-preprocessor-vs-compositeprocessor)).

The model is the DLRM-DCNv2 from [`examples/dlrm`](../dlrm): 26 categorical
features looked up in CPU-resident embedding tables (≈6.2M rows at
`embedding_size=128`, ≈3.2 GB) plus 13 dense features, feeding a DCNv2 tower
that runs on the accelerator. In the default arms CPU preprocessing uses XSE's
`CompositeProcessor` / JAX CPU Embedding (JCE) path — **no TensorFlow**; the
optional `xse-tf-*` arms replace only that preprocessor with a TF SavedModel
function.

## What is compared

| Arm | Server | Preprocessing | Batching |
|---|---|---|---|
| `xse-batch` | `xla_serving_main_{cpu,gpu,tpu}` (C++) | JCE in-process | server-side, `PRE_PROCESSOR_AND_MODEL_FUNCTION`, pad-up to the power-of-two buckets `1..4096` |
| `xse-nobatch` | same binary, second export | JCE in-process | `NO_BATCHING` (request batch must be an exported bucket) |
| `xse-tf-batch` | same binary (built with TF support), export from [`export_tf_benchmark_model.py`](export_tf_benchmark_model.py) | TF SavedModel function in-process (TFRT): `tf.sparse.segment_sum` per table = one fused `SparseSegmentSum` kernel, the `embedding_lookup_sparse` fast path | as `xse-batch` |
| `xse-tf-nobatch` | same | same | `NO_BATCHING` |
| `xse-tf-gather-batch` | same | TF: `tf.gather` → `[B, V, D]` → `reduce_sum` (`ResourceGather` + `Sum`; what `TPUEmbeddingForServing` / dense `tf.nn.embedding_lookup` emit) | as `xse-batch` |
| `xse-tf-gather-nobatch` | same | same | `NO_BATCHING` |
| `py` | [`python_server.py`](python_server.py): `grpc.server` + numpy lookup + `jax.jit` | numpy `take`+`sum` | none (one `jit` call per request, embeddings padded to the bucket) |
| `py-batch` | same + [`DynamicBatcher`](python_server.py) | numpy | client requests coalesced, same buckets / timeout / batch threads as XSE |
| `py-mp` | `python_server.py --num_processes=N` ([`multiprocess_server.py`](multiprocess_server.py)): N gRPC frontends on one `SO_REUSEPORT` port + 1 device-owner process | numpy, in the frontends | none (one `jit` call per request in the owner) |
| `py-mp-batch` | same + a `DynamicBatcher` in **each frontend**, in front of its gather | numpy, in the frontends, once per batch | client requests coalesced per frontend (same buckets / timeout / queue bound as XSE, batch threads and queue budget split over the frontends); the owner additionally merges frontend batches that are already waiting for the device, without waiting for more |

The default `--arms` are the two JCE and the four Python arms; the `xse-tf-*`
arms are opt-in because they need a TF-enabled server (below).

Everything that is *not* the serving runtime is held constant:

* identical weights (generated once, saved as `npz` for Python and baked into
  the XSE export) — [`parity_check.py`](parity_check.py) asserts
  `max|Δ| ≤ 1e-4` against an in-process JAX reference before any load is run;
* identical wire protocol (`PredictionService/Predict`, `HostBuffer` tensors)
  and identical pre-serialised requests from the same
  [`loadgen.py`](loadgen.py). The proto package in front of the service name
  is `jsv.serving` in  and `xla_serving.serving` in the OSS export;
  [`common.py`](common.py) takes it from the installed proto descriptor so the
  same tools work against either build;
* identical allowed batch sizes, batch timeout, batch threads and batch queue
  bound (see "Batching configuration" below); identical *execution* thread
  count (`--num_threads` → XSE `--thread_pool_num_threads` / Python handler
  threads). In `py-batch` and `py-mp-batch` the gRPC handler threads only
  parse, validate and park on a future while `--num_batch_threads` compute,
  so those arms' handler pools are widened to `max(--num_threads, 512)`
  (split over the frontends for `py-mp-batch`) — otherwise they could never
  batch more than `--num_threads` requests, which XSE's scheduler (fed by
  async gRPC) is not limited by;
* identical batch *placement*: all three batched arms coalesce requests
  **before** the embedding lookup (XSE `PRE_PROCESSOR_AND_MODEL_FUNCTION`,
  one JCE / numpy gather per batch), so the comparison measures the runtimes,
  not where the batch boundary sits;
* identical client connection count: every load generator worker opens
  `--channels_per_worker` gRPC channels (default 8, so 64 connections with
  the default 8 workers) against *every* arm. `py-mp` needs it because
  `SO_REUSEPORT` balances connections, not requests; the other arms get the
  same so the client side is identical;
* identical CPU pinning (`--server_cpus` / `--client_cpus`), one accelerator,
  one *arm* at a time;
* identical JAX / XLA version on the VM (the Python arm uses whatever
  `pip install jax[...]` gives; XSE lowers through the same `jax.export`).

Reported per arm: peak throughput (examples/s), the closed-loop
concurrency sweep (req/s, p50/p99), open-loop Poisson latency at 50 % and
80 % of *that arm's* peak, server CPU-seconds per 1k examples, peak memory
(PSS summed over the server's processes, so `py-mp`'s N table copies are
counted once each and shared pages are not double-counted) and GPU
utilisation.

### Batching configuration

All batching arms (`xse-batch`, `py-batch`, `py-mp-batch`) share one
configuration, written into the export by `export_benchmark_model.py` and
passed to the Python servers on the command line:

* **Buckets** (`--batch_sizes`, default `1,2,4,…,4096`): the exported
  `input_polymorphic_shape_symbol_values`, XSE's `allowed_batch_sizes` with
  `PAD_UP`, and the shapes the Python arms `jit` and pad to. Powers of two
  bound the padding waste at 2×. The earlier default `{1,16,128,512,4096}`
  did not: a request batch of 128 plus one more coalesced request padded up to
  512, which made the batched arms *slower* at `B=128` than without batching.
  Every arm pads to the same buckets, so the choice is neutral; override with
  `--batch_sizes` to sweep it (the `xse-nobatch` arm then also only accepts
  request batch sizes from that list).
* **Batch threads** (`--num_batch_threads`, default `= --num_threads`): XSE's
  `num_batch_threads` and the Python `DynamicBatcher`'s worker count. The
  earlier default of 2 capped `xse-batch` at two concurrent model executions
  regardless of `--num_threads`. `py-mp-batch` splits them over its
  frontends (at least one each) and gives the owner's merger the full count.
* **Timeout** (`--batch_timeout_micros`, default 1000).
* **Queue bound / admission control** (`--max_enqueued_batches`, default 64):
  XSE's `max_enqueued_batches` — the batch scheduler rejects a request with
  `UNAVAILABLE` ("The batch scheduling queue … is full") once that many
  batches are waiting — and the Python batchers' equivalent (more than
  `max_enqueued_batches × max_batch_size` examples queued ⇒ `UNAVAILABLE`
  with the same wording, before any lookup work). `py-mp-batch` splits the
  budget over its frontends so the total matches. With the default 64 × 4096
  examples it does not fire in the standard sweep — a Python batcher's queue
  can never hold more than one request per parked handler thread
  (`--handler_threads × B` examples, 512 × 16 = 8 192 at `B=16`), so the
  default bound is only reachable at request batches of several hundred;
  below that an overloaded Python arm is shed earlier by gRPC core
  (`CANCELLED`, see Troubleshooting). It exists so that the batched Python
  arms fail fast the way XSE does instead of queueing without bound, and so
  that a sweep can lower it on purpose (e.g. `--max_enqueued_batches 1`
  turns the overload points into `UNAVAILABLE` whose count equals the
  servers' `rejected` counter). The unbatched arms (`xse-nobatch`, `py`,
  `py-mp`) have no such bound on either side.

### Multi-process Python (`py-mp`, `py-mp-batch`)

The honest answer to "the single-process Python server is GIL-bound at ~2
cores; would N processes behind one port close the gap?". Constraints that
shape the design:

* **One accelerator, one process.** A TPU chip can be opened by exactly one
  process (libtpu holds a lock) and a second JAX process on a GPU competes
  for its memory, so the model runs in a single *owner* process. Everything
  that is GIL-bound — gRPC parsing/serialisation and the 26-table numpy
  gather — runs in N *frontend* processes.
* **"One acceptor" means `SO_REUSEPORT`.** The frontends bind the same port;
  the kernel hashes each *connection* to one of them. Per-frontend request
  counts are printed at shutdown (`Frontend i served R requests / E
  examples`) so an imbalance is visible; with few connections (closed loop at
  `C=1`) the imbalance is inherent to the design and part of the result.
* **Re-exec, not fork.** gRPC and JAX threads make `fork()` unsafe, so the
  owner re-executes its own binary with `--frontend_id=i`; each frontend
  loads its **own private copy of the embedding tables** (`N × 3.2 GB` at
  full size — `run_benchmark.sh` checks `MemAvailable` and warns). A shared
  read-only `mmap` of `tables.npz` would avoid the copies but would make the
  frontends' hottest loop run on page-cache-backed memory instead of the
  anonymous (THP-eligible) arrays the single-process arm uses; the private
  copies keep the two Python designs comparable and their cost is reported
  as PSS.
* **IPC.** Each frontend owns a shared-memory ring (`--ring_mb`, default 256,
  in `/dev/shm` or `--ipc_dir`) into which it gathers the looked-up
  embeddings and dense features; it always reserves and zeroes a whole
  bucket so the owner can run the model on the ring rows directly. A 17-byte
  record over a `SOCK_SEQPACKET` UNIX socket hands the rows over; the owner
  writes the predictions back into the ring and acknowledges. No per-request
  serialisation happens between the processes.
* **Batching in the frontends (`py-mp-batch`).** Each frontend runs the same
  `DynamicBatcher` as `py-batch`, *in front of* its gather: handler threads
  parse, validate and park; the frontend's batcher thread(s) do one gather
  and one ring hand-over per batch. The owner keeps a `DynamicBatcher` too,
  but with a zero timeout — it merges the frontend batches that are already
  queued for the device into one launch (a single frontend's batch runs on
  its ring rows in place; several are copied into one padded array) and
  never waits for more. Without it, at request batch 1 each frontend would
  see only 1/N of the traffic, form tiny batches, and the accelerator's
  launch rate rather than its throughput would bind — XSE's single scheduler
  does not have that problem. The earlier design (gather per request in 64
  handler threads per frontend, batch the gathered rows in the owner)
  collapsed under open-loop overload — see Troubleshooting, `CANCELLED`.
  Shutdown prints each frontend's batcher counters, their sum (`Batcher
  stats:`, comparable with `py-batch`) and the owner's merge ratio (`Owner
  merged M frontend batches into K device batches`).
* **Thread accounting.** Each frontend has `--num_threads` gRPC handler
  threads (widened as described above for `py-mp-batch`); the owner runs the
  model on `--num_threads` threads (`py-mp`) or `--num_batch_threads` merger
  threads (`py-mp-batch`, whose frontends get `--num_batch_threads / N`
  batcher threads each, at least one). `--num_processes` defaults to
  `--num_threads` in `run_benchmark.sh`, i.e. one frontend per server core.

### TF preprocessor vs. CompositeProcessor

The question behind the `xse-tf-*` arms: a team with an existing TensorFlow
recommender (embedding lookup in TF, dense tower in JAX or TF) can run the TF
preprocessor **inside XSE** as a TF SavedModel function — does rewriting it to
XSE's `CompositeProcessor` (JCE) buy anything? Both preprocessors are run by
the same XSE binary, in the same process, under the same batch scheduler, with
the same hand-off to the same accelerator model, so the measurement isolates
the lookup implementation.

**What the two preprocessors do.** Both take the request's 26 `int32 [B, V_i]`
id tensors and the dense features (the only input format the benchmark sends)
and hand the DCNv2 model 27 tensors: 26 SUM-pooled `float32 [B, D]`
embeddings and the dense passthrough.

* `xse-batch` / `xse-nobatch` — JCE: one `CpuEmbeddingModule` call per
  request/batch, a C++ grouped lookup over the 26 tables on an Eigen pool of
  `max_cpu_slots` threads, writing the pooled rows straight into the output
  `HostBuffer`s.
* `xse-tf-batch` / `xse-tf-nobatch` — TF *fused* lookup
  ([`export_tf_benchmark_model.py`](export_tf_benchmark_model.py),
  `--tf_lookup=fused`, the default): for every multi-valent feature
  `tf.sparse.segment_sum(table, ids, segment_ids)` — a single fused
  `SparseSegmentSum[WithNumSegments]` kernel that reads the `V_i` rows of each
  example and accumulates them into the `[B, D]` output with no intermediate.
  This is the op graph `tf.nn.embedding_lookup_sparse(..., combiner='sum',
  allow_fast_lookup=True)` / `safe_embedding_lookup_sparse` — and hence
  `tf.feature_column.embedding_column` and Keras' sparse-input embedding
  layers — emit, minus the `SparseTensor` plumbing that dense fixed-valency
  ids do not need. Single-valent features are a plain `ResourceGather`.
* `xse-tf-gather-batch` / `xse-tf-gather-nobatch` — TF *gather* lookup
  (`--tf_lookup=gather`): `tf.reduce_sum(tf.gather(table, ids), axis=1)`,
  i.e. `ResourceGather` to `[B, V_i, D]` then `Sum`. This is what
  `tf.nn.embedding_lookup` on a dense id tensor,
  `tf.tpu.experimental.embedding.TPUEmbeddingForServing` (the layer used by
  [`examples/dlrm/export.py`](../dlrm/export.py)) and Keras' dense-input
  `Embedding` layer produce — the most common shape of an existing TF
  recommender export. It materialises up to `4096 × 100 × 128 × 4 B ≈ 210 MB`
  for the valency-100 feature at the largest bucket before reducing it.

Both TF variants exist so that the TF side cannot be called a straw man:
`fused` is the most efficient lookup TF offers for this input (minimal memory
traffic, one kernel per table); `gather` is what most production TF exports
actually contain and parallelises each gather across the intra-op pool, so it
can win single-request latency at large batches while losing under
concurrency. Report both. The TF SavedModel is served by TFRT with XSE's
production defaults (MLRT on, invariant `ReadVariableOp`s hoisted out of the
per-request graph), 26 non-trainable `tf.Variable` tables restored into the
TFRT resource manager at load time, and `HostBuffer` ↔ `tf.Tensor` conversion
on the way in and out is zero-copy.

**Held constant across the JCE and TF arms** (in addition to everything in
"What is compared"):

* *Weights*: `export_tf_benchmark_model.py` loads the `tables.npz` /
  `params.npz` the JCE exporter wrote, so the arms serve bit-identical tables
  and parameters and `parity_check.py` runs unchanged against every arm.
* *Server binary*: one binary per run. In  the script builds the CPU
  server with `--define=ENABLE_TF=1` whenever an `xse-tf-*` arm is selected;
  `--define` does not change the output directory, so the JCE arms of that
  run execute on the same TF-enabled binary. With `--xse_bin` / `--xse_image`
  the same server serves all XSE arms by construction.
* *Batching*: the TF exporter reuses the JCE exporter's `BatchOptions` builder
  (`PRE_PROCESSOR_AND_MODEL_FUNCTION`, same buckets / timeout / batch threads
  / queue bound), so the TF function sees exactly the batches JCE sees; the
  unbatched arms are both `NO_BATCHING`. Warm-up requests are identical too.
* *Hand-off to the accelerator*: both preprocessors emit the same 27 tensors,
  so the DCNv2 `JaxModel` (identical `jax.export` lowering) sees identical
  inputs and the same number of host-to-device copies.
* *Preprocessor parallelism*: `--preprocessor_threads N` (default 8, which is
  both built-in defaults) sets JCE's lookup pool (`CpuEmbeddingResource(
  max_cpu_slots=N)`, baked into the export via `--jce_max_cpu_slots`) **and**
  the TFRT work queue that executes the TF function (server flag
  `--tfrt_num_threads`, added by this benchmark; the script passes it only
  when `N ≠ 8` so that images predating the flag still start). Request
  threads (`--num_threads`) and batch threads are shared with all arms as
  before. The value is recorded in `env.txt` and in every result's labels
  (`preprocessor=jce|tf-fused|tf-gather`, `preprocessor_threads`).

**Reading the result.** `REPORT.md` gains a *CompositeProcessor vs TF*
headline column: each JCE arm divided by its TF counterpart with the same
batching (batched vs. batched, unbatched vs. unbatched), plus the ratio of
server CPU-seconds per 1k examples at the two arms' peaks — since the
preprocessor is the only difference, that ratio is the host-CPU cost of the
TF lookup relative to JCE. Where the accelerator, not the host, is the
bottleneck, throughput will match and the CPU ratio is the number to quote.

**Getting a TF-enabled server.**

* , `--platform cpu`: automatic (`blaze build -c opt
  --define=ENABLE_TF=1 //third_party/neptune/runtime/serving/gcp/server:xla_serving_main_cpu`).
  For GPU/TPU build the matching `xla_serving_main_{gpu,tpu}` with the same
  define and pass `--xse_bin`.
* OSS images: the nightly workflow publishes `cpu-jax-with-tf`,
  `gpu-jax-cuda12-with-tf` and `gpu-jax-cuda13-with-tf` next to the
  `*-jax-only` images — use those tags with `--xse_image` (the script warns
  when the image name does not end in `-with-tf`). **There is no TPU image
  with TF**; for TPU build from source:
  `bazel build -c opt --define=ENABLE_TF=1 //runtime/serving/gcp/server:xla_serving_main_tpu`
  and pass `--xse_bin` (the script refuses `--platform tpu --xse_image` with
  an `xse-tf-*` arm).
* The exporter needs `tensorflow` in the venv (`pip install tensorflow`; it is
  not in `requirements.txt` because the default arms do not need it). The
  exporter hides GPUs from TF and keeps JAX on CPU so that it never touches
  the accelerator the server is about to open.

A JAX-only server fails to load a TF-preprocessor export with
``No `OptionalFunctionHandler` is found for key: …``; `run_benchmark.sh`
checks the binary / image name up front and stops with that hint within
seconds if it happens anyway.

**Caveats.** The TF arm emits 27 separate outputs to mirror JCE; a single
stacked `[B, 26, D]` output (fewer host-to-device copies) would be a different
design for *both* preprocessors and is out of scope. The TF `fused` graph is
hand-written against the dense fixed-valency inputs; a real model fed
`SparseTensor`/`RaggedTensor` ids pays extra for the sparse plumbing on the TF
side only. Each full-size TF export adds ≈3.2 GB of SavedModel checkpoint on
disk (only the variants the selected arms need are exported) and the exporter
holds the tables twice (numpy + `tf.Variable`) while writing.

## Prerequisites

### GPU VM (A100 / L4 / H100)

```bash
gcloud compute instances create xse-bench-gpu \
  --zone=us-central1-a --machine-type=a2-highgpu-1g \
  --accelerator=type=nvidia-tesla-a100,count=1 \
  --image-family=common-cu129-ubuntu-2204-nvidia-580 --image-project=deeplearning-platform-release \
  --boot-disk-size=200GB --maintenance-policy=TERMINATE
# (L4: --machine-type=g2-standard-16 --accelerator=type=nvidia-l4,count=1)
```

The Deep Learning VM image ships the NVIDIA driver (580, which runs both the
CUDA 12 and CUDA 13 server images below), Docker and the NVIDIA container
toolkit. Check
[the current image families](https://cloud.google.com/deep-learning-vm/docs/images)
if the family above has been rotated out.

### TPU VM (v5e / v6e)

```bash
gcloud compute tpus tpu-vm create xse-bench-tpu \
  --zone=us-central1-a --accelerator-type=v5litepod-1 \
  --version=v2-alpha-tpuv5-lite
# (v6e: --accelerator-type=v6e-1 --version=v2-alpha-tpuv6e)
```

or, as a regular Compute Engine instance with a TPU machine type (TPU
quota permitting; use the TPU image family, a stock Ubuntu image needs the
extra setup described below):

```bash
gcloud compute instances create xse-bench-tpu \
  --zone=us-central1-b --machine-type=ct6e-standard-1t \
  --image-family=ubuntu-accel-2204-amd64-tpu-v5e-v5p-v6e \
  --image-project=ubuntu-os-accelerator-images --boot-disk-size=200GB
```

**libtpu.** The OSS TPU server (the `tpu-jax-only` image or an OSS-built
`xla_serving_main_tpu`) does not link the TPU runtime; it `dlopen`s the PJRT
plugin named by `TPU_LIBRARY_PATH` at startup and aborts before `main()` if
the variable is unset. The image is distroless and contains **no
`libtpu.so`**, so one has to come from the host. `run_benchmark.sh` takes
the first of `--tpu_library_path`, `$TPU_LIBRARY_PATH`, the venv's `libtpu`
package (installed by `jax[tpu]` below) and `/lib/libtpu.so`, bind-mounts it
read-only into the container (or sets the variable for a binary) and refuses
to start, before the export, if none is found. The venv's copy is the right
default: it keeps the XSE arm and the Python arm on the same TPU runtime, and
PJRT's forward compatibility lets a nightly image load it.

**TPU runtime environment.** libtpu reads its topology from the
`TPU_CHIPS_PER_HOST_BOUNDS` / `TPU_HOST_BOUNDS` / `TPU_ACCELERATOR_TYPE` /
`TPU_WORKER_ID` variables and, when they are unset, from the instance
metadata attribute `tpu-env`. Only VMs created by the Cloud TPU API
(`gcloud compute tpus tpu-vm create`) have that attribute. On a VM created
with `gcloud compute instances create` the lookup returns 404 and libtpu
retries it **30 × 1 s per variable**: the server is silent for minutes and
eventually logs `Failed to convert value for key 'CHIPS_PER_HOST_BOUNDS' in
'tpu-env' to string: ... Failed to fetch URL after 30 tries (http status:
404)`. The fix is `TPU_SKIP_MDS_QUERY=1` plus the single-host topology for
the machine type (the convention Google
[documents](https://cloud.google.com/dataflow/docs/tpu/use-tpus) for TPU
containers):

| machine type | `TPU_ACCELERATOR_TYPE` | `TPU_CHIPS_PER_HOST_BOUNDS` |
|---|---|---|
| `ct6e-standard-1t` / `-4t` / `-8t` | `v6e-1` / `v6e-4` / `v6e-8` | `1,1,1` / `2,2,1` / `2,4,1` |
| `ct5lp-hightpu-1t` / `-4t` / `-8t` | `v5litepod-1` / `-4` / `-8` | `1,1,1` / `2,2,1` / `2,4,1` |
| `ct5p-hightpu-4t` | `v5p-8` | `2,2,1` |

together with `TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_HOSTNAMES=localhost
TPU_WORKER_ID=0`. `run_benchmark.sh` probes the metadata server and, when
`tpu-env` is missing, derives this set from the machine type for any variable
you have not exported yourself (an unknown machine type is an error with the
variables to set). Because `docker run` does not inherit the shell
environment, it forwards **every exported `TPU_*` variable (except
`TPU_LIBRARY_PATH`) and `LIBTPU_INIT_ARGS`** into the container as `-e`
flags; the `--xse_bin` and Python arms inherit them as usual, so all arms
see the same runtime configuration, which `results/env.txt` records
(`tpu_env_metadata`, `tpu_env_source`, `tpu_env`).

A **stock (non-TPU) Ubuntu image** additionally needs the chip bound to
`vfio-pci` with `TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/0`, and, since it has no
`tpunetd`, `LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true`
(otherwise `DEADLINE_EXCEEDED: Failed to connect to [::]:8353`). The script
forwards both when they are in the environment but cannot set them up;
put the whole block in `/etc/environment` (or a file you `source`) so that
`jax.devices()`, the Python arm and the container agree:

```bash
export TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
export TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0
export TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/0            # stock Ubuntu only
export LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true  # stock Ubuntu only
```

> **One owner per accelerator.** Only one process can open a TPU, and a second
> JAX process on a GPU VM pre-allocates most of the GPU memory. The exporter
> and `parity_check.py` therefore pin JAX to CPU (`jax_platforms=cpu`), the
> servers pin it to `--device`, `loadgen.py`/`monitor.py` never import JAX,
> and `run_benchmark.sh` runs the arms strictly sequentially. Do not start
> another JAX process (notebook, `python -c "import jax"`) while it runs.

### Software (OSS checkout)

Nothing needs to be built with Bazel on the benchmark VM: the exporter and
the Python arm come from release wheels, the XSE server from a prebuilt image
(next section). This is the same recipe the repository's own
[nightly GPU e2e test](../../.github/workflows/nightly-test-gpu.yml) uses.

```bash
git clone https://github.com/AI-Hypercomputer/xla-serving && cd xla-serving

# Python >= 3.12 is required: the release xla_serving_data wheel (JCE /
# CompositeProcessor extension used by the exporter) is built for 3.12-3.14
# only, while Ubuntu 22.04 DLVM / TPU VM images ship a 3.10 system Python.
curl -LsSf https://astral.sh/uv/install.sh | sh
uv venv --python 3.12 && source .venv/bin/activate

# Release wheels (gh: https://cli.github.com; or download *.whl from the
# GitHub Releases page and scp them to the VM).
gh release download --repo AI-Hypercomputer/xla-serving --pattern '*.whl' --dir wheels
uv pip install wheels/xla_serving_data-*-cp312-*.whl
uv pip install wheels/xla_saved_model-*.whl           # exporter + protos
uv pip install -r examples/dlrm_benchmark/requirements.txt
uv pip install "jax[cuda12]"                        # GPU VM
# uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html   # TPU VM

python3 -c 'import jax; print(jax.devices())'       # must list the GPU / TPU
```

Both `xla_saved_model` (the exporter and the `xla_saved_model.protos.*`
modules the benchmark tools import) and `xla_serving_data` (the JCE kernels)
come from the release wheels, as in CI. **Do not `pip install -e .` on the
benchmark VM**: `setup.py` generates the `protos/` package with
`bazel fetch` + `grpcio-tools`, and an editable install without `bazel` on
`PATH` *succeeds with a warning* and leaves you with a package that has no
`xla_saved_model.protos` at all. If the exporter or the benchmark code needs
something newer than the latest release, build the wheel(s) from the checkout
on a machine with Bazel instead (no accelerator needed) and install them:

```bash
# xla_saved_model: pure Python, seconds (bazel is only used to fetch the XLA protos).
uv build --wheel && uv pip install --force-reinstall dist/xla_saved_model-*.whl
# xla_serving_data: C++ extension, ~30 min.
bazel build -c opt //data:xla_serving_data --@rules_python//python/config_settings:python_version=3.12
uv pip install --force-reinstall bazel-bin/data/xla_serving_data-*.whl
```

> **Never run `python -m examples.dlrm_benchmark.…` from the repo root.**
> `python -m` puts the current directory first on `sys.path`, and the
> checkout's `xla_saved_model/` source tree (which has no generated `protos/`)
> then shadows the installed wheel; every tool fails with
> `ModuleNotFoundError: No module named 'xla_saved_model.protos'`.
> `run_benchmark.sh` therefore runs the tools from `<work_dir>/pysrc`, a
> directory that contains only a symlink to `examples/` (the same trick the
> [nightly CPU e2e test](../../.github/workflows/nightly-test-cpu.yml) uses
> with `/tmp/test_workspace`), and refuses to start if
> `import xla_saved_model.protos.grpc.predict_pb2` does not work from there.

(`xla_saved_model_runner`, described in the top-level
[README](../../README.md#modelrunner), is only needed to run this directory's
unit tests, not the benchmark.)

### XSE server (prebuilt image or binary)

The benchmark model uses `CompositeProcessor` for its CPU preprocessor, so the
server must link `interpreter_composite_processor_support` (it is part of
`XLA_SERVING_CALLBACK_GRPC_SERVER_COMMON_DEPS` in
[`runtime/serving/gcp/server/BUILD`](../../runtime/serving/gcp/server/BUILD)
since this directory was added). Every nightly image built from a `main`
that contains `examples/dlrm_benchmark` has it; **older images — including a
stale `:latest` — fail to load the model.**

**Recommended: the nightly image, date-pinned, via `--xse_image`.** Images are
pushed daily at 04:00 UTC by
[push-nightly-serving-images.yml](../../.github/workflows/push-nightly-serving-images.yml)
as `:nightly` and `:nightly-YYYYMMDD`; pick a date on or after the day after
this directory landed on `main` (`git log -1 --format=%cs -- examples/dlrm_benchmark`):

```bash
gcloud auth configure-docker gcr.io --quiet
# GPU VM: cuda12 to match the venv's jax[cuda12], so both arms use the same
# CUDA major. (gpu-jax-cuda13 + jax[cuda13] is the equivalent pair; CUDA 13
# needs driver >= 580, which the nvidia-580 DLVM provides.)
docker pull gcr.io/neptune-serving-engine/gpu-jax-cuda12:nightly-YYYYMMDD
# TPU VM
docker pull gcr.io/neptune-serving-engine/tpu-jax-only:nightly-YYYYMMDD
```

Use the dated tag rather than `:nightly` so a report is reproducible;
`run_benchmark.sh` additionally records the image ID and registry digest in
`results/env.txt`. The script runs the container with `--net=host`,
`--cpuset-cpus=<--server_cpus>`, `--gpus all` (GPU) or `--privileged` plus a
read-only bind mount of the host's `libtpu.so` at `/usr/lib/libtpu.so` with
`TPU_LIBRARY_PATH` pointing at it and `-e NAME=VALUE` for every other
exported `TPU_*` variable and `LIBTPU_INIT_ARGS` (TPU, see "libtpu" and
"TPU runtime environment" above), and bind-mounts the export at
`/tmp/model` — the same `docker run` lines as in the top-level
[README](../../README.md#using-docker-images). The user running the script
must be able to talk to the Docker daemon (`sudo usermod -aG docker $USER`
on a DLVM, then re-login).

**Before a nightly contains the change** (e.g. benchmarking a pending PR):
get an image without building on the benchmark VM.

* *GitHub Actions:* run **Push Nightly XLA Serving Images** via
  `workflow_dispatch` on the branch, with `variant=gpu-jax-cuda12` (or
  `tpu-jax-only`), `tag=<branch-or-pr>` and `add_latest_tag=false`; it uses
  the `n2-32` runners and the remote Bazel cache. Then
  `--xse_image gcr.io/neptune-serving-engine/gpu-jax-cuda12:<tag>`.
* *Build VM:* an `n2-standard-32` with 128 GB RAM and a 200 GB disk, pushing
  to your own Artifact Registry:

  ```bash
  # GPU
  bazel run -c opt --config=cuda_v12 //runtime/serving/gcp/server:xla_serving_pusher_gpu \
    -- --repository=<region>-docker.pkg.dev/<project>/<repo>/gpu-jax-cuda12 --tag=<tag>
  # TPU
  bazel run -c opt //runtime/serving/gcp/server:xla_serving_pusher_tpu \
    -- --repository=<region>-docker.pkg.dev/<project>/<repo>/tpu-jax-only --tag=<tag>
  ```

  Building the GPU server needs ~80 GB RAM and, without the remote cache,
  several hours; do **not** build it on the (12–16 vCPU) benchmark VM.
* *Binary:* `bazel build -c opt [--config=cuda_v12] //runtime/serving/gcp/server:xla_serving_main_{gpu,tpu,cpu}`
  on the same kind of build VM and copy the binary over; pass it with
  `--xse_bin`. The GPU binary then relies on the host's CUDA libraries being
  on `LD_LIBRARY_PATH`, which the image path avoids. The OSS TPU binary needs
  `TPU_LIBRARY_PATH` just like the image; the script starts it with
  `env TPU_LIBRARY_PATH=<resolved libtpu.so>`.

Inside  everything is built by the script
(`blaze build -c opt //third_party/neptune/examples/dlrm_benchmark:all`); for
GPU/TPU pass the matching `xla_serving_main_{gpu,tpu}` via `--xse_bin`.

## Running

Pin the server and the load generator to disjoint CPU sets that exist on the
VM (`nproc`), and keep the client set large enough not to be the bottleneck
(≈ one third of the cores works well):

```bash
# Activate the venv from "Software (OSS checkout)" first; the script runs the
# tools as `python3 -m examples.dlrm_benchmark.<tool>` from <work_dir>/pysrc
# (see the warning above), not from the repo root.
source .venv/bin/activate

# GPU VM: a2-highgpu-1g (12 vCPUs, 85 GB RAM, 1×A100), full-size tables.
# All six arms; py-mp* start --num_processes (= --num_threads) frontends,
# each with its own 3.2 GB copy of the tables (8 x 3.2 GB here).
examples/dlrm_benchmark/run_benchmark.sh \
  --platform cuda \
  --xse_image gcr.io/neptune-serving-engine/gpu-jax-cuda12:nightly-YYYYMMDD \
  --server_cpus 0-7 --client_cpus 8-11 --num_threads 8 \
  --work_dir /mnt/disks/scratch/dlrm_benchmark
# g2-standard-16 (16 vCPUs, 64 GB, 1×L4): --server_cpus 0-9 --client_cpus 10-15
#   --num_threads 10 --num_processes 8   (10 x 3.2 GB would not fit)

# TPU VM: v5litepod-1 (24 vCPUs, 48 GB). Needs the venv's jax[tpu] (for
# libtpu.so) or --tpu_library_path /path/to/libtpu.so; see "libtpu" under
# Prerequisites. 16 table copies do not fit in 48 GB: cap --num_processes.
examples/dlrm_benchmark/run_benchmark.sh \
  --platform tpu \
  --xse_image gcr.io/neptune-serving-engine/tpu-jax-only:nightly-YYYYMMDD \
  --server_cpus 0-15 --client_cpus 16-23 --num_threads 16 --num_processes 8
# v6e-1 (44 vCPUs, 176 GB): --server_cpus 0-29 --client_cpus 30-43 --num_threads 30

# With a locally built / copied server instead of the image:
#   --xse_bin bazel-bin/runtime/serving/gcp/server/xla_serving_main_{gpu,tpu}

# Anything with a CPU: 5-minute smoke test with tiny tables
# (--xse_image gcr.io/neptune-serving-engine/cpu-jax-only:nightly-YYYYMMDD,
# or the xla_serving_main_cpu binary; built automatically in ).
examples/dlrm_benchmark/run_benchmark.sh --platform cpu \
  --max_vocab_size 1000 --duration 5 --warmup 1 --repeats 1 \
  --concurrencies 1,8 --request_batch_sizes 1,16 \
  --num_threads 2 --num_processes 2

# "CompositeProcessor vs TF" preset (see the section of that name): the two
# JCE arms against the two fused-TF arms, plus the dense-gather TF variant.
# Needs a TF-enabled server: the *-with-tf image on GPU / CPU, a
# --define=ENABLE_TF=1 build (automatic in  for cpu) otherwise.
# `pip install tensorflow` in the venv for the exporter (OSS).
examples/dlrm_benchmark/run_benchmark.sh \
  --platform cuda \
  --xse_image gcr.io/neptune-serving-engine/gpu-jax-cuda12-with-tf:nightly-YYYYMMDD \
  --server_cpus 0-7 --client_cpus 8-11 --num_threads 8 \
  --arms xse-batch,xse-nobatch,xse-tf-batch,xse-tf-nobatch,xse-tf-gather-batch,xse-tf-gather-nobatch \
  --work_dir /mnt/disks/scratch/dlrm_benchmark
# Same on CPU in  (smoke size; the script builds the TF-enabled server):
examples/dlrm_benchmark/run_benchmark.sh --platform cpu \
  --max_vocab_size 1000 --duration 5 --warmup 1 --repeats 1 \
  --concurrencies 1,8 --request_batch_sizes 1,16 --num_threads 2 \
  --arms xse-batch,xse-nobatch,xse-tf-batch,xse-tf-nobatch
# --preprocessor_threads N changes the lookup parallelism of *both*
# preprocessors (JCE pool and TFRT work queue) at once.
```

`--server_cpus` becomes `taskset -c` for a binary and `--cpuset-cpus` for a
container, so the pinning is identical across arms either way. A dry run
(`--dry_run`) prints every command, including the full `docker run` line,
without starting anything. `--arms` selects a subset (e.g.
`--arms xse-batch,py-mp-batch` for the two batched head-to-heads; the old
name `xse` is accepted for `xse-batch`). Before starting a `py-mp*` arm the
script compares `--num_processes × size(tables.npz) + 4 GiB` with
`MemAvailable` and warns when it does not fit — lower `--num_processes` or
drop the arm rather than let the kernel OOM-kill a frontend mid-run.

The script:

1. exports once (`export_benchmark_model.py`): `model/xse_batched`,
   `model/xse_unbatched`, `model/baseline_weights/{config.json,tables.npz,params.npz}`;
   then, for the selected `xse-tf-*` arms only, `export_tf_benchmark_model.py`
   from those weights: `model/xse_tf_{fused,gather}_{batched,unbatched}`;
2. per arm: starts the server pinned to `--server_cpus`, waits for the port,
   runs `parity_check.py`, then the closed-loop sweep
   (`--request_batch_sizes` × `--concurrencies` × `--repeats`), computes that
   arm's peak req/s and runs the open-loop sweep at
   `--open_loop_fractions` of it, with `monitor.py` (pinned to
   `--client_cpus`, so its sampling never runs on the server's cores)
   recording server CPU/memory/GPU over the whole process tree for every
   point; stops the server;
3. renders `results/REPORT.md` with `report.py`.

Each point is one JSON under `results/<arm>/` (plus `.monitor.json`), so runs
can be resumed or merged by re-running `report.py --results_dir`.
`results/env.txt` records host, CPU model, memory, GPU/driver and JAX
versions, the batching / process / channel knobs, and the XSE binary path or
image ID / digest.

### Individual tools

Run them from an import root that contains only `examples/`, never from the
repo root (see the warning in "Software (OSS checkout)"):

```bash
# <work_dir>/pysrc is created by run_benchmark.sh; or make one by hand:
d=$(mktemp -d) && ln -s "$PWD/examples" "$d/examples" && cd "$d"

# Export only (--batch_sizes / --num_batch_threads are baked into the export)
python -m examples.dlrm_benchmark.export_benchmark_model \
  --batched_export_path=/tmp/m/xse_batched --unbatched_export_path=/tmp/m/xse_unbatched \
  --baseline_weights_dir=/tmp/m/weights --platforms=cuda --embedding_size=128 \
  --num_batch_threads=12
# TF-preprocessor exports from the same weights (same batching flags;
# --tf_lookup=fused|gather; needs `pip install tensorflow`)
python -m examples.dlrm_benchmark.export_tf_benchmark_model \
  --weights_dir=/tmp/m/weights --tf_lookup=fused --platforms=cuda \
  --batched_export_path=/tmp/m/xse_tf_fused_batched \
  --unbatched_export_path=/tmp/m/xse_tf_fused_unbatched --num_batch_threads=12

# Python baseline (same flags as the XSE server where they exist)
python -m examples.dlrm_benchmark.python_server --weights_dir=/tmp/m/weights \
  --device=cuda --port=8501 --num_threads=12 [--dynamic_batching]
# Multi-process Python: 8 frontends + the device owner, same port
python -m examples.dlrm_benchmark.python_server --weights_dir=/tmp/m/weights \
  --device=cuda --port=8501 --num_threads=12 --num_processes=8 [--dynamic_batching]

# Load: closed loop C=64, 16 examples per request, 8 client processes with
# 8 connections each (64 connections; needed for SO_REUSEPORT to spread load)
python -m examples.dlrm_benchmark.loadgen --server=localhost:8500 \
  --weights_dir=/tmp/m/weights --request_batch_size=16 --concurrency=64 \
  --duration_s=30 --num_workers=8 --channels_per_worker=8 \
  --output=/tmp/r/xse-batch/closed_B16_C64_r1.json \
  --labels=arm=xse-batch,platform=cuda
# Load: open loop at 2000 req/s
python -m examples.dlrm_benchmark.loadgen ... --qps=2000

# Parity between two live servers
python -m examples.dlrm_benchmark.parity_check --server_a=localhost:8500 \
  --server_b=localhost:8501 --weights_dir=/tmp/m/weights
```

In  replace `python -m examples.dlrm_benchmark.X` with
`blaze-bin/third_party/neptune/examples/dlrm_benchmark/X`.

## Reading the results

`results/REPORT.md` has four sections:

* **Peak throughput** — best closed-loop examples/s per arm and request
  batch size, with p99 at that point, each arm's speedup vs. `py`, and an
  **"XSE vs best Python"** column: each XSE arm divided by the best Python
  arm at that request batch size (naming which one it was). That column is
  the headline number — it compares XSE against the strongest Python design
  on that platform rather than the weakest. When `xse-tf-*` arms are present
  a **"CompositeProcessor vs TF"** column follows: each JCE arm divided by
  the TF arm with the same batching, with the TF ÷ JCE ratio of server
  CPU-seconds per 1k examples in parentheses (see "TF preprocessor vs.
  CompositeProcessor").
* **Closed-loop sweep** — the full throughput/latency curve. Look for where
  each arm saturates (throughput flat, latency ∝ C) and compare
  `cpu-s/1k` (server CPU cost per 1k examples) at equal throughput. For
  `py-mp*` also check the per-frontend counts in `logs/py-mp*.server.log`
  at low concurrency: with one connection per worker only a subset of the
  frontends receives traffic, which is a property of `SO_REUSEPORT`, not a
  measurement error.
* **Open-loop latency** — p50/p90/p99 at 50 % / 80 % of each arm's own peak:
  the latency a user would see at a realistic utilisation. Error cells in
  both sweeps break the count down by gRPC status code (`error_codes` in the
  result JSON), e.g. `UNAVAILABLE` = the server's batch queue admission
  control, `CANCELLED` = gRPC core shedding load inside an overloaded Python
  server, `DEADLINE_EXCEEDED` = the client's `--timeout_s` — see
  Troubleshooting for what each means.
* **Run metadata** — labels, hosts, number of files.

Results table template (fill in from `REPORT.md`):

| Platform | B_req | xse-batch ex/s (p99) | xse-nobatch ex/s (p99) | best Python arm ex/s (p99) | XSE vs best Python | XSE cpu-s/1k | Python cpu-s/1k | Python peak PSS |
|---|---|---|---|---|---|---|---|---|
| A100 | 1 | | | | | | | |
| A100 | 16 | | | | | | | |
| A100 | 128 | | | | | | | |
| v5e | 1 | | | | | | | |
| … | | | | | | | | |

## Fairness rules and known caveats

* **Client must not be the bottleneck.** `loadgen.py` pre-serialises
  requests and does not parse responses; still, verify
  `client_cpu_per_1k_requests_s` in the JSON and keep `--client_cpus`
  separate from `--server_cpus`. If the client cores are saturated at the
  peak (`top` shows ~100 % on every client core), raise `--num_workers` or
  give the client more cores — for *all* arms.
* **Request batch sizes must be exported buckets** for `xse-nobatch`
  (`1,16,128 ⊂ {1,2,4,…,4096}` by default; with a custom `--batch_sizes`
  make sure `--request_batch_sizes` is a subset). The other arms pad.
* **`py` is a fair, idiomatic baseline, not a straw man:** one
  `jax.jit`-compiled function per bucket, explicit `device_put`, warm-up of
  every bucket before `READY`, a real thread pool, numpy vectorised lookup,
  padding done on the gathered embeddings rather than on the ids (cheaper).
  `py-batch` additionally gives it the same dynamic batching XSE has (same
  placement, in front of the lookup, and the same queue bound);
  `py-mp` / `py-mp-batch` give it as many processes as XSE has threads.
  Things it does *not* do (and XSE does) are exactly what the benchmark
  measures: C++ request handling without the GIL, a native embedding kernel,
  batching the preprocessor together with the model, zero-copy tensors.
* **`py-mp*` pay for their design, and the report shows it:** N private
  copies of the tables (peak PSS), one extra IPC hop per request (latency at
  `C=1`), and `SO_REUSEPORT`'s per-connection hashing (uneven frontend load
  at low connection counts; `connections` in the result JSON and the
  per-frontend counts in the server log make it visible). These are not
  tuned away because they are what a multi-process Python server costs.
* **Throughput is what the single node delivers.** Every server sees the
  same gRPC connection count (`--num_workers × --channels_per_worker`);
  XSE's execution pool and the Python executor(s) get the same
  `--num_threads` (see above for why the `*-batch` Python arms get extra
  *parked* handler threads; override with `--handler_threads`), and the
  batching arms the same `--num_batch_threads` and `--max_enqueued_batches`.
* Repeat counts of 3 with the median are the default; `REPORT.md` shows the
  coefficient of variation when it exceeds 0 — re-run if it is above ~10 %.
* GPU numbers are for a single accelerator opened by a single process (also
  in `py-mp*`, where only the owner touches the device); MIG or
  multi-GPU serving is out of scope.
* **Single node by design.** The Python servers bind with gRPC *local*
  (loopback-only, plaintext) server credentials and the clients use the
  matching local channel credentials for `localhost` targets, so nothing in
  the benchmark needs TLS and all arms are reached over the identical
  transport. Point `--server` at a non-loopback address only for the XSE
  arms (it falls back to a plain insecure channel).

## Troubleshooting

* `Check failed: RegisterDynamicPjrtPlugin(kTpuPjrtName, "TPU_LIBRARY_PATH") is OK
  (NOT_FOUND: Environment variable TPU_LIBRARY_PATH is not set. Can't load
  PJRT plugin TPU.)` in `logs/xse-batch.server.log` (TPU, image or OSS binary;
  `run_benchmark.sh` prints the last lines of that log when a server exits
  before opening its port): the OSS TPU server found no `libtpu.so`. It
  crashes in a static initializer, so no flag can help; the library must be
  provided through `TPU_LIBRARY_PATH`, and for the distroless `tpu-jax-only`
  image it must also be mounted into the container. `run_benchmark.sh` does
  both when it can find one — install `jax[tpu]` into the venv or pass
  `--tpu_library_path` (see "libtpu" under Prerequisites); when starting the
  container by hand use the TPU recipe in the top-level
  [README](../../README.md#using-docker-images).
* `Mismatched PJRT plugin PJRT API major version` (or a `minor version`
  complaint) right after that: the mounted `libtpu.so` is too old or too new
  for the server. PJRT is forward compatible by default (plugin ≥ server), so
  refresh the venv's `libtpu` (`uv pip install -U "jax[tpu]" -f
  https://storage.googleapis.com/jax-releases/libtpu_releases.html`) or pick
  an image date close to the installed `libtpu` release (`pip show libtpu`;
  `results/env.txt` records both).
* The TPU server (image or binary) opens no port and its log stays silent for
  minutes, then shows `Failed to convert value for key 'CHIPS_PER_HOST_BOUNDS'
  in 'tpu-env' to string: INTERNAL: Failed to fetch URL after 30 tries (http
  status: 404)`; the same happens to `jax.devices()` on the host: the VM has
  no `tpu-env` instance metadata (it was created with `gcloud compute
  instances create`, not the Cloud TPU API) and libtpu is retrying the
  metadata server, 30 × 1 s per variable (`TPU_STDERR_LOG_LEVEL=0` makes the
  retries visible). Export `TPU_SKIP_MDS_QUERY=1` and the topology variables
  for the machine type (see "TPU runtime environment" under Prerequisites;
  the script derives them itself for the known `ct*` types and forwards them
  into the container). `run_benchmark.sh` stops as soon as that line appears
  in the server log instead of waiting out the 30-minute port timeout.
* `DEADLINE_EXCEEDED: Failed to connect to [::]:8353` (or similar) from the
  TPU server or `jax.devices()` on a stock Ubuntu image: libtpu is trying to
  reach `tpunetd`, which only the TPU images run. For a single host set
  `LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true` (forwarded
  into the container by the script) or use the TPU image family.
* ``No `OptionalFunctionHandler` is found for key: …`` in
  `logs/xse-tf-*.server.log` (the server CHECK-fails right after it and
  `run_benchmark.sh` prints the hint): the model's preprocessor is a TF
  SavedModel function but the server is a JAX-only build. Use a server built
  with `--define=ENABLE_TF=1` or a `*-with-tf` image — see "TF preprocessor
  vs. CompositeProcessor" (there is no TPU image with TF; build from source).
  The script warns before exporting when `--xse_bin` lacks the TF runtime or
  `--xse_image` is not a `-with-tf` tag; `results/env.txt` records the check
  as `xse_tf_support`.
* `ModuleNotFoundError: No module named 'tensorflow'` from
  `export_tf_benchmark_model`: the `xse-tf-*` arms need TensorFlow in the
  venv for the export only (`pip install tensorflow`); it is deliberately not
  in `requirements.txt`. The server does not use the venv's TF.
* `ModuleNotFoundError: No module named 'xla_saved_model.protos'`: see the
  warning in "Software (OSS checkout)" — install the release wheel and never
  run the tools from the repo root.
* `grpc._channel._InactiveRpcError ... status = StatusCode.UNIMPLEMENTED,
  details = ""` from `parity_check.py` or `loadgen.py` against an XSE arm
  while the Python arms work and `logs/xse-batch.server.log` shows nothing:
  the client is calling a method the server never registered. gRPC method
  names carry the proto package, which is `jsv.serving` in  and
  `xla_serving.serving` in the OSS export, and the tools take it from the
  `predict_pb2` they import — so the venv's `xla_saved_model` protos and the
  server (image or binary) come from different builds, or a stale copy of
  `common.py` still hardcodes the  name. Check both sides:
  `python3 -c 'from xla_saved_model.protos.grpc import predict_pb2;
  print(predict_pb2.DESCRIPTOR.package)'` must print `xla_serving.serving` in
  an OSS venv, and `grpcurl -plaintext localhost:8500 list` (reflection is
  linked into the server) must list the same `PredictionService`.
* A `py-mp*` arm never prints `READY`, or `logs/py-mp*.server.log` shows
  `frontend i exited with -9 during startup` / `frontend i died (exit -9)`:
  a frontend was OOM-killed while loading its copy of the tables
  (`dmesg | tail`). Lower `--num_processes` (the script warned about
  `MemAvailable` before starting the arm) or cap the tables with
  `--max_vocab_size`. `AF_UNIX path too long` or `No space left on device`
  while creating the rings means `--ipc_dir` is too deep or `/dev/shm` is too
  small for `--num_processes × --ring_mb`; point `--ipc_dir` at a short path
  on a filesystem with room, or lower `--ring_mb` (the owner silently raises
  a ring to at least two maximum batches, 2 × 4096 rows by default). The
  owner removes its IPC directory on SIGTERM/SIGINT; after a SIGKILL or an
  OOM kill a `/dev/shm/dlrm_mp_<pid>_*` directory is left behind and can be
  deleted by hand.
* The `Frontend i served R requests / E examples` lines of a `py-mp*` log
  are very uneven at `C=1`: with one in-flight request there is effectively
  one active connection and `SO_REUSEPORT` pins it to one frontend. That is
  the design, not a bug; the higher concurrencies (and `--channels_per_worker`)
  are where the arm shows what N processes buy.
* Errors with `status = StatusCode.CANCELLED, details = "CANCELLED"`
  (`error_codes: {"CANCELLED": n}` in the result JSON) against a Python arm,
  typically in the open-loop points, with nothing in the server log and a
  latency distribution that plateaus (max ≈ a few seconds) instead of
  growing: that status is not produced by any code in this directory. It is
  gRPC **core's** server-side backlog protector — calls the transport has
  accepted but the Python layer has not yet picked up are kept in a
  per-process pending queue bounded at 1 000 (probabilistic rejects) /
  3 000 (hard) requests, and a shed call is destroyed with a bare
  `CANCELLED` status. gRPC-Python drains that queue with a single event
  thread that needs the GIL for every completion, so the symptom means that
  thread is being starved by many GIL-heavy handler threads in the same
  process, or that the offered load simply exceeds what one Python event
  thread can accept. The old `py-mp-batch` layout (64 handler threads per
  frontend each doing the numpy gather) did the former at 80 % of its peak;
  batching in front of the gather (handlers only parse and park, one or two
  threads per process do the heavy work) removed it — on a 2-frontend CPU
  reproduction the arm is now clean at 50 % and 80 % of its peak (flat p99)
  and sheds only at ≥ 100 %, which is the overload behaviour of every
  gRPC-Python server (`grpc.server(maximum_concurrent_rpcs=…)` does not
  help: the queue is below the Python layer, so that limit is never
  reached). If it reappears *below* the peak, look for a configuration that
  puts heavy work back into a wide handler pool (`--handler_threads` far
  above `--num_threads` *without* batching, or a `--num_processes` so small
  that each frontend gets hundreds of handlers) rather than for a bug in the
  servers. By contrast
  `status = StatusCode.UNAVAILABLE, details = "The batch scheduling queue
  … is full"` (`error_codes: {"UNAVAILABLE": n}`) is the *intended*
  admission control of the batched arms — XSE's `max_enqueued_batches` or
  the Python equivalent (`--max_enqueued_batches`, shared by all three
  batched arms) — and simply says the offered load exceeded the arm's
  capacity by more than its queue budget; the `Batcher stats:` line of the
  Python server log counts the same rejections. `DEADLINE_EXCEEDED` is the
  client's `--timeout_s` (30 s) and `drops` are requests the open-loop
  client never sent because `--max_inflight` was reached; both mean the
  server was queueing rather than rejecting.

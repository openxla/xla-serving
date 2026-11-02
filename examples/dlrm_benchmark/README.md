<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# DLRM-DCNv2 serving benchmark: XSE vs. pure-Python JAX

Single-node, apples-to-apples benchmark suite comparing
[XLA Serving Engine (XSE)](../../README.md) against single- and multi-process
Python `jax.jit` gRPC servers on the same model, weights, wire protocol, and
CPU/accelerator budget.

For published Cloud TPU v6e-1 results, charts, bottleneck analysis, and
end-to-end reproduction recipes, see the
**[DLRM serving benchmark guide](../../docs/benchmarks/dlrm_serving.md)**.

## Workload and arms

The workload is the Criteo-scale **DLRM-DCNv2** model from
[`examples/dlrm`](../dlrm): 26 categorical features looked up and sum-pooled
from host-CPU embedding tables (~6.2M rows at `embedding_size = 128`, ~3.2 GB)
plus 13 `float32` dense features feeding an accelerator DCNv2 network. Host
preprocessing in XSE runs via `xla_serving_data`'s C++ `CompositeProcessor`
(JAX CPU Embedding / JCE), with no TensorFlow dependency.

| Arm | Server | Host lookup | Batching |
|---|---|---|---|
| `xse-batch` | `xla_serving_main_{cpu,gpu,tpu}` (C++) | C++ `CompositeProcessor` (JCE) | `PRE_PROCESSOR_AND_MODEL_FUNCTION`, `PAD_UP` to power-of-two buckets `1..4096` |
| `xse-nobatch` | Same binary, second bundle | C++ `CompositeProcessor` (JCE) | `NO_BATCHING` |
| `py` | [`python_server.py`](python_server.py) (`grpc.server` + `jax.jit`) | NumPy `take` + `sum` | None (one `jit` call per request, padded to bucket) |
| `py-batch` | `python_server.py --dynamic_batching` | NumPy `take` + `sum` | [`DynamicBatcher`](python_server.py) in front of lookup (same buckets, timeout, threads, and queue bound as XSE) |
| `py-mp` | `python_server.py --num_processes=N` ([`multiprocess_server.py`](multiprocess_server.py)) | NumPy in `N` `SO_REUSEPORT` frontends | None (`1` accelerator-owner process runs `jax.jit` via `/dev/shm` rings) |
| `py-mp-batch` | `python_server.py --num_processes=N --dynamic_batching` | NumPy in `N` frontends (once per batch) | Per-frontend `DynamicBatcher` + zero-timeout batch merger in the accelerator owner |

### Fairness invariants

- **Shared weights & numerical parity:**
  [`export_benchmark_model.py`](export_benchmark_model.py) generates seeded
  weights once for all exports and `.npz` archives. Before timing any arm,
  [`parity_check.py`](parity_check.py) verifies predictions agree with an
  independent reference within `max|Δ| ≤ 1e-4`.
- **Identical wire protocol & load pattern:** [`loadgen.py`](loadgen.py) sends
  identical pre-serialized `PredictionService/Predict` (`HostBufferProto`)
  requests over `--num_workers × --channels_per_worker` persistent gRPC
  connections (64 by default so `SO_REUSEPORT` balances evenly).
- **Matched batching & admission control:** All `*-batch` arms coalesce
  requests *before* the embedding lookup using identical power-of-two buckets
  (`--batch_sizes`, `1..4096`), `--batch_timeout_micros` (`1000`),
  `--num_batch_threads`, and `--max_enqueued_batches` (`64`, rejecting excess
  load with `UNAVAILABLE`).
- **Disjoint CPU pinning, matched thread pools & full-tree monitoring:**
  `--server_cpus` pins the container (`--cpuset-cpus`) or host process
  (`taskset -c`), `--num_threads` sets both the server worker threads and XSE's
  JCE lookup pool (`--jce_max_cpu_slots`), and `--client_cpus` pins both
  `loadgen.py` and [`monitor.py`](monitor.py). `monitor.py` sums CPU time and
  proportional set size (PSS) across the entire server process tree so
  multi-process table copies are accounted accurately.

## Quick start

### 1. Environment setup

Install Python `3.12+`, the `xla-saved-model` and `xla-serving-data` packages
(see [Installation](../../docs/installation.md)), the benchmark dependencies,
and the JAX accelerator package matching your VM:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

pip install xla-saved-model xla-serving-data
pip install -r examples/dlrm_benchmark/requirements.txt

# Choose one matching the host hardware:
pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
# pip install "jax[cuda12]"   # NVIDIA GPU (or jax[cuda13])
# pip install jax             # CPU smoke test
```

> **Note:** Do not run `pip install -e .` without `bazel` on `PATH` (generated
> `xla_saved_model.protos` modules require `bazel`), and do not invoke
> `python -m examples.dlrm_benchmark.*` directly from the repository root where
> the source `xla_saved_model/` folder shadows the installed wheel.
> `run_benchmark.sh` automatically runs Python tools from an isolated
> `<work_dir>/pysrc` directory containing only a symlink to `examples/`.

### 2. Run the full suite

Authenticate Docker (`gcloud auth configure-docker us-docker.pkg.dev --quiet`)
and run [`run_benchmark.sh`](run_benchmark.sh):

```bash
# Cloud TPU (e.g. ct6e-standard-1t / v6e-1)
examples/dlrm_benchmark/run_benchmark.sh \
  --platform tpu \
  --xse_image us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only:nightly \
  --server_cpus 0-7 \
  --client_cpus 8-23 \
  --num_threads 8 \
  --num_processes 8 \
  --work_dir /tmp/dlrm_benchmark

# NVIDIA GPU (e.g. a2-highgpu-1g / A100)
examples/dlrm_benchmark/run_benchmark.sh \
  --platform cuda \
  --xse_image us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12:nightly \
  --server_cpus 0-7 \
  --client_cpus 8-11 \
  --num_threads 8 \
  --work_dir /tmp/dlrm_benchmark

# CPU functional smoke test (~5 min, capped vocabularies; pass --server_cpus
# and --client_cpus when timing on CPU)
examples/dlrm_benchmark/run_benchmark.sh \
  --platform cpu \
  --xse_image us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:nightly \
  --max_vocab_size 1000 \
  --duration 5 --warmup 1 --repeats 1 \
  --concurrencies 1,8 --request_batch_sizes 1,16 \
  --num_threads 2 --num_processes 2
```

To benchmark a locally compiled server binary instead of a container image, pass
`--xse_bin bazel-bin/runtime/serving/gcp/server/xla_serving_main_{cpu,gpu,tpu}`
(see [Serving and runtime](../../docs/guides/serving.md)). Pass `--dry_run` to
inspect all resolved commands without executing them.

## Running individual tools

Create an isolated import directory linking `examples/` when running individual
modules directly:

```bash
pysrc=$(mktemp -d) && ln -s "$PWD/examples" "$pysrc/examples" && cd "$pysrc"

# 1. Export XSE bundles and Python baseline weights
python3 -m examples.dlrm_benchmark.export_benchmark_model \
  --batched_export_path=/tmp/m/xse_batched \
  --unbatched_export_path=/tmp/m/xse_unbatched \
  --baseline_weights_dir=/tmp/m/weights \
  --platforms=tpu --num_batch_threads=8

# 2. Start single-process or multi-process Python server
python3 -m examples.dlrm_benchmark.python_server \
  --weights_dir=/tmp/m/weights --device=tpu --port=8501 \
  --num_threads=8 --num_processes=8 --dynamic_batching

# 3. Check numerical parity between two live servers (or vs. local reference)
python3 -m examples.dlrm_benchmark.parity_check \
  --server_a=localhost:8500 --server_b=localhost:8501 \
  --weights_dir=/tmp/m/weights

# 4. Run closed-loop (--concurrency) or open-loop (--qps) load
python3 -m examples.dlrm_benchmark.loadgen \
  --server=localhost:8500 --weights_dir=/tmp/m/weights \
  --request_batch_size=16 --concurrency=64 --duration_s=30 \
  --num_workers=8 --channels_per_worker=8 \
  --output=/tmp/r/xse-batch/closed_B16_C64_r1.json \
  --labels=arm=xse-batch,platform=tpu

# 5. Render REPORT.md from collected JSON files
python3 -m examples.dlrm_benchmark.report \
  --results_dir=/tmp/r --p99_budget_ms=20 --output=/tmp/r/REPORT.md
```

## Reading `REPORT.md`

[`report.py`](report.py) produces five sections:

1. **Throughput under `p99 ≤ 20 ms` latency budget** (configurable via
   `--p99_budget_ms`): highest error-free `examples/s` per arm whose median
   `p99` stayed within budget, plus `Speedup vs py` and `XSE vs best Python`.
2. **Peak throughput (no latency limit)**: highest closed-loop `examples/s` per
   arm across all concurrency levels.
3. **Closed-loop concurrency sweep**: `req/s`, `examples/s`, `p50 / p99` (ms),
   `server CPU-s per 1k examples`, peak memory (RSS or tree PSS), and repeat
   coefficient of variation.
4. **Open-loop (Poisson) latency under load**: achieved `req/s`,
   `p50 / p90 / p99 / max` latency, and error breakdown at 50% and 80% of each
   arm's own peak.
5. **Run metadata**: result file count, hosts, and recorded benchmark labels.

## Troubleshooting

- **TPU metadata timeouts (`Failed to fetch URL after 30 tries` on `tpu-env`):**
  Compute Engine VMs created with `gcloud compute instances create` (rather than
  `gcloud compute tpus tpu-vm create`) lack `tpu-env` metadata.
  `run_benchmark.sh` automatically derives `TPU_SKIP_MDS_QUERY=1` and the
  single-host `TPU_*` topology variables for `ct6e-*`, `ct5lp-*`, and `ct5p-*`
  machine types and forwards all exported `TPU_*` and `LIBTPU_INIT_ARGS`
  variables into `docker run`. On stock non-TPU Ubuntu images without
  `tpunetd`, also export `TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/0` and
  `LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true`.
- **Bare TPU binary missing `libtpu.so`:** The `tpu-jax-only` container image
  bundles `/usr/lib/libtpu.so` out of the box. A bare `--xse_bin` TPU binary
  requires `TPU_LIBRARY_PATH`; `run_benchmark.sh` resolves it from
  `--tpu_library_path`, `$TPU_LIBRARY_PATH`, or the installed `libtpu` Python
  package.
- **`py-mp*` OOM during startup (`frontend i exited with -9`):** Each frontend
  process loads a private copy of the 3.2 GB embedding tables. Lower
  `--num_processes` (or `--max_vocab_size` for functional tests) so
  `num_processes × 3.2 GB + 4 GB` fits in available RAM.
- **Understanding RPC error codes (`error_codes` in JSON / `REPORT.md`):**
  - `UNAVAILABLE` (`batch scheduling queue … is full`): server admission
    control (`--max_enqueued_batches`) rejecting load above queue capacity.
  - `CANCELLED`: gRPC C-core server backlog shedding when a Python process's
    single GIL-bound completion thread cannot drain incoming calls fast enough.
  - `DEADLINE_EXCEEDED`: client RPC deadline (`--timeout_s`, default `30 s`).

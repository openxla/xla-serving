# DCN v2 CPU benchmark

Benchmark a synthetic FP32 Deep & Cross Network (DCN v2) with one ModelRunner
per process. No dataset or trained checkpoint is required.

Follow the [Python setup](../../README.md#python-development) and install matching
native wheels from `//data:xla_serving_data` and `//runtime/python:xla_saved_model_runner`.

Run this small batch-4 example from the repository root:

```bash
unset NPROC JCE_BLOCK_SIZE
export PJRT_NPROC=1 OMP_NUM_THREADS=1 OMP_DYNAMIC=FALSE JCE_MAX_CPU_SLOTS=1
dlrm_dir=$(mktemp -d)

python -m examples.dlrm.run_export_jce \
  --export_path "$dlrm_dir/model" --benchmark_inputs "$dlrm_dir/inputs.npz" \
  --global_batch_size 4 --vocab_size_cap 32 --num_input_batches 2 --seed 0

python -m examples.dlrm.benchmark_model_runner \
  --model_path "$dlrm_dir/model" --inputs "$dlrm_dir/inputs.npz" \
  --concurrency 1 --warmup 1 --duration 1
```

To compare process layouts with 64 requested worker threads, keep the exported
model and inputs fixed and vary processes × threads per process. For example,
run 8 processes × 8 threads:

```bash
OMP_NUM_THREADS=8 PJRT_NPROC=8 JCE_MAX_CPU_SLOTS=8 OMP_DYNAMIC=FALSE \
  python -m examples.dlrm.benchmark_model_runner \
  --model_path "$dlrm_dir/model" --inputs "$dlrm_dir/inputs.npz" \
  --concurrency 8
```

Workers share the inherited CPU affinity. The environment variables set separate
per-process runtime pools, so 8 × 8 is a requested layout rather than a hard cap
of 64 operating-system threads.

| Argument | Meaning / unit | Default |
| --- | --- | --- |
| `--export_path` / `--model_path` | Model directory: write / load | Required |
| `--benchmark_inputs` / `--inputs` | Input/reference `.npz`: write / load | Optional / required |
| `--global_batch_size` | Rows per request | 2 |
| `--vocab_size_cap` | Maximum rows per embedding table | No cap |
| `--num_input_batches` | Distinct prepared input batches | 8 |
| `--seed` | Integer random seed | 0 |
| `--num_dense_features` | Dense features per row | 13 |
| `--embedding_size` | Values per embedding vector | 128 |
| `--use_capped_vocab` | Cap tables at 100 rows; overrides `--vocab_size_cap` | false |
| `--concurrency` | ModelRunner processes | 1 |
| `--warmup` | Extra untimed requests per process | 20 |
| `--duration` | Timed window (seconds) | 10 |
| `--timeout` | Worker wait allowance (seconds) | 60 |
| `--signature_name` | Serving signature | `serving_default` |

JSON reports requests and rows/s through output materialization and final drain.
Loading, warmup and prediction checks are untimed.

Re-export when changing model or prepared input options.

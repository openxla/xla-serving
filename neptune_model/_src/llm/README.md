# Neptune LLM Export & Benchmarking Framework

The **Neptune LLM Export Framework** (`neptune_model._src.llm`) provides a
streamlined pipeline for exporting Large Language Models (LLMs) backed by
`tpu-inference` into native **Orbax Model** bundles and benchmarking their
performance across JAX execution, C++ model runner bindings, and gRPC serving
engines.

Exported Orbax model bundles embed KerasHub/TF tokenizer pre-processing and
post-processing graphs alongside compiled XLA/HLO decoding loops,
enabling low-latency, zero-overhead deployment on **Google Cloud TPU**
accelerators via Neptune's C++ Model Serving Server
(`neptune_serving_main_tpu`).

---

## 1. Installation & Environment Setup

Install the LLM module dependencies directly using the `llm` optional dependency
group defined in `pyproject.toml`. This automatically installs `vllm-tpu`
directly from PyPI along with its TPU execution dependencies:

```bash
# Install package with LLM export dependencies
pip install -e ".[llm]"

# Or with test dependencies:
pip install -e ".[llm,test]"
```

### Environment Variables
When running CLI export, server, or benchmark commands on TPU VMs, set:

```bash
export TPU_LIBRARY_PATH=$(python3 -c "import libtpu; print(libtpu.get_library_path())")
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

## 2. Model Export CLI (`neptune_model._src.llm.export_llm`)

Export models backed by `tpu-inference` into serving-ready Orbax model bundles:

```bash
# Display detailed usage help
python3 -m neptune_model._src.llm.export_llm --help

# Example 1: Export Qwen/Qwen3-0.6B with Greedy Sampling
python3 -m neptune_model._src.llm.export_llm \
  --model Qwen/Qwen3-0.6B \
  --export-dir /tmp/qwen3_greedy_export \
  --sampling-type greedy \
  --allowed-batch-sizes 1 \
  --max-prompt-len 2000 \
  --max-new-tokens 4

# Example 2: Export Qwen/Qwen3-0.6B with Beam Search (beam_width=30)
python3 -m neptune_model._src.llm.export_llm \
  --model Qwen/Qwen3-0.6B \
  --export-dir /tmp/qwen3_serving_b30_p2000 \
  --sampling-type beam_search \
  --beam-width 30 \
  --allowed-batch-sizes 1 \
  --max-prompt-len 2000 \
  --max-new-tokens 4

# Example 3: Export Qwen/Qwen3-0.6B with Beam Search (beam_width=100)
python3 -m neptune_model._src.llm.export_llm \
  --model Qwen/Qwen3-0.6B \
  --export-dir /tmp/qwen3_serving_b100_p2000 \
  --sampling-type beam_search \
  --beam-width 100 \
  --allowed-batch-sizes 1 \
  --max-prompt-len 2000 \
  --max-new-tokens 4
```

### Command-Line Arguments

| Argument | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `--model` | `str` | **Required** | Model name or Hugging Face ID (e.g. `Qwen/Qwen3-0.6B`). |
| `--export-dir` | `str` | **Required** | Output directory for the Orbax model bundle. |
| `--allowed-batch-sizes` | `str` | `1` | Comma-separated list of allowed serving batch sizes (e.g. `1`, `1,2,4`). |
| `--max-prompt-len` | `int` | `1024` | Maximum allowed input prompt tokens. |
| `--max-new-tokens` | `int` | `8` | Maximum new tokens to generate per request. |
| `--max-seq-len` | `int` | `None` | Maximum sequence length (`prompt_len + new_tokens`). |
| `--sampling-type` | `str` | `greedy` | Method: `greedy`, `top_k`, `beam_search`, `scoring`. |
| `--beam-width` | `int` | `4` | Beam width (used with `beam_search`). |
| `--load-format` | `str` | `auto` | Model weights loader format (`auto`, `jax_dummy`, etc.). |
| `--tp-size` | `int` | `None` | Tensor parallel size (defaults to local JAX devices). |

---

## 3. Deploying & Serving on Native C++ Model Server

You can serve any exported model bundle using Neptune's native C++ Model Serving
Server (`neptune_serving_main_tpu`), which executes the Orbax model bundle on
Google Cloud TPU via Orbax and IFRT with zero Python runtime overhead.

### A. Build the C++ Serving Server Binary
Build the TPU model serving binary using Bazel (enabling TensorFlow tokenization
support with `--define=ENABLE_TF=1`):

```bash
bazel build -c opt --define=ENABLE_TF=1 //runtime/serving/gcp/server:neptune_serving_main_tpu
```

### B. Launch the C++ Model Server
Set environment variables so the server can locate `libtpu.so` and start the
server:

```bash
export TPU_LIBRARY_PATH=$(python3 -c "import libtpu; print(libtpu.get_library_path())")

./bazel-bin/runtime/serving/gcp/server/neptune_serving_main_tpu \
  --model_name=qwen3 \
  --model_path=/tmp/qwen3_serving_b30_p2000 \
  --port=8500 \
  --use_insecure_grpc_credentials=true \
  --orbax_grpc_server_platform=tpu
```

When the server finishes loading the Orbax model bundle, it prints
`JAX Serving listening on [::]:8500`.

### C. Send Requests via Benchmark CLI
Query the running server and measure throughput using `benchmark_llm`:

```bash
python3 -m neptune_model._src.llm.benchmark_llm \
  --mode grpc_client \
  --server-address localhost:8500 \
  --model-name qwen3 \
  --prompt "The most unexpected thing happened when I opened the door," \
  --batch-size 1 \
  --new-tokens 4 \
  --num-batches 5
```

---

## 4. Benchmarking Suite (`neptune_model._src.llm.benchmark_llm`)

Use the consolidated `benchmark_llm` CLI to benchmark model generation loops,
C++ model runner bindings, or gRPC serving performance. It supports three modes
via `--mode`:

### Mode 1: `model_fn` (Direct JAX Model Generation Loop)
Benchmarks the JIT-compiled JAX generation loop directly without server
overhead:

```bash
python3 -m neptune_model._src.llm.benchmark_llm \
  --mode model_fn \
  --model Qwen/Qwen3-0.6B \
  --sampling-type beam_search \
  --beam-width 30 \
  --prompt-len 2000 \
  --new-tokens 4 \
  --batch-size 1 \
  --num-batches 5
```

### Mode 2: `grpc_client` (C++ Model Server via gRPC)
Sends gRPC requests to an active C++ Neptune Model Server
(`neptune_serving_main_tpu`), verifying generated output beams and measuring
end-to-end request throughput:

```bash
python3 -m neptune_model._src.llm.benchmark_llm \
  --mode grpc_client \
  --server-address localhost:8500 \
  --model-name qwen3 \
  --batch-size 1 \
  --new-tokens 4 \
  --num-batches 5
```

### Mode 3: `model_runner` (C++ Model Runner Python Bindings)
Benchmarks exported Orbax model bundles directly using Neptune's C++
`ModelRunner` Python bindings:

```bash
python3 -m neptune_model._src.llm.benchmark_llm \
  --mode model_runner \
  --model-path /tmp/qwen3_serving_b30_p2000 \
  --prompt "The most unexpected thing happened when I opened the door," \
  --batch-size 1 \
  --num-batches 5
```

---

## 5. Running Unit & Integration Tests

### Pytest Testing (OSS Setup)
Run unit tests with pytest:

```bash
# Run unit tests
pytest -s neptune_model/_src/llm/export_llm_test.py
pytest -s neptune_model/_src/llm/benchmark_llm_test.py
pytest -s neptune_model/_src/llm/tpu_inference/tpu_exporter_test.py
```

### Google3 / Blaze Testing
Run the full LLM test suite using `blaze`:

```bash
SKYBUILD=1 blaze test //third_party/neptune/neptune_model/_src/llm/...
```

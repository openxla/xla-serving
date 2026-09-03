# Neptune LLM Example: Qwen3-0.6B Beam Search

Note: Full documentation for Neptune LLM can be found at
`neptune/neptune_model/_src/llm/README.md`.

This example demonstrates how to export, serve, and benchmark an open-source
Large Language Model (**[Qwen/Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B)**)
with **Beam Search sampling** (`beam_width = 30`) using Neptune on Google Cloud
TPU.

---

## Target Configuration

- **Model**: `Qwen/Qwen3-0.6B`
- **Sampling Strategy**: Beam Search (`beam_width = 30`)
- **Max Input Prompt Length**: 2000 tokens
- **Max New Generation Tokens**: 4 tokens
- **Allowed Serving Batch Sizes**: `1`

---

## 1. Environment Setup

Create and activate a virtual environment (Python >= 3.12 is recommended), then
install the LLM export dependencies:

```bash
# Create and activate a Python virtual environment
python3.12 -m venv neptune-venv
source neptune-venv/bin/activate

# Install Neptune with LLM dependencies
pip install -e ".[llm]"

# Set environment variables for TPU runtime and protobuf
export TPU_LIBRARY_PATH=$(python3 -c "import libtpu; print(libtpu.get_library_path())")
export PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python
```

---

## 2. Exporting the Model via CLI (`export_llm`)

Export the model directly using Neptune's LLM export CLI:

```bash
python3 -m neptune_model._src.llm.export_llm \
  --model Qwen/Qwen3-0.6B \
  --export-dir /tmp/qwen3_serving_b30_p2000 \
  --sampling-type beam_search \
  --beam-width 30 \
  --max-prompt-len 2000 \
  --max-new-tokens 4 \
  --allowed-batch-sizes 1
```

The exported bundle at `/tmp/qwen3_serving_b30_p2000` contains:

- `checkpoint/`: Model weights saved in Orbax format and compiled JAX/XLA
  programs.
- `tokenizer/`: Serialized KerasHub tokenizer assets.
- `orchestration.pb`: Graph linking pre-processing (tokenization),
  compiled model execution (XLA), and post-processing (detokenization).

---

## 3. Serving via Neptune C++ Model Server

Build the standalone C++ Model Serving Server binary
(`neptune_serving_main_tpu`) using Bazel (enabling TensorFlow tokenization
support with `--define=ENABLE_TF=1`):

```bash
# Build the server binary
bazel build -c opt --define=ENABLE_TF=1 //runtime/serving/gcp/server:neptune_serving_main_tpu

# Launch the server on TPU
export TPU_LIBRARY_PATH=$(python3 -c "import libtpu; print(libtpu.get_library_path())")

./bazel-bin/runtime/serving/gcp/server/neptune_serving_main_tpu \
  --model_name=qwen3 \
  --model_path=/tmp/qwen3_serving_b30_p2000 \
  --port=8500 \
  --use_insecure_grpc_credentials=true \
  --orbax_grpc_server_platform=tpu
```

When ready, the server logs `JAX Serving listening on [::]:8500`.

---

## 4. Querying the Server via Standalone Python Client

When querying the server over gRPC, ensure `grpcio` and `grpcio-tools` are
installed:

```bash
pip install grpcio grpcio-tools
```

Query the running server using the standalone example client
`run_grpc_client.py`:

```bash
python3 -m examples.llm.run_grpc_client \
  --server_address=localhost:8500 \
  --model_name=qwen3 \
  --prompt="The most unexpected thing happened when I opened the door,"
```

Sample output:
```text
==================================================
Prompt: The most unexpected thing happened when I opened the door,
==================================================
  [Beam  0]:  and it was a
  [Beam  1]:  and it was the
  [Beam  2]:  and it was not
  ...
  [Beam 29]:  and I was not
==================================================
```

---

## 5. Using the C++ Model Runner Python Bindings

You can also run the exported bundle locally in Python without the gRPC server
using Neptune's `neptune_model_runner` Python bindings.

You can install `neptune_model_runner-*.whl` directly from
**[Neptune GitHub Releases](https://github.com/AI-Hypercomputer/neptune/releases)**,
or build the wheel from source with Bazel:

```bash
# Build the wheel from source:
bazel build -c opt //runtime/python:neptune_model_runner --@rules_python//python/config_settings:python_version=3.12

# Install the wheel:
pip install bazel-bin/runtime/python/neptune_model_runner-*.whl
```

Run inference directly in Python:

```python
from neptune_model_runner import model_runner
import numpy as np

# Load exported bundle on TPU
runner = model_runner.ModelRunner(
    model_name="qwen3",
    model_path="/tmp/qwen3_serving_b30_p2000",
    platform="tpu",
)
sig = runner.signature("serving_default")

# Run inference with raw string input
prompts = np.array(
    ["The most unexpected thing happened when I opened the door,"],
    dtype=object,
)
outputs = sig.run_with_dict_return(text_input=prompts)

# outputs["text_output"] has shape [1, 30] (1 prompt, 30 candidate beams)
print("Decoded Beams:", outputs["text_output"][0])
```

---

## 6. Benchmarking Performance (`benchmark_llm`)

Use `neptune_model._src.llm.benchmark_llm` to measure latency, throughput (QPS),
and token generation rates across different execution layers:

### A. Direct JAX Generation Loop (`model_fn`)
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

### B. End-to-End gRPC Server (`grpc_client`)
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

### C. Local C++ Model Runner (`model_runner`)
```bash
python3 -m neptune_model._src.llm.benchmark_llm \
  --mode model_runner \
  --model-path /tmp/qwen3_serving_b30_p2000 \
  --prompt "The most unexpected thing happened when I opened the door," \
  --batch-size 1 \
  --num-batches 5
```

---

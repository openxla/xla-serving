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

## 4. Querying the Server via gRPC

Neptune exposes standard gRPC endpoints over `neptune.serving.PredictionService`
(RPC method `Predict`). Clients can query the server directly in Python, C++, or
any gRPC-supported language.

### A. API Specifications

- **Service**: `neptune.serving.PredictionService`
- **RPC Method**: `Predict` (URI: `/neptune.serving.PredictionService/Predict`)
- **Request Message**: `neptune.serving.PredictRequest`
  - `model_spec.name`: Model name matching the server (`--model_name`, e.g. `"qwen3"`).
  - `model_spec.signature_name`: Serving signature (defaults to `"serving_default"`).
  - `inputs`: `map<string, neptune.HostBufferProto>`.
- **Response Message**: `neptune.serving.PredictResponse`
  - `outputs`: `map<string, neptune.HostBufferProto>`.

#### Expected Tensor I/O Signatures

| Mode | Input Tensor Name | Input DType & Shape | Output Tensor Name | Output DType & Shape |
| :--- | :--- | :--- | :--- | :--- |
| **Generation (Greedy / Top-K)** | `"text_input"` | `KIND_STRING`, `[batch_size]` | `"text_output"` | `KIND_STRING`, `[batch_size]` |
| **Generation (Beam Search)** | `"text_input"` | `KIND_STRING`, `[batch_size]` | `"text_output"` | `KIND_STRING`, `[batch_size, beam_width]` |
| **Candidate Scoring** | `"text_input"`<br>`"candidate_texts"` | `KIND_STRING`, `[batch_size]`<br>`KIND_STRING`, `[batch_size, num_cands]` | `"scores"`<br>`"step0_logprobs"` | `KIND_F32` / `KIND_F64`, `[batch_size, num_cands]` |

---

### B. Python Querying Snippets

```python
import grpc
import numpy as np
from neptune_model.protos.grpc import host_buffer_pb2, predict_pb2
from neptune_model.protos.xla.python.ifrt import dtype_pb2

channel = grpc.insecure_channel("localhost:8500")
predict_fn = channel.unary_unary(
    "/neptune.serving.PredictionService/Predict",
    request_serializer=predict_pb2.PredictRequest.SerializeToString,
    response_deserializer=predict_pb2.PredictResponse.FromString,
)


# --- 1. Text Generation (Greedy, Top-K, Beam Search) ---
def generate_text(
    prompts: list[str],
    model_name: str = "qwen3",
    signature_name: str = "serving_default",
) -> list[str] | list[list[str]]:
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name

  hb_proto = host_buffer_pb2.HostBufferProto()
  hb_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
  hb_proto.shape.dims.append(len(prompts))
  for p in prompts:
    hb_proto.string_data.elements.append(p.encode("utf-8"))
  request.inputs["text_input"].CopyFrom(hb_proto)

  response = predict_fn(request)
  out = response.outputs["text_output"]
  elements = [
      raw.decode("utf-8", errors="replace") for raw in out.string_data.elements
  ]
  return np.array(elements, dtype=object).reshape(list(out.shape.dims)).tolist()


# --- 2. Candidate Scoring ---
def score_candidates(
    prompts: list[str],
    candidate_texts: list[list[str]],
    model_name: str = "qwen3",
    signature_name: str = "serving_default",
) -> np.ndarray:
  request = predict_pb2.PredictRequest()
  request.model_spec.name = model_name
  request.model_spec.signature_name = signature_name

  batch_size = len(prompts)
  num_cands = len(candidate_texts[0])

  # text_input: shape [batch_size]
  p_proto = host_buffer_pb2.HostBufferProto()
  p_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
  p_proto.shape.dims.append(batch_size)
  for p in prompts:
    p_proto.string_data.elements.append(p.encode("utf-8"))
  request.inputs["text_input"].CopyFrom(p_proto)

  # candidate_texts: shape [batch_size, num_cands]
  c_proto = host_buffer_pb2.HostBufferProto()
  c_proto.dtype.kind = dtype_pb2.DTypeProto.Kind.Value("KIND_STRING")
  c_proto.shape.dims.extend([batch_size, num_cands])
  for batch in candidate_texts:
    for cand in batch:
      c_proto.string_data.elements.append(cand.encode("utf-8"))
  request.inputs["candidate_texts"].CopyFrom(c_proto)

  response = predict_fn(request)
  scores_buf = response.outputs["scores"]
  dtype = (
      np.float32
      if scores_buf.dtype.kind == dtype_pb2.DTypeProto.Kind.Value("KIND_F32")
      else np.float64
  )
  return np.frombuffer(scores_buf.raw_data, dtype=dtype).reshape(
      list(scores_buf.shape.dims)
  )
```

---

### C. C++ Querying Snippets

```cpp
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include "include/grpcpp/create_channel.h"
#include "protos/grpc/host_buffer.pb.h"
#include "protos/grpc/predict.pb.h"
#include "protos/grpc/prediction_service.grpc.pb.h"
#include "xla/python/ifrt/dtype.pb.h"

// --- 1. Text Generation ---
void QueryTextGeneration(neptune::serving::PredictionService::Stub* stub) {
  neptune::serving::PredictRequest request;
  request.mutable_model_spec()->set_name("qwen3");
  request.mutable_model_spec()->set_signature_name("serving_default");

  auto* input = &(*request.mutable_inputs())["text_input"];
  input->mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_STRING);
  input->mutable_shape()->add_dims(1);
  input->mutable_string_data()->add_elements(
      "The most unexpected thing happened when I opened the door,");

  grpc::ClientContext context;
  neptune::serving::PredictResponse response;
  if (stub->Predict(&context, request, &response).ok()) {
    const auto& out = response.outputs().at("text_output");
    for (const auto& elem : out.string_data().elements()) {
      std::cout << "Completion: " << elem << std::endl;
    }
  }
}

// --- 2. Candidate Scoring ---
void QueryCandidateScoring(neptune::serving::PredictionService::Stub* stub) {
  neptune::serving::PredictRequest request;
  request.mutable_model_spec()->set_name("qwen3");
  request.mutable_model_spec()->set_signature_name("serving_default");

  // text_input: shape [1]
  auto* text_in = &(*request.mutable_inputs())["text_input"];
  text_in->mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_STRING);
  text_in->mutable_shape()->add_dims(1);
  text_in->mutable_string_data()->add_elements("Context prompt");

  // candidate_texts: shape [1, 2]
  auto* cands_in = &(*request.mutable_inputs())["candidate_texts"];
  cands_in->mutable_dtype()->set_kind(xla::ifrt::DTypeProto::KIND_STRING);
  cands_in->mutable_shape()->add_dims(1);
  cands_in->mutable_shape()->add_dims(2);
  cands_in->mutable_string_data()->add_elements(" candidate one");
  cands_in->mutable_string_data()->add_elements(" candidate two");

  grpc::ClientContext context;
  neptune::serving::PredictResponse response;
  if (stub->Predict(&context, request, &response).ok()) {
    const auto& scores = response.outputs().at("scores");
    const float* score_data = reinterpret_cast<const float*>(scores.raw_data().data());
    std::cout << "Scores: " << score_data[0] << ", " << score_data[1] << std::endl;
  }
}

int main() {
  auto channel = grpc::CreateChannel("localhost:8500", grpc::InsecureChannelCredentials());
  auto stub = neptune::serving::PredictionService::NewStub(channel);
  QueryTextGeneration(stub.get());
  QueryCandidateScoring(stub.get());
  return 0;
}
```

## 5. Benchmarking Suite (`neptune_model._src.llm.benchmark_llm`)

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

## 6. Running Unit & Integration Tests

### Pytest Testing (OSS Setup)
Run unit tests with pytest:

```bash
# Run unit tests
pytest -s neptune_model/_src/llm/export_llm_test.py
pytest -s neptune_model/_src/llm/benchmark_llm_test.py
pytest -s neptune_model/_src/llm/tpu_inference/tpu_exporter_test.py
```

<!-- copybara:strip_begin -->
### Google3 / Blaze Testing
Run the full LLM test suite using `blaze`:

```bash
SKYBUILD=1 blaze test //third_party/neptune/neptune_model/_src/llm/...
```
<!-- copybara:strip_end -->

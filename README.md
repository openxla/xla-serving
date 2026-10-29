# XLA Serving Engine

XLA Serving Engine (XSE) is a C++ inference runtime and model packaging toolkit for
the OpenXLA ecosystem. Export a model once from JAX (Flax, Keras 3, or plain
`jax.jit`) or from PyTorch via [torchax](https://github.com/pytorch/xla/tree/master/torchax);
serve it on CPU, NVIDIA GPU, or Cloud TPU from a single self-contained bundle,
with no Python in the request path.

**Documentation: <https://xla-serving.readthedocs.io>**

## Why XLA Serving Engine

- **Native XLA from training to production.** Export uses `jax.export` to
  capture StableHLO and an [Orbax](https://github.com/google/orbax) checkpoint
  directly—no graph conversion to ONNX or TensorFlow, and full support for
  custom XLA calls, Pallas kernels, and multi-device sharding.
- **No Python on the serving path.** A standalone C++ server handles gRPC,
  dynamic request batching, host-side pre- and post-processing, and PJRT/IFRT
  device execution in one process without GIL contention or a multi-gigabyte
  Python environment.
- **One portable artifact across hardware.** A single exported directory can
  carry lowerings for `cpu`, `cuda`, and `tpu`. Validate it locally on CPU and
  deploy the same bundle to GPU or TPU containers.
- **Serving policy lives with the model.** Dynamic batching, warmup requests,
  host-side feature pipelines, and optional ahead-of-time (AOT) compiled
  executables are packaged inside the bundle rather than scattered across
  per-server flags.

XSE is built for **recommenders and rankers**, **vision and multimodal
encoders**, **encoder-style language models**, and **short fixed-horizon
decoding** (such as beam search over a few tokens), where request latency and
cost per query dominate. It is not a continuous-batching LLM engine; for
open-ended long-form text generation with a paged KV cache, use
[vLLM](https://github.com/vllm-project/vllm) (with
[vllm-jax](https://github.com/vllm-project/vllm-jax) or
[vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) on TPUs) or
[SGLang](https://github.com/sgl-project/sglang) (with
[sglang-jax](https://github.com/sgl-project/sglang-jax) or
[sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu) on TPUs).

## Quick look

Export a JAX model in Python:

```python
import xla_saved_model

xla_saved_model.save(
    xla_saved_model.JaxModel(
        apply_fn_bundle=xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,                              # (params, inputs) -> outputs
            input_polymorphic_shapes={'features': 'b, _'},  # symbolic batch dimension
            jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
        ),
        params=params,
    ),
    export_path='/tmp/classifier',
    serving_configs=[xla_saved_model.ServingConfig(input_signature=input_signature)],
)
```

Serve it over gRPC with a prebuilt container image:

```bash
docker run \
  --net=host \
  --mount type=bind,src=/tmp/classifier,dst=/models/classifier,readonly \
  us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest \
  --model_name=classifier \
  --model_path=/models/classifier \
  --orbax_grpc_server_platform=cpu \
  --use_insecure_grpc_credentials=true
```

The server exposes `PredictionService/Predict` and
`PredictionService/GetModelMetadata` on port 8500. Walk through the full
example—including an in-process Python runner and a gRPC client—in the
[quickstart](docs/quickstart.md).

## What is in the repository

- **`xla_saved_model`** — pure-Python export library and the `xsm-cli` CLI for
  inspecting bundles, tuning batching post-export, and ahead-of-time compiling
  executables.
- **`xla_saved_model_runner`** — Python bindings (`ModelRunner`) around the C++
  `Interpreter` for validating bundles in-process on CPU.
- **`xla_serving_data`** — C++ feature transforms and Python bindings for
  high-throughput host preprocessing without TensorFlow.
- **`runtime/`** — the C++ `Interpreter`, batch scheduler, kernel library, and
  gRPC server binary (`//runtime/serving/gcp/server:xla_serving_main_{cpu,gpu,tpu}`).
- **`examples/`** — runnable exports for Flax NNX, ViT (Keras 3), DLRM with
  embeddings, PyTorch via `torchax`, C++ composite preprocessors, AOT
  compilation benchmarks, and Qwen3 beam search.

## Installation

Install `xla-saved-model` with `pip` (until the PyPI package is live, pass
`--find-links` pointing at the wheels from
[GitHub Releases](https://github.com/openxla/xla-serving/releases)):

```bash
gh release download \
  --repo openxla/xla-serving \
  --pattern '*.whl' \
  --dir wheels

pip install \
  --find-links wheels \
  xla-saved-model
```

Prebuilt server images are published to
`us-docker.pkg.dev/xla-serving-engine/xla-serving/` for CPU, CUDA 12, CUDA 13,
and Cloud TPU. See [Installation](docs/installation.md) for optional extras
(`[tf]`, `[torch]`, `[llm]`), container tags, and building from source with
Bazel.

## Documentation

- **Start:** [What XLA Serving Engine is](docs/overview.md) ·
  [Concepts](docs/concepts.md) · [Installation](docs/installation.md) ·
  [Quickstart](docs/quickstart.md)
- **Guides:** [Exporting models](docs/guides/exporting.md) ·
  [Serving and runtime](docs/guides/serving.md) ·
  [Pre- and post-processing](docs/guides/processing.md) ·
  [Batching](docs/guides/batching.md) · [Warmup](docs/guides/warmup.md) ·
  [Ahead-of-time compilation](docs/guides/aot.md) ·
  [`xsm-cli`](docs/guides/xsm_cli.md)

To build the HTML documentation locally:

```bash
pip install -r docs/requirements.txt
sphinx-build \
  -W \
  -b html \
  docs \
  docs/_build/html
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Licensed under
[Apache 2.0](LICENSE).

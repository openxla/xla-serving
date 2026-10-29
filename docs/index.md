---
html_theme.sidebar_secondary.remove: true
---

# XLA Serving Engine

:::{div} xse-hero xse-tagline
A C++ inference runtime and model packaging toolkit for the OpenXLA ecosystem.
Export once from JAX or PyTorch (`torchax`); serve on CPU, NVIDIA GPU, or Cloud
TPU from a single bundle, with no Python in the request path.
:::

XLA Serving Engine (XSE) takes models trained in JAX—or lowered to JAX from PyTorch
via [torchax](https://github.com/pytorch/xla/tree/master/torchax)—into
production without rewriting them or converting them to another graph format.

On the **export** side, `xla_saved_model` packages your model's StableHLO
computation, its [Orbax](https://github.com/google/orbax) weight checkpoint,
and any host-side pre- and post-processing into a self-contained directory
called an **XLA SavedModel**. On the **serving** side, a standalone C++ runtime
loads that directory, compiles it for the attached hardware via PJRT (or loads
a precompiled executable), dynamically batches incoming requests, and serves
predictions over gRPC or in-process.

XSE is designed for **recommenders and rankers**, **vision and multimodal
encoders**, **encoder-style language models**, and **short fixed-horizon
generation** where tail latency and hardware efficiency per request matter. It
is not a continuous-batching chat engine; for open-ended autoregressive LLM
serving with a paged KV cache, use [vLLM](https://github.com/vllm-project/vllm)
(with [vllm-jax](https://github.com/vllm-project/vllm-jax) or
[vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) on TPUs) or
[SGLang](https://github.com/sgl-project/sglang) (with
[sglang-jax](https://github.com/sgl-project/sglang-jax) or
[sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu) on TPUs).

::::{grid} 1 1 3 3
:gutter: 3

:::{grid-item-card} 1. Export
:link: guides/exporting
:link-type: doc

Package a JAX function, Flax module, Keras model, or PyTorch module into a
portable bundle with symbolic batch dimensions.
:::

:::{grid-item-card} 2. Serve
:link: guides/serving
:link-type: doc

Run a prebuilt container on CPU, CUDA, or Cloud TPU, validate in-process from
Python with `ModelRunner`, or embed the C++ `Interpreter`.
:::

:::{grid-item-card} 3. Tune for production
:link: guides/batching
:link-type: doc

Configure dynamic batching, host-side preprocessing, startup warmup, and
ahead-of-time (AOT) compilation directly in the bundle.
:::
::::

## Thirty-second tour

Export a model in Python:

```python
import xla_saved_model

xla_saved_model.save(
    xla_saved_model.JaxModel(
        apply_fn_bundle=xla_saved_model.ApplyFnBundle(
            apply_fn=apply_fn,                              # (params, inputs) -> outputs
            input_polymorphic_shapes={'features': 'b, _'},  # batch dim stays symbolic
            jax_export_kwargs={'platforms': ['cpu', 'cuda', 'tpu']},
        ),
        params=params,
    ),
    export_path='/tmp/classifier',
    serving_configs=[xla_saved_model.ServingConfig(input_signature=input_signature)],
)
```

Start the gRPC server container:

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

The server answers `PredictionService/Predict` and
`PredictionService/GetModelMetadata` on port 8500. The
[quickstart](quickstart.md) runs this end to end with a working model, an
in-process check, and a Python gRPC client.

## Where to go next

- **Exploring XSE?** Start with [What XLA Serving Engine is](overview.md) for why XSE
  exists, which workloads it fits, and how the runtime is structured, then read
  [Concepts](concepts.md) for the five core terms.
- **Ready to try it?** Follow [Installation](installation.md) and the
  ten-minute [Quickstart](quickstart.md).
- **Building a pipeline?** Read [Exporting models](guides/exporting.md),
  [Serving and runtime](guides/serving.md), and
  [Pre- and post-processing](guides/processing.md).
- **Going to production?** Configure [Batching](guides/batching.md),
  [Warmup](guides/warmup.md), and
  [Ahead-of-time compilation](guides/aot.md), and inspect or update bundles
  with [`xsm-cli`](guides/xsm_cli.md).

```{toctree}
:hidden:
:caption: Start

overview
concepts
installation
quickstart
```

```{toctree}
:hidden:
:caption: Guides

guides/exporting
guides/serving
guides/processing
guides/batching
guides/warmup
guides/aot
guides/xsm_cli
```

```{toctree}
:hidden:
:caption: Project

GitHub <https://github.com/openxla/xla-serving>
Releases <https://github.com/openxla/xla-serving/releases>
```

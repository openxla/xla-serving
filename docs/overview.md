# What XLA Serving Engine is

XLA Serving Engine (XSE) is a C++ inference server and runtime for models that compile
with XLA. You export a model in Python with `xla_saved_model.save()` and get a
self-contained directory—an **XLA SavedModel**—that XSE serves over gRPC or
in-process on CPU, NVIDIA GPU, or Cloud TPU.

## Why XLA Serving Engine?

When a model trained in JAX (or PyTorch on XLA) is ready for production, teams
usually face two imperfect options:

1. **Wrap `jax.jit` in a Python server (FastAPI, Ray Serve, Python gRPC).**
   Simple to start, hard to scale. Every serving replica ships a full Python
   and JAX environment; concurrent requests and host-side feature parsing
   contend on the Python GIL; you end up writing your own dynamic batching
   loop; and every pod restart pays XLA compilation latency on live traffic.
2. **Convert the model to another format (ONNX, TensorRT, or TensorFlow via
   `jax2tf`).** Translation passes work for standard layers, but custom XLA
   calls, Pallas kernels, multi-device sharding annotations, or mixed-precision
   numerics often break or require maintaining a separate inference model
   definition per hardware backend.

XLA Serving Engine takes a third route: **use XLA's native portable IR end to end.**

- **Native `jax.export` artifact.** `xla_saved_model` lowers your function with
  `jax.export` into [StableHLO](https://openxla.org/stablehlo) bytecode and
  saves parameters as an [Orbax](https://github.com/google/orbax) checkpoint.
  If `jax.jit` can compile your function, XSE can serve it—custom calls,
  Pallas kernels, and sharding included.
- **Pure C++ serving path.** The runtime is a C++ binary built on OpenXLA's
  PJRT and IFRT device interfaces. Request handling, dynamic batching,
  host-side preprocessing, and device execution all run in C++ thread pools
  with no Python interpreter in the container.
- **The bundle is the deployment contract.** Everything the server needs to
  know—the computation, weights, target platforms, dynamic batching rules,
  host-side pre- and post-processing stages, warmup requests, and optional
  ahead-of-time compiled executables—lives inside the exported directory. The
  server binary itself only takes flags for ports, credentials, and the path to
  the bundle.

## Use cases

XSE is built for workloads where tail latency, predictable startup, and
request-level throughput per chip matter:

- **Recommenders, ranking, and tabular models.** Two-tower retrieval, DLRM, and
  wide-and-deep ranking models combine memory-bound feature lookups (string
  hashing, bucketization, large embedding tables on host CPU or TPU embedding
  cores) with compute-bound dense networks on the accelerator. XSE runs both in
  a single orchestrated pipeline so embeddings stay in host RAM while the MLP
  or transformer batches on the GPU or TPU. See `examples/dlrm`,
  `examples/nnx_jax_preprocessor`, and `examples/math_preprocessor`.
- **Vision and multimodal encoders.** Image classifiers, ViT backbones, and
  embedding extractors that decode raw inputs on the host and batch fixed-shape
  tensors onto the accelerator. See `examples/vit`.
- **Encoder language models and bounded-horizon scorers.** BERT/T5-style
  classification, reranking, embedding generation, and short fixed-horizon
  decoding (such as beam search over a handful of output tokens). When
  generation is only a few tokens long, compiling the entire prefill-and-decode
  loop into a single static XLA program avoids per-token host scheduling
  overhead. See `examples/llm`.

### When to use something else

XSE is **not** a continuous-batching engine for open-ended autoregressive LLM
generation (chatbots, long-form text or code generation). It does not manage a
paged KV cache across in-flight requests of varying lengths. For those
workloads, use [vLLM](https://github.com/vllm-project/vllm) (with
[vllm-jax](https://github.com/vllm-project/vllm-jax) or
[vllm-torchtpu](https://github.com/vllm-project/vllm-torchtpu) on TPUs) or
[SGLang](https://github.com/sgl-project/sglang) (with
[sglang-jax](https://github.com/sgl-project/sglang-jax) or
[sglang-torchtpu](https://github.com/sgl-project/sglang-torchtpu) on TPUs).

## Key capabilities

Portable multi-platform bundles
: A function exported with `platforms=['cpu', 'cuda', 'tpu']` carries StableHLO
  lowerings for all three targets in one directory. Export on a CPU workstation
  without an accelerator attached, test the bundle in CI on CPU, and deploy the
  exact same artifact to GPU or Cloud TPU.

Dynamic request batching
: The runtime's batch scheduler groups concurrent client requests along the
  batch dimension, pads to an allowed batch size, executes once on the
  accelerator, and splits responses back to each caller. Batch sizes, timeouts,
  and padding policies live in `ServingConfig.batch_options` and can be tuned
  after export with `xsm-cli update`. See [Batching](guides/batching.md).

Host-side pre- and post-processing
: A serving signature can chain host stages before and after the accelerator
  call: TensorFlow `tf.function`s (executed via TFRT), additional JAX functions
  lowered for CPU, or custom C++ transforms built with `xla_serving_data`. See
  [Pre- and post-processing](guides/processing.md).

Warmup and ahead-of-time (AOT) compilation
: Record representative inputs at export (`warmup_examples=`) and pass
  `--warmup_server=true` so the server compiles and exercises every pipeline
  stage before accepting traffic. To eliminate compilation on serving nodes
  entirely, run `xsm-cli recompile` in your build pipeline to embed a
  target-specific PJRT executable inside the bundle, dropping cold start from
  tens of seconds to milliseconds. See [Warmup](guides/warmup.md) and
  [Ahead-of-time compilation](guides/aot.md).

## Architecture

```{mermaid}
flowchart TD
    subgraph export ["1. Export (Python)"]
        direction TB
        J["JAX / Flax / Keras / PyTorch"] --> S["xla_saved_model.save()"]
        T["TF functions (optional)"] --> S
        C["C++ transforms (optional)"] --> S
    end

    subgraph bundle ["2. XLA SavedModel bundle (on disk)"]
        direction TB
        M["manifest.pb (StableHLO + index)"] --> O["orchestration.pb"]
        M --> K["checkpoint/ (Orbax)"]
        M --> X["tf_saved_model/, assets, AOT"]
    end

    subgraph runtime ["3. Runtime (C++)"]
        direction LR
        G["gRPC PredictionService<br/>Python ModelRunner<br/>C++ API"] --> I["Interpreter<br/>pre → batch → StableHLO → post"] --> D["PJRT / IFRT<br/>CPU · CUDA · TPU"]
    end

    export --> bundle --> runtime
```

1. **Export (`xla_saved_model`).** Traces your model with `jax.export` using
   symbolic batch dimensions, writes the weights as an Orbax checkpoint, and
   records an `orchestration.pb` describing the pipeline stages and batching
   rules. If your pipeline includes TensorFlow functions or C++ transforms,
   their artifacts are saved alongside the checkpoint.
2. **Bundle (`XLA SavedModel`).** A plain directory on disk or object storage
   indexed by `manifest.pb`. Inspect it, dump its StableHLO MLIR, or update its
   batching config from any machine with `xsm-cli`.
3. **Runtime (`Interpreter` & gRPC server).** At startup the C++ `Interpreter`
   reads `manifest.pb`, constructs the execution graph for each serving
   signature, loads the checkpoint onto the target device via IFRT/PJRT, and
   compiles the StableHLO (or maps an embedded AOT executable). The gRPC server
   exposes `PredictionService/Predict` and `PredictionService/GetModelMetadata`
   on top of `Interpreter::RunAsyncWithMapReturn`. The Python
   `xla_saved_model_runner` package wraps the same `Interpreter` for local
   in-process validation.

## Relationship to the ecosystem

[JAX](https://github.com/jax-ml/jax) and [StableHLO](https://openxla.org/stablehlo)
: Export is built directly on `jax.export`. Symbolic shape polymorphism (`'b, _'`)
  is how a single exported function accepts arbitrary batch sizes at runtime.

[Orbax](https://github.com/google/orbax)
: Weights are stored as standard Orbax checkpoints under `checkpoint/`, and the
  bundle format shares conventions with Orbax Model (`orbax_model_version.txt`).

[PyTorch](https://pytorch.org) via [torchax](https://github.com/pytorch/xla/tree/master/torchax)
: `torchax` traces a `torch.nn.Module` into a JAX function and state dict,
  which `xla_saved_model` exports through the same path as native JAX models.
  See [Exporting models](guides/exporting.md#exporting-from-different-frameworks).

[TensorFlow Serving](https://github.com/tensorflow/serving)
: The `PredictionService` gRPC structure, dynamic batch scheduler options, and
  warmup file format follow familiar TF Serving patterns, adapted for StableHLO
  and flat `HostBufferProto` payloads.

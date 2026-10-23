# Installation

XLA Serving Engine consists of three Python packages and a set of prebuilt server
container images. Most users need `xla-saved-model` on the machine that exports
models and one container image where the model is served.

| Component | What it is | When you need it |
|---|---|---|
| `xla-saved-model` | Pure-Python export library and the `xsm-cli` CLI | Always, wherever you export or inspect models |
| `xla-saved-model-runner` | Python bindings (`ModelRunner`) around the C++ `Interpreter` | Optional, to run and validate bundles in-process from Python (CPU in the open-source build) |
| `xla-serving-data` | C++ feature transforms and Python bindings | Optional, only if you author or inspect C++ `CompositeModel` stages |
| Server images | `us-docker.pkg.dev/xla-serving-engine/xla-serving/*` | To serve exported bundles over gRPC on CPU, GPU, or TPU |

## Requirements

- **OS:** Linux on `x86_64` (binary wheels are `manylinux_2_28`; `aarch64` can
  be built from source).
- **Python:** 3.12, 3.13, or 3.14. Binary wheels are built for these three
  versions; `xla-saved-model` itself is pure Python (`>=3.10`) and tested on
  3.12+.
- **NVIDIA GPU (for serving):** An NVIDIA driver supporting CUDA 12 or 13, and
  the [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/)
  when running the GPU container images.
- **Cloud TPU (for serving):** A Cloud TPU VM. The `tpu-jax-only` container
  image bundles `libtpu.so` out of the box; you only need a local `libtpu`
  installation if you run a bare server binary built from source.

Exporting a model does **not** require a GPU or TPU on the export machine. JAX
lowers functions to StableHLO for `cpu`, `cuda`, and `tpu` on any CPU host.

## Python packages

:::{note}
PyPI publishing for `xla-saved-model`, `xla-saved-model-runner`, and
`xla-serving-data` is being finalized. Once published, installing the export
package will be a standard `pip install xla-saved-model`. Until then, download
the wheel artifacts from [GitHub Releases](https://github.com/AI-Hypercomputer/xla-serving/releases)
and pass `--find-links wheels` to `pip`.
:::

Download the wheels from the latest release into `./wheels`:

```bash
python3.12 -m venv .venv
source .venv/bin/activate

gh release download \
  --repo AI-Hypercomputer/xla-serving \
  --pattern '*.whl' \
  --dir wheels
```

Then install `xla-saved-model` (once on PyPI, omit `--find-links wheels`):

```bash
# Core export library and xsm-cli
pip install \
  --find-links wheels \
  xla-saved-model

# Optional: in-process C++ runtime wrapper for Python (picks the wheel for your Python version)
pip install \
  --find-links wheels \
  xla-saved-model-runner

# Optional: C++ feature transform library
pip install \
  --find-links wheels \
  xla-serving-data
```

`xla-saved-model` defines optional extras for the frameworks it can export:

```bash
# TensorFlow pre/post-processing stages
pip install \
  --find-links wheels \
  "xla-saved-model[tf]"

# PyTorch models via torchax
pip install \
  --find-links wheels \
  "xla-saved-model[torch]"

# Hugging Face LLM export (xsm-llm-export)
pip install \
  --find-links wheels \
  "xla-saved-model[llm]"
```

Verify the installation:

```bash
python -c "import xla_saved_model; print(xla_saved_model.__version__)"
xsm-cli --help
```

## Server container images

Prebuilt server images are published to
`us-docker.pkg.dev/xla-serving-engine/xla-serving/`. Each platform has a lean
JAX-only image and (for CPU and GPU) a `-with-tf` variant that compiles in
TensorFlow support. Use `-with-tf` only when your exported bundle includes a
`TensorflowModel` pre- or post-processing stage.

| Image | `--orbax_grpc_server_platform` | Notes |
|---|---|---|
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only` | `cpu` | Smallest image. Ideal for local development, CI, and CPU inference. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-with-tf` | `cpu` | Adds TensorFlow (TFRT) support for `TensorflowModel` stages. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12` | `cuda` | Built against CUDA 12.9, cuDNN 9.8. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12-with-tf` | `cuda` | CUDA 12.9 + TensorFlow support. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda13` | `cuda` | Built against CUDA 13.0, cuDNN 9.12. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda13-with-tf` | `cuda` | CUDA 13.0 + TensorFlow support. |
| `us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only` | `tpu` | Cloud TPU image with `libtpu.so` bundled at `/usr/lib/libtpu.so`. |

Available tags:

- `latest` — most recent release
- `nightly` — built nightly from `main`
- `nightly-YYYYMMDD` — pinned nightly snapshot

```bash
docker pull us-docker.pkg.dev/xla-serving-engine/xla-serving/cpu-jax-only:latest
```

The container entrypoint is the server binary (`xla_serving_main_*`), so all
arguments after the image name are passed directly as server flags. See
[Serving and runtime](guides/serving.md) for the complete flag reference.

:::{admonition} Keep the exporter and the server in step
:class: tip
The bundle format is versioned, and the server rejects bundles produced by a
newer incompatible exporter rather than guessing. If startup exits with an
error such as `Orchestration: only mime type ... is supported, but got ...`,
your container image is older than the `xla-saved-model` wheel that exported
the bundle—pull a newer image tag.
:::

## Building from source

To build the C++ server or native wheels from source, you need
[Bazel](https://bazel.build) 8.7 (the repo includes `.bazelversion`, so
[Bazelisk](https://github.com/bazelbuild/bazelisk) selects it automatically).
The build downloads a hermetic Clang toolchain.

```bash
git clone https://github.com/AI-Hypercomputer/xla-serving
cd xla-serving
```

### Server binaries

```bash
# CPU
bazel build \
  -c opt \
  //runtime/serving/gcp/server:xla_serving_main_cpu

# NVIDIA GPU (choose CUDA 12 or CUDA 13)
bazel build \
  -c opt \
  --config=cuda_v12 \
  //runtime/serving/gcp/server:xla_serving_main_gpu

bazel build \
  -c opt \
  --config=cuda_v13 \
  //runtime/serving/gcp/server:xla_serving_main_gpu

# Cloud TPU
bazel build \
  -c opt \
  //runtime/serving/gcp/server:xla_serving_main_tpu
```

Add `--define ENABLE_TF=1` to any command above to compile in TensorFlow
support for `TensorflowModel` stages.

The compiled binary is placed at
`bazel-bin/runtime/serving/gcp/server/xla_serving_main_<platform>` and accepts
the same flags as the container images. To run the C++ test suite:

```bash
bazel test //runtime/... //data/...
```

### Container images

The `xla_serving_pusher*` targets build an OCI image and push it to your own
registry:

```bash
bazel run \
  -c opt \
  //runtime/serving/gcp/server:xla_serving_pusher \
  -- \
  --repository=us-docker.pkg.dev/my-project/xse/cpu \
  --tag=dev
```

Use `xla_serving_pusher_gpu` (with `--config=cuda_v12` or `--config=cuda_v13`)
for GPU and `xla_serving_pusher_tpu` for TPU. Add `--define ENABLE_TF=1` for
the TensorFlow-enabled variants.

### Python wheels

`xla-saved-model` is a standard `pyproject.toml` package whose build step invokes
`bazel` to generate Python protobuf modules:

```bash
# Editable install for development
pip install -e ".[tf,test]"

# Run Python unit tests
pytest xla_saved_model/

# Build pure-Python wheel into dist/xla_saved_model-*.whl
pip install build
python -m build --wheel
```

The two native wheels are built directly with Bazel for your target Python
version (`3.12`, `3.13`, or `3.14`):

```bash
bazel build \
  -c opt \
  --@rules_python//python/config_settings:python_version=3.12 \
  //runtime/python:xla_saved_model_runner

bazel build \
  -c opt \
  --@rules_python//python/config_settings:python_version=3.12 \
  //data:xla_serving_data

pip install bazel-bin/runtime/python/xla_saved_model_runner-*.whl
pip install bazel-bin/data/xla_serving_data-*.whl
```

A cold build compiles XLA from source and takes significant time. `.bazelrc`
defines `--config=remote_cache_readonly`, which reads from the public GCS build
cache populated by CI when your environment matches the CI toolchain.

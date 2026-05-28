# Neptune Serving Engine

[Neptune Serving Engine (NSE)](https://docs.cloud.google.com/tpu/docs/jax-ai-stack#jax_serving_orbax_serialization_and_neptune_serving_engine)
is a general-purpose inference runtime for the XLA-based (TPU) software
ecosystem. It is designed to be a modular infrastructure component that can be
plugged into a variety of higher abstraction serving solutions.

NSE is a generic serving solution for a diversity of deep learning models,
including computer vision, NLP, transformers, recommenders, etc.
It is not meant for high-throughput LLM serving.

**Why use NSE?**

- Battle-tested by major Google product areas including YouTube, Deepmind,
Waymo, and Ads. Designed for doing inference on TPUs.
- Capable of serving a diversity of architectures of models, including LEMs,
LRMs, encoder-heavy LLMs.
- Portable file format that can be loaded on GPUs and TPUs.
- Minimal dependencies (no Python required).

## Python Development

A `neptune_model` Python library provides an API for exporting models that can
run on NSE.

### Prerequisites

-   **Bazel**: The build process requires `bazel` (or `bazelisk`) to be
    installed and available in your system `PATH` to fetch or build XLA protocol
    buffers.

### Installation

To work on `neptune_model` you can install the package in editable mode with
development/test dependencies.

```bash
pip install -e .[tf,test]
```

Run tests using `pytest`:

```bash
pytest neptune_model/
```

Run example model exports with `python -m`:

```bash
python -m examples.nnx.export_test
python -m examples.nnx.run_export
```

Build and install a wheel with `python -m build`:

```bash
pip install --upgrade build
python -m build --wheel

pip install dist/neptune_model-*.whl

# Use [tf] tag if you want TensorFlow preprocessor support:

wheel=(dist/neptune_model-*.whl)
pip install "${wheel}[tf]"
```

## ModelRunner

ModelRunner allows quickly testing inference on an exported model from Python.

```bash
# Python 3.11
bazel build //runtime/python:neptune_model_runner --@rules_python//python/config_settings:python_version=3.11

# Python 3.12+
bazel build //runtime/python:neptune_model_runner --@rules_python//python/config_settings:python_version=3.12
```

Nanobind supports limited Python ABI for 3.12+, so you can use
`neptune_model_runner` built with 3.12 for any Python 3.12+.

Install it with:

```bash
# Replace *.whl with an actual version under your bazel-bin/.
pip install bazel-bin/runtime/python/neptune_model_runner-0.0.1-cp311-cp311-manylinux_2_28_x86_64.whl
```

Test the model:

```python
import numpy as np
from neptune_model_runner import model_runner

runner = model_runner.ModelRunner(model_path='/your/model')

inputs = {
    'feature_1': np.zeros((2, 1), dtype=np.int32),
    'feature_2': np.zeros((2, 1), dtype=np.int32),
    'feature_3': np.zeros((2, 1), dtype=np.int32),
    'feature_4': np.zeros((2, 1), dtype=np.int32),
}

outputs = runner.signature('serving_default').run_with_dict_return(**inputs)
```

## Docker images for GRPC Server

### CPU

You can use prebuilt images from Artifact Registry here:

```
gcr.io/neptune-serving-engine/cpu-jax-only:latest
gcr.io/neptune-serving-engine/cpu-jax-with-tf:latest
```
In order to build the image at head, please run:

```
bazel run -c opt \
//runtime/serving/gcp/server:neptune_serving_pusher_cpu \
-- --repository=<Path to Artifact registry> \
--tag=latest
```

To build with Tensorflow dependencies

```
bazel run -c opt --define ENABLE_TF=1 \
//runtime/serving/gcp/server:neptune_serving_pusher_cpu \
-- --repository=<Path to Artifact registry> --tag=latest
```

### GPU

We currently support building GPU image with CUDA 12.

The prerequisite is to have CUDA toolkit and drivers 
already installed on the Cloud VM.

To build docker image for GPU:

```
bazel run -c opt --config=cuda_v12 \
//runtime/serving/gcp/server:neptune_serving_pusher_gpu \
-- --repository="<Path to Artifact registry>" \
--tag="latest"
```

### TPU

To Be Added.

## Starting GRPC Servers

### Using Docker images

#### CPU
```
docker run   --net=host   --ulimit memlock=-1  \
--mount type=bind,src=<exported model>,dst=/tmp/model \
 -e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
 gcr.io/neptune-serving-engine/cpu-jax-only:latest  \
 --model_path /tmp/model  \
 --model_name nnx \
 --orbax_grpc_server_platform cpu \
 --port=8501 
 --use_insecure_grpc_credentials=true
```

#### GPU
```
docker run --gpus all --net=host -p 8500:8500 \
--ulimit memlock=-1 \
--mount type=bind,src=<Exported model>,dst=/tmp/model \
--mount type=bind,src=/tmp/profile-data,dst=/tmp/ \
profile-data \
-e PJRT_DEVICE=cuda \
-e NVIDIA_VISIBLE_DEVICES=all \
-e NVIDIA_DRIVER_CAPABILITIES=all,compute,utility \
-e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64 \
-e XLA_PYTHON_CLIENT_MEM_FRACTION=0.85 \
gcr.io/neptune-serving-engine/gpu-jax-cuda12:latest \
--model_path /tmp/model \
--model_name nnx \
--orbax_grpc_server_platform cuda \
--use_insecure_grpc_credentials=true
```

### Running locally built server

```
bazel run //runtime/serving/gcp/server:neptune_serving_main_<cpu|gpu|tpu> \
--model_path <Exported model path> \
--model_name <model_name> \
--orbax_grpc_server_platform <cpu|cuda|tpu>  \
--port 8500 --use_insecure_grpc_credentials=<true|false>
```

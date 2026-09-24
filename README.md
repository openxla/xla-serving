# XLA Serving Engine

[XLA Serving Engine (XSE)](https://docs.cloud.google.com/tpu/docs/jax-ai-stack#jax_serving_orbax_serialization_and_neptune_serving_engine)
is a general-purpose inference runtime for the XLA-based (TPU) software
ecosystem. It is designed to be a modular infrastructure component that can be
plugged into a variety of higher abstraction serving solutions.

XSE is a generic serving solution for a diversity of deep learning models,
including computer vision, NLP, transformers, recommenders, etc.
It is not meant for high-throughput LLM serving.

**Why use XSE?**

- Battle-tested by major Google product areas including YouTube, Deepmind,
Waymo, and Ads. Designed for doing inference on TPUs.
- Capable of serving a diversity of architectures of models, including LEMs,
LRMs, encoder-heavy LLMs.
- Portable file format that can be loaded on GPUs and TPUs.
- Minimal dependencies (no Python required).

## Python Development

A `xla_saved_model` Python library provides an API for exporting models that can
run on XSE.

### Prerequisites

-   **Bazel**: The build process requires `bazel` (or `bazelisk`) to be
    installed and available in your system `PATH` to fetch or build XLA protocol
    buffers.

### Installation

To work on `xla_saved_model` you can install the package in editable mode with
development/test dependencies.

```bash
pip install -e .[tf,test]
```

Run tests using `pytest`:

```bash
pytest xla_saved_model/
```

Run example model exports with `python -m`:

```bash
python -m examples.nnx.export_test
python -m examples.nnx.run_export --export_path=/path/to/export_dir
```

Build and install a wheel with `python -m build`:

```bash
pip install --upgrade build
python -m build --wheel

pip install dist/xla_saved_model-*.whl

# Use [tf] tag if you want TensorFlow preprocessor support:

wheel=(dist/xla_saved_model-*.whl)
pip install "${wheel}[tf]"
```

## ModelRunner

ModelRunner allows quickly testing inference on an exported model from Python.

```bash
# Python 3.11
bazel build //runtime/python:xla_saved_model_runner --@rules_python//python/config_settings:python_version=3.11

# Python 3.12+
bazel build //runtime/python:xla_saved_model_runner --@rules_python//python/config_settings:python_version=3.12
```

Nanobind supports limited Python ABI for 3.12+, so you can use
`xla_saved_model_runner` built with 3.12 for any Python 3.12+.

Install it with:

```bash
# Replace *.whl with an actual version under your bazel-bin/.
pip install bazel-bin/runtime/python/xla_saved_model_runner-0.0.1-cp311-cp311-manylinux_2_28_x86_64.whl
```

Test the model:

```python
import numpy as np
from xla_saved_model_runner import model_runner

runner = model_runner.ModelRunner(model_path='/your/model')

inputs = {
    'feature_1': np.zeros((2, 1), dtype=np.int32),
    'feature_2': np.zeros((2, 1), dtype=np.int32),
    'feature_3': np.zeros((2, 1), dtype=np.int32),
    'feature_4': np.zeros((2, 1), dtype=np.int32),
}

outputs = runner.signature('serving_default').run_with_dict_return(**inputs)
```

## Ahead-Of-Time (AOT) Model Recompilation

XLA Serving supports Ahead-Of-Time precompilation to eliminate online JIT
latency and protect host container memory. See the
[AOT User Guide](docs/aot_user_guide.md) and
[AOT Benchmark](examples/aot_benchmark/README.md) for full details.

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
//runtime/serving/gcp/server:xla_serving_pusher \
-- --repository=<Path to Artifact registry> \
--tag=latest
```

To build with Tensorflow dependencies

```
bazel run -c opt --define ENABLE_TF=1 \
//runtime/serving/gcp/server:xla_serving_pusher \
-- --repository=<Path to Artifact registry> --tag=latest
```

### GPU

We currently support building GPU image with CUDA 12.

The prerequisite is to have CUDA toolkit and drivers 
already installed on the Cloud VM.

Prebuilt GPU image
```
gcr.io/neptune-serving-engine/gpu-jax-cuda12
```

To build docker image for GPU:

```
bazel run -c opt --config=cuda_v12 \
//runtime/serving/gcp/server:xla_serving_pusher_gpu \
-- --repository="<Path to Artifact registry>" \
--tag="latest"
```

### TPU

Prebuilt image for TPU

```
gcr.io/neptune-serving-engine/tpu-jax-only
```


To build docker image for GPU:

```
bazel run -c opt --config=cuda_v12 \
//runtime/serving/gcp/server:xla_serving_pusher_tpu \
-- --repository="<Path to Artifact registry>" \
--tag="latest"
```

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
 --port=8500 \
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

You can also use partial GPUs with Multi Instance Group(MIG). In order to start
docker with one of the MIG GPUs, pass the following parameters in the above
command:
```
--gpu '"device=<mig_id>" # instead of `--gpu all`
-e NVIDIA_VISIBLE_DEVICES=<mig_id>  # instead of NVIDIA_VISIBLE_DEVICES=all
```

### Running locally built server

#### CPU
```bash
bazel run //runtime/serving/gcp/server:xla_serving_main_cpu -- \
  --model_path <Exported model path> \
  --model_name <model_name> \
  --orbax_grpc_server_platform <cpu|cuda|tpu>  \
  --port 8500 --use_insecure_grpc_credentials=<true|false>
```

#### GPU and TPU
```bash
bazel run //runtime/serving/gcp/server:xla_serving_main_<gpu|tpu> -- \
  --model_path <Exported model path> \
  --model_name <model_name> \
  --orbax_grpc_server_platform <cpu|cuda|tpu>  \
  --port 8500 --use_insecure_grpc_credentials=<true|false>
```

### Server Warmup on Startup

The server can optionally run warmup requests on startup to trigger XLA compilation before serving live traffic. This is disabled by default.

Running warmup requests avoids "cold start" latency spikes for the first few client requests by pre-compiling the XLA graphs, ensuring the server is ready to serve with predictable latency immediately after it starts reporting healthy.

#### Prerequisites
The model directory must contain a pre-generated warmup file at `assets.extra/xla_serving_warmup_requests`.

This file can be generated during model export by passing warmup examples to `xla_saved_model.save(..., warmup_examples=...)`.

Note that the warmup examples must follow the input specification (input names, shapes, and data types) of the corresponding serving signature.

Here is an example of exporting a model with warmup requests:

```python
import numpy as np
import xla_saved_model

# Assume you have a JaxModel instance `model` and your ServingConfig `serving_config`
model = ...
serving_config = xla_saved_model.ServingConfig(
    serving_signature="serving_default",
    # ... other config options ...
)

# Prepare warmup examples.
# The keys must match the serving signatures defined in your serving configs.
# Each example in the sequence is a dictionary mapping input names to numpy arrays.
# The inputs (names, shapes, and dtypes) must exactly match the input signature of the serving pipeline.
warmup_examples = {
    "serving_default": [
        # Example request 1 (e.g. batch size 1)
        {
            "x": np.zeros((1, 224, 224, 3), dtype=np.float32),
        },
    ],
}

xla_saved_model.save(
    model=model,
    export_path="/path/to/export_dir",
    serving_configs=[serving_config],
    warmup_examples=warmup_examples,
)
```

#### Running with Warmup
To enable warmup, set `--warmup_server=true`. Note that if the warmup file is missing, the server startup will fail.

```bash
bazel run //runtime/serving/gcp/server:xla_serving_main_cpu -- \
  --model_path <Exported model path> \
  --model_name <model_name> \
  --orbax_grpc_server_platform <cpu|cuda|tpu> \
  --warmup_server=true
```

#### Disabling Warmup (Default)
By default, warmup is disabled. You can also explicitly disable it with `--warmup_server=false`:

```bash
bazel run //runtime/serving/gcp/server:xla_serving_main_cpu -- \
  --model_path <Exported model path> \
  --model_name <model_name> \
  --orbax_grpc_server_platform <cpu|cuda|tpu> \
  --warmup_server=false
```

### Concurrent Execution on CPU

The PjRt CPU client runs one XLA program at a time per execution stream, and
by default every execution uses stream 0. On the `cpu` platform this means all
batches funnel through a single queue no matter how many batch threads the
model configures: batch threads block in `StableHloRunner::Execute` waiting for
their turn, and throughput caps at one batch per program execution time.

`--num_execution_streams=N` spreads consecutive executions round-robin across
N IFRT execution streams. Executions on different streams may run
concurrently, so up to N batches execute at once on the same device with a
single copy of the weights. Batches are independent, so this is safe for
serving. Start with 4 on a 16 vCPU host and keep the batch thread count near
N + 2; more threads only add queueing and oversubscribe the intra-op pool.

```bash
bazel run //runtime/serving/gcp/server:xla_serving_main_cpu -- \
  --model_path <Exported model path> \
  --model_name <model_name> \
  --orbax_grpc_server_platform cpu \
  --num_execution_streams=4
```

Each `StableHloRunner::Execute` trace event carries an `execution_stream_id`
argument, so a profile shows directly whether executions overlap. The flag has
little effect on `cuda` and `tpu`, where a device executes programs in stream
order on the hardware and the host does not block.

## XLA SavedModel Export Migration Guide

With the release of `TensorflowModel` and explicit orchestration pipelines, several APIs in `xla_saved_model` have been updated to support exporting multiple models (such as JAX models combined with TensorFlow preprocessing and postprocessing) and complex multi-stage pipelines.

### Key Changes

1. **Model Containers (`JaxModel` and `TensorflowModel`)**:
   - `xla_saved_model.save()` accepts a sequence of user-level model containers (`Sequence[Model] | Model`).
     - `JaxModel`: Encapsulates JAX functions and related shared variables (mesh, model parameters).
     - `TensorflowModel`: Encapsulates TensorFlow functions and trackable resources. Currently, at most one `TensorflowModel` instance is supported per export bundle.
   - When exporting multiple models, explicitly assign unique names to your model containers (e.g., `name='jax_model'`, `name='tf_model'`). If only a single model is exported, naming is optional as it will be assigned a default name.
   - Pass a sequence of models `[model1, model2, ...]` to `xla_saved_model.save()`.

2. **Explicit Orchestration (`ServingConfig.orchestration`)**:
   - Fixed processing pipeline structures (`processor.SequentialProcessing` and `TfProcessor`) are replaced by explicit `ServingConfig.orchestration` configurations using `ModelCall` objects.
   - Each `ModelCall` references a target model by `model_name` and specifies the function to execute via `method_key`.
   - Specify pipeline execution order by defining `orchestration=serving_config.Orchestration(model_calls=[...])`. Note that you only need to specify `role=orchestration_pb2.ROLE_MODEL` for the on-device function (typically the JAX function to be run on device); setting roles for other `ModelCall` entries is optional.

---

### Step-by-Step Migration Journey (TF + JAX Example)

Suppose you have an existing export script that uses JAX with TensorFlow preprocessing and postprocessing under the previous API:

```python
import xla_saved_model
from xla_saved_model import processor
from xla_saved_model import tf_processor

xla_saved_model.save(
    export_path=export_path,
    models=xla_saved_model.JaxModel(
        apply_fn_bundle=apply_fn_bundle,
        params=model_params,
    ),
    serving_configs=[
        xla_saved_model.ServingConfig(
            serving_signature='serving_default',
            input_signature=input_signature,
            processing=processor.SequentialProcessing(
                preprocessors=[
                    tf_processor.TfProcessor(
                        name='embedding_layer',
                        fn=embedding_layer,
                    )
                ],
                postprocessors=[
                    tf_processor.TfProcessor(
                        name='simple_postprocessor',
                        fn=simple_postprocessor,
                    )
                ],
                options=processor.ProcessingOptions(
                    tf_trackable_resources=[embedding_table]
                ),
            ),
        )
    ],
)
```

Follow these 4 steps to convert this script to the new API:

1. **Consolidate TensorFlow Processors into a `TensorflowModel` Container**:
   - Instead of creating individual `TfProcessor` objects inside `SequentialProcessing`, create a single `TensorflowModel` container holding your preprocessing and postprocessing functions.
   - Bind any required trackable resources (such as embedding tables) directly to `trackable_resources`:
     ```python
     tf_proc_model = xla_saved_model.TensorflowModel(
         name='tf_model',
         tf_functions={
             'preprocessing': embedding_layer,
             'postprocessing': simple_postprocessor,
         },
         trackable_resources=[embedding_table],
     )
     ```

2. **Assign a Unique Name to Your `JaxModel`**:
   - When exporting multiple models in a pipeline, add a `name='jax_model'` parameter to your `JaxModel` definition so that pipeline steps can reference it by name (if exporting only a single model, naming is optional as it will use a default name).

3. **Replace `SequentialProcessing` with Explicit `Orchestration`**:
   - In your `ServingConfig`, remove `processing=processor.SequentialProcessing(...)`.
   - Define `orchestration=xla_saved_model.serving_config.Orchestration(model_calls=[...])`, where each `ModelCall` references a model by `model_name`, specifies the function via `method_key`, and assigns `role=orchestration_pb2.ROLE_MODEL` (other role types are not required).

4. **Pass the Sequence of Models to `save()`**:
   - Update `xla_saved_model.save()` to take the list of model containers: `[tf_proc_model, jax_model]`.

---

### Converted Code (New API)

After applying the steps above, the converted script using `TensorflowModel` and `Orchestration` is:

```python
import xla_saved_model
from xla_saved_model.protos import orchestration_pb2

# 1. Bundle TensorFlow pre/post-processing functions and trackable resources
tf_proc_model = xla_saved_model.TensorflowModel(
    name='tf_model',
    tf_functions={
        'preprocessing': embedding_layer,
        'postprocessing': simple_postprocessor,
    },
    trackable_resources=[embedding_layer],
)

# 2. Define your JAX model with a unique name
jm = xla_saved_model.JaxModel(
    apply_fn_bundle=apply_fn_bundle,
    params=model_params,
    name='jax_model',
)

# 3. Pass all models as a sequence and explicitly define the orchestration pipeline
xla_saved_model.save(
    [tf_proc_model, jm],
    export_path=export_path,
    serving_configs=[
        xla_saved_model.ServingConfig(
            serving_signature='serving_default',
            input_signature=input_signature,
            orchestration=xla_saved_model.serving_config.Orchestration(
                model_calls=[
                    xla_saved_model.serving_config.ModelCall(
                        model_name='tf_model',
                        method_key='preprocessing',
                    ),
                    xla_saved_model.serving_config.ModelCall(
                        model_name='jax_model',
                        method_key='serving_default',
                        role=orchestration_pb2.ROLE_MODEL,
                    ),
                    xla_saved_model.serving_config.ModelCall(
                        model_name='tf_model',
                        method_key='postprocessing',
                    ),
                ]
            ),
        )
    ],
)
```


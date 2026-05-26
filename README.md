# Neptune Serving Engine

Neptune Serving Engine (NSE) is a general-purpose inference runtime for the
XLA-based (TPU) software ecosystem. It is designed to be a modular
infrastructure component that can be plugged into a variety of higher
abstraction serving solutions.

https://docs.cloud.google.com/tpu/docs/jax-ai-stack#jax_serving_orbax_serialization_and_neptune_serving_engine

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
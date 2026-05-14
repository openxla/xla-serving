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
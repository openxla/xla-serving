<!-- disableFinding(LINK_RELATIVE_G3DOC) -->

# E5 export

Export `intfloat/e5-base-v2` at revision
`1c644c92ad3ba1efdad3f1451a637716616a20e8` to XLA SavedModel via TorchAX.
The graph runs the encoder, masked mean pooling, and L2 normalization in FP32.

## Tensor interface

| Signature | Shape of each input | Output `embeddings` |
|---|---|---|
| `embed_b1_s128` | `[1, 128]` | `[1, 768]`, FP32 |
| `embed_b1_s256` | `[1, 256]` | `[1, 768]`, FP32 |
| `embed_b1_s512` | `[1, 512]` | `[1, 768]`, FP32 |

Each signature takes `input_ids`, `attention_mask`, and `token_type_ids` as
`int32` tensors. Tokenize outside the graph using the same checkpoint revision,
including E5's `query: ` or `passage: ` prefix. Pad/truncate to the selected
sequence length, cast tokenizer outputs to `int32`, and use a binary attention
mask with at least one valid token. There is no raw-text serving signature.

## Setup and export

Build and install the XLA SavedModel Python package, then export E5 from this
checkout's root with Python 3.12 and Bazel available:

```bash
python3.12 -m venv .venv-e5
. .venv-e5/bin/activate
python -m pip install --upgrade build
python -m build --wheel
python -m pip install dist/xla_saved_model-*.whl \
  -r examples/e5/requirements.txt
python -m pip check
python -m examples.e5.run_export --export_path=exported_models/e5
```

Use a new output directory because saving may overwrite existing files.

## Parity test

Build and install [`xla_saved_model_runner`](../../docs/installation.md#python-wheels) from this checkout,
then run:

```bash
python -m pip install pytest
python -m pytest -v examples/e5/export_test.py
```

The test compares all three E5 signatures with an independent PyTorch reference
using padded queries and full-length passages.

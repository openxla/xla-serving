# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared helpers for the DLRM XSE-vs-Python serving benchmark.

Everything that must be *identical* between the XSE arm and the pure-Python
baseline arm lives here, so that the two can never drift apart:

* the wire format (`HostBufferProto` <-> numpy) and the gRPC method name,
* the synthetic request generator (seeded),
* the batch-size bucket policy,
* the numpy reference implementation of the SUM-pooled embedding lookup that
  mirrors `CpuEmbeddingModule(dense_lookup=True, combine_mode='SUM')`.
"""

from __future__ import annotations

import dataclasses
import functools
import json
import os
import sys
from typing import Any, Mapping, Sequence

import grpc
import numpy as np

from examples.dlrm import model_jce
from xla_saved_model.protos.grpc import host_buffer_pb2
from xla_saved_model.protos.grpc import predict_pb2
from xla_saved_model.protos.xla.python.ifrt import dtype_pb2  # pylint: disable=g-direct-tensorflow-import

# Fully-qualified gRPC method served by both XSE and the Python baseline.
# `PredictionService` lives in prediction_service.proto, which shares its
# package with predict.proto. Deriving the service path from the descriptor
# keeps all clients and servers in step.
PREDICT_SERVICE = f'{predict_pb2.DESCRIPTOR.package}.PredictionService'
PREDICT_METHOD = f'/{PREDICT_SERVICE}/Predict'

DEFAULT_MODEL_NAME = 'dlrm'
DEFAULT_SIGNATURE = 'serving_default'
DENSE_FEATURE_NAME = 'dense_features'
OUTPUT_NAME = 'predictions'
NUM_DENSE_FEATURES = 13
CONFIG_FILENAME = 'config.json'
TABLES_FILENAME = 'tables.npz'
PARAMS_FILENAME = 'params.npz'

# Concrete batch sizes the exported XSE model is specialised for. The Python
# arms pad to the same buckets, and the load generator only issues request
# batch sizes drawn from this set, so no arm ever pays for an unplanned
# recompile. Powers of two up to 4096 bound PAD_UP padding overhead below 2x
# (~1.4x on average) across all dynamically formed batch shapes.
DEFAULT_BATCH_SIZES = tuple(2**i for i in range(13))  # 1, 2, 4, ..., 4096

_NP_TO_KIND = {
    np.dtype(np.int32): dtype_pb2.DTypeProto.KIND_S32,
    np.dtype(np.int64): dtype_pb2.DTypeProto.KIND_S64,
    np.dtype(np.float32): dtype_pb2.DTypeProto.KIND_F32,
}
_KIND_TO_NP = {v: k for k, v in _NP_TO_KIND.items()}

_LOOPBACK_HOSTS = frozenset({'localhost', '127.0.0.1', '::1', '[::1]', ''})


def is_loopback_target(target: str) -> bool:
  """True if `host:port` points at this machine."""
  return (
      target in _LOOPBACK_HOSTS
      or target.rsplit(':', 1)[0] in _LOOPBACK_HOSTS
  )


def make_channel(
    target: str, options: Sequence[tuple[str, Any]] = ()
) -> grpc.Channel:
  """Creates a plaintext gRPC channel to an XSE or Python baseline server.

  For loopback targets (the benchmark's single-node setup) gRPC *local*
  credentials are used: zero-overhead loopback verification matching XSE's
  default `LocalServerCredentials` and the Python baseline's `add_local_port`.
  Remote targets fall back to `grpc.insecure_channel`.

  Args:
    target: `host:port`.
    options: gRPC channel options.

  Returns:
    An open channel.
  """
  if is_loopback_target(target):
    creds = grpc.local_channel_credentials(grpc.LocalConnectionType.LOCAL_TCP)
    return grpc.secure_channel(target, creds, options=options)
  return grpc.insecure_channel(target, options=options)


def add_local_port(server: grpc.Server, port: int) -> int:
  """Binds `server` to `port` (0 = ephemeral) for plaintext loopback clients.

  Uses gRPC *local* server credentials, matching XSE's default
  `LocalServerCredentials` so the same loopback clients work against both
  servers while accepting only loopback peers.

  Args:
    server: The gRPC server.
    port: Requested port, or 0 to let the OS pick one.

  Returns:
    The bound port.
  """
  creds = grpc.local_server_credentials(grpc.LocalConnectionType.LOCAL_TCP)
  return server.add_secure_port(f'[::]:{port}', creds)


@functools.cache
def _sparse_names(num_tables: int) -> tuple[str, ...]:
  return tuple(str(i) for i in range(num_tables))


@functools.cache
def _expected_feature_names(num_tables: int) -> frozenset[str]:
  return frozenset(_sparse_names(num_tables)) | {DENSE_FEATURE_NAME}


def sparse_feature_names(
    num_tables: int = len(model_jce.VOCAB_SIZES),
) -> list[str]:
  return list(_sparse_names(num_tables))


@dataclasses.dataclass(frozen=True)
class ModelConfig:
  """Static description of the benchmark model shared by all arms."""

  embedding_size: int
  vocab_sizes: tuple[int, ...]
  valency: tuple[int, ...]
  num_dense_features: int = NUM_DENSE_FEATURES
  batch_sizes: tuple[int, ...] = DEFAULT_BATCH_SIZES
  seed: int = 0

  @property
  def num_tables(self) -> int:
    return len(self.vocab_sizes)

  def to_json(self) -> str:
    return json.dumps(dataclasses.asdict(self), indent=2, sort_keys=True)

  @classmethod
  def from_json(cls, text: str) -> 'ModelConfig':
    d = json.loads(text)
    return cls(
        embedding_size=int(d['embedding_size']),
        vocab_sizes=tuple(int(v) for v in d['vocab_sizes']),
        valency=tuple(int(v) for v in d['valency']),
        num_dense_features=int(d.get('num_dense_features', NUM_DENSE_FEATURES)),
        batch_sizes=tuple(
            int(b) for b in d.get('batch_sizes', DEFAULT_BATCH_SIZES)
        ),
        seed=int(d.get('seed', 0)),
    )

  def save(self, weights_dir: str) -> None:
    os.makedirs(weights_dir, exist_ok=True)
    with open(os.path.join(weights_dir, CONFIG_FILENAME), 'w') as f:
      f.write(self.to_json())

  @classmethod
  def load(cls, weights_dir: str) -> 'ModelConfig':
    with open(os.path.join(weights_dir, CONFIG_FILENAME)) as f:
      return cls.from_json(f.read())


def make_model_config(
    embedding_size: int,
    max_vocab_size: int = 0,
    batch_sizes: Sequence[int] = DEFAULT_BATCH_SIZES,
    seed: int = 0,
) -> ModelConfig:
  """Builds the Criteo-like DLRM config, optionally capping vocabularies."""
  if embedding_size < 1:
    raise ValueError(f'embedding_size must be >= 1, got {embedding_size}')
  normalized_batches = tuple(sorted(set(int(b) for b in batch_sizes)))
  if not normalized_batches or normalized_batches[0] < 1:
    raise ValueError(
        f'batch_sizes must be non-empty positive integers, got {batch_sizes}'
    )
  if max_vocab_size > 0:
    vocab_sizes = tuple(min(v, max_vocab_size) for v in model_jce.VOCAB_SIZES)
  else:
    vocab_sizes = tuple(model_jce.VOCAB_SIZES)
  return ModelConfig(
      embedding_size=embedding_size,
      vocab_sizes=vocab_sizes,
      valency=tuple(model_jce.VALENCY),
      batch_sizes=normalized_batches,
      seed=seed,
  )


# --------------------------------------------------------------------------- #
# Wire format
# --------------------------------------------------------------------------- #


def to_host_buffer(
    arr: np.ndarray,
    out: host_buffer_pb2.HostBufferProto | None = None,
) -> host_buffer_pb2.HostBufferProto:
  """Converts a numpy array into the HostBufferProto used by XSE."""
  arr = np.ascontiguousarray(arr)
  if arr.dtype not in _NP_TO_KIND:
    raise ValueError(f'Unsupported dtype for HostBufferProto: {arr.dtype}')
  if out is None:
    hb = host_buffer_pb2.HostBufferProto()
  else:
    hb = out
    hb.Clear()
  hb.dtype.kind = _NP_TO_KIND[arr.dtype]
  hb.shape.dims.extend(arr.shape)
  hb.raw_data = arr.tobytes()
  return hb


def from_host_buffer(hb: host_buffer_pb2.HostBufferProto) -> np.ndarray:
  """Converts a HostBufferProto back into a numpy array (zero-copy view)."""
  if hb.dtype.kind not in _KIND_TO_NP:
    raise ValueError(f'Unsupported HostBufferProto dtype kind: {hb.dtype.kind}')
  dims = tuple(int(d) for d in hb.shape.dims)
  if any(d < 0 for d in dims):
    raise ValueError(f'Negative dimension in HostBufferProto shape: {dims}')
  dtype = _KIND_TO_NP[hb.dtype.kind]
  return np.frombuffer(hb.raw_data, dtype=dtype).reshape(dims)


def make_predict_request(
    features: Mapping[str, np.ndarray],
    model_name: str = DEFAULT_MODEL_NAME,
    signature_name: str = DEFAULT_SIGNATURE,
) -> predict_pb2.PredictRequest:
  req = predict_pb2.PredictRequest()
  req.model_spec.name = model_name
  req.model_spec.signature_name = signature_name
  for name, arr in features.items():
    to_host_buffer(arr, out=req.inputs[name])
  return req


def parse_predict_request(
    req: predict_pb2.PredictRequest,
) -> dict[str, np.ndarray]:
  return {name: from_host_buffer(hb) for name, hb in req.inputs.items()}


def make_predict_response(
    outputs: Mapping[str, np.ndarray],
    request: predict_pb2.PredictRequest | None = None,
) -> predict_pb2.PredictResponse:
  resp = predict_pb2.PredictResponse()
  if request is not None:
    resp.model_spec.CopyFrom(request.model_spec)
  for name, arr in outputs.items():
    to_host_buffer(arr, out=resp.outputs[name])
  return resp


def parse_predict_response(
    resp: predict_pb2.PredictResponse,
) -> dict[str, np.ndarray]:
  return {name: from_host_buffer(hb) for name, hb in resp.outputs.items()}


def predictions_from_response(resp: predict_pb2.PredictResponse) -> np.ndarray:
  """Extracts the (batch,) prediction vector from either server's response."""
  outputs = parse_predict_response(resp)
  if OUTPUT_NAME in outputs:
    out = outputs[OUTPUT_NAME]
  elif len(outputs) == 1:
    out = next(iter(outputs.values()))
  else:
    raise ValueError(f'Cannot find predictions in outputs {list(outputs)}')
  return np.asarray(out, dtype=np.float32).ravel()


# --------------------------------------------------------------------------- #
# Synthetic data
# --------------------------------------------------------------------------- #


def generate_features(
    config: ModelConfig,
    batch_size: int,
    rng: np.random.Generator,
) -> dict[str, np.ndarray]:
  """Generates one request worth of features (ids are int32, dense float32)."""
  features: dict[str, np.ndarray] = {}
  for name, vocab, valency in zip(
      _sparse_names(config.num_tables), config.vocab_sizes, config.valency
  ):
    features[name] = rng.integers(
        0, vocab, size=(batch_size, valency), dtype=np.int32
    )
  features[DENSE_FEATURE_NAME] = rng.uniform(
      0.0, 0.9, size=(batch_size, config.num_dense_features)
  ).astype(np.float32)
  return features


def batch_size_of(features: Mapping[str, np.ndarray]) -> int:
  sizes = {int(v.shape[0]) for v in features.values()}
  if len(sizes) != 1:
    raise ValueError(f'Inconsistent batch dimension across inputs: {sizes}')
  return sizes.pop()


def _raise_input_names_mismatch(
    expected: frozenset[str], features: Mapping[str, np.ndarray]
) -> None:
  got = set(features)
  raise ValueError(
      f'Input names mismatch. Missing: {sorted(expected - got)}, '
      f'unexpected: {sorted(got - expected)}'
  )


def validate_features(
    config: ModelConfig, features: Mapping[str, np.ndarray]
) -> int:
  """Checks names/shapes/dtypes against `config`; returns the batch size."""
  expected = _expected_feature_names(config.num_tables)
  if len(features) != len(expected):
    _raise_input_names_mismatch(expected, features)
  dense = features.get(DENSE_FEATURE_NAME)
  if dense is None:
    _raise_input_names_mismatch(expected, features)
  assert dense is not None
  if dense.ndim != 2:
    if set(features) != expected:
      _raise_input_names_mismatch(expected, features)
    raise ValueError(
        f'{DENSE_FEATURE_NAME} must be rank 2, got shape {dense.shape}'
    )
  batch = int(dense.shape[0])
  for i, (name, valency) in enumerate(
      zip(_sparse_names(config.num_tables), config.valency)
  ):
    arr = features.get(name)
    if arr is None:
      _raise_input_names_mismatch(expected, features)
    assert arr is not None
    if arr.ndim != 2:
      if set(features) != expected:
        _raise_input_names_mismatch(expected, features)
      raise ValueError(
          f'Feature {i} must have shape ({batch}, {valency}), got {arr.shape}'
      )
    if arr.shape[0] != batch:
      if set(features) != expected:
        _raise_input_names_mismatch(expected, features)
      batch_size_of(features)
    if arr.shape[1] != valency:
      raise ValueError(
          f'Feature {i} must have shape ({batch}, {valency}), got {arr.shape}'
      )
    if arr.dtype != np.int32 and arr.dtype != np.int64:
      raise ValueError(f'Feature {i} must be int32/int64, got {arr.dtype}')
  if dense.shape[1] != config.num_dense_features:
    raise ValueError(
        f'{DENSE_FEATURE_NAME} must have shape '
        f'({batch}, {config.num_dense_features}), got {dense.shape}'
    )
  if dense.dtype != np.float32:
    raise ValueError(
        f'{DENSE_FEATURE_NAME} must be float32, got {dense.dtype}'
    )
  if batch < 1:
    raise ValueError(f'Batch dimension must be >= 1, got {batch}')
  return batch


# --------------------------------------------------------------------------- #
# Bucketing
# --------------------------------------------------------------------------- #


def next_bucket(n: int, buckets: Sequence[int]) -> int:
  """Smallest allowed batch size >= n (mirrors XSE's PAD_UP policy)."""
  for b in buckets:
    if b >= n:
      return b
  raise ValueError(f'Batch size {n} exceeds largest bucket {buckets[-1]}')


def pad_features(
    features: Mapping[str, np.ndarray], target_batch: int
) -> dict[str, np.ndarray]:
  """Zero-pads every input along axis 0 up to `target_batch` rows."""
  batch = batch_size_of(features)
  if batch == target_batch:
    return dict(features)
  if batch > target_batch:
    raise ValueError(f'Cannot pad batch {batch} down to {target_batch}')
  padded = {}
  for name, arr in features.items():
    pad = np.zeros((target_batch - batch,) + arr.shape[1:], dtype=arr.dtype)
    padded[name] = np.concatenate([arr, pad], axis=0)
  return padded


def concat_features(
    feature_list: Sequence[Mapping[str, np.ndarray]],
) -> dict[str, np.ndarray]:
  """Concatenates several requests along the batch axis (dynamic batching)."""
  if len(feature_list) == 1:
    return dict(feature_list[0])
  keys = feature_list[0].keys()
  return {k: np.concatenate([f[k] for f in feature_list], axis=0) for k in keys}


def pad_rows(arr: np.ndarray, target_rows: int) -> np.ndarray:
  """Zero-pads `arr` along axis 0 up to `target_rows` (no-op if already there).

  Used by the Python arms to pad the *looked-up* embeddings and dense features
  rather than the ids: padding rows then cost a memset instead of a gather.

  Args:
    arr: Input array whose leading axis is the batch dimension.
    target_rows: Desired leading dimension (>= `arr.shape[0]`).

  Returns:
    `arr` unchanged if `arr.shape[0] == target_rows`, else a zero-padded copy.
  """
  rows = arr.shape[0]
  if rows == target_rows:
    return arr
  if rows > target_rows:
    raise ValueError(f'Cannot pad {rows} rows down to {target_rows}')
  out = np.empty((target_rows,) + arr.shape[1:], dtype=arr.dtype)
  out[:rows] = arr
  out[rows:] = 0
  return out


# --------------------------------------------------------------------------- #
# Reference embedding lookup (what the CPU preprocessor computes)
# --------------------------------------------------------------------------- #


def embedding_lookup_numpy(
    tables: Sequence[np.ndarray],
    features: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
  """SUM-pooled gather: `out[i] = sum_j tables[i][ids[i][:, j]] -> (B, dim)`.

  Numerically equivalent to `CpuEmbeddingModule` with `combine_mode=SUM` and
  `dense_lookup=True`. Kept as the readable reference; the servers use
  `embedding_lookup_stacked`.

  Args:
    tables: Embedding tables in feature order, each `(vocab_i, dim)`.
    features: Request feature dict mapping `str(i)` to `(B, valency_i)` ids.

  Returns:
    Dict mapping `str(i)` to float32 pooled embeddings of shape `(B, dim)`.
  """
  out = {}
  for i, table in enumerate(tables):
    ids = features[str(i)]
    # `np.take` with axis=0 is the fastest pure-numpy gather for 2-D tables.
    gathered = np.take(table, ids.ravel(), axis=0)
    gathered = gathered.reshape(ids.shape[0], ids.shape[1], table.shape[1])
    out[str(i)] = gathered.sum(axis=1, dtype=np.float32)
  return out


def embedding_lookup_stacked(
    tables: Sequence[np.ndarray],
    features: Mapping[str, np.ndarray],
    out: np.ndarray | None = None,
) -> np.ndarray:
  """Same gather as `embedding_lookup_numpy`, written into one (B, T, D) array.

  Each table's pooled sum is reduced straight into `out[:, i, :]`, so the
  result needs one host-to-device transfer instead of 26, and a caller can
  hand in a view of shared memory (the multi-process server does).

  Args:
    tables: Embedding tables in feature order, each (vocab_i, D).
    features: Request features; ids for table i under key `str(i)`.
    out: Optional float32 (B, T, D) array to write into (may be a view).

  Returns:
    `out`, or a new (B, T, D) float32 array.
  """
  batch = features['0'].shape[0]
  dim = tables[0].shape[1]
  if out is None:
    out = np.empty((batch, len(tables), dim), dtype=np.float32)
  elif out.shape != (batch, len(tables), dim) or out.dtype != np.float32:
    raise ValueError(
        f'out must be float32 {(batch, len(tables), dim)}, got '
        f'{out.dtype} {out.shape}'
    )
  for i, (name, table) in enumerate(zip(_sparse_names(len(tables)), tables)):
    ids = features[name]
    if ids.shape[1] == 1:
      out[:, i, :] = np.take(table, ids.ravel(), axis=0)
    else:
      gathered = np.take(table, ids.ravel(), axis=0)
      gathered = gathered.reshape(batch, ids.shape[1], dim)
      gathered.sum(axis=1, dtype=np.float32, out=out[:, i, :])
  return out


def stacked_to_dict(stacked: np.ndarray) -> dict[str, np.ndarray]:
  """(B, T, D) -> {'0': (B, D), ...}; the layout `model_jce` consumes."""
  return {
      name: stacked[:, i, :]
      for i, name in enumerate(_sparse_names(stacked.shape[1]))
  }


# --------------------------------------------------------------------------- #
# Re-exec helper
# --------------------------------------------------------------------------- #


def self_command() -> list[str]:
  """Returns the argv prefix that re-launches the running program.

  Helper processes of the benchmark (load generator workers, multi-process
  server frontends) are separate *processes* (own GIL, own gRPC stack) started
  by re-executing the current program with extra flags. Re-exec rather than
  `multiprocessing`: gRPC's core starts threads at import time, so a forked
  child cannot use it, and `spawn` needs `sys.executable`, which is None in
  single-file Python binaries.
  """
  main_mod = sys.modules.get('__main__')
  spec = getattr(main_mod, '__spec__', None)
  if sys.executable and spec is not None and spec.name:
    return [sys.executable, '-m', spec.name]  # `python -m pkg.tool`
  if sys.executable and sys.argv[0].endswith('.py'):
    return [sys.executable, sys.argv[0]]  # `python path/to/tool.py`
  return [os.path.abspath(sys.argv[0])]  # self-contained binary


# --------------------------------------------------------------------------- #
# Baseline weight files
# --------------------------------------------------------------------------- #


def flatten_params(params: Any) -> dict[str, np.ndarray]:
  """Flattens a nested dict pytree into {'a/b/c': array}."""
  flat: dict[str, np.ndarray] = {}

  def _walk(node: Any, prefix: str) -> None:
    if isinstance(node, Mapping):
      for k in sorted(node):
        _walk(node[k], f'{prefix}/{k}' if prefix else str(k))
    else:
      flat[prefix] = np.asarray(node)

  _walk(params, '')
  return flat


def unflatten_params(flat: Mapping[str, np.ndarray]) -> dict[str, Any]:
  """Inverse of `flatten_params`."""
  root: dict[str, Any] = {}
  for key, value in flat.items():
    parts = key.split('/')
    node = root
    for p in parts[:-1]:
      node = node.setdefault(p, {})
    node[parts[-1]] = np.asarray(value)
  return root


def save_baseline_weights(
    weights_dir: str,
    config: ModelConfig,
    tables: Mapping[str, np.ndarray],
    params: Any,
) -> None:
  """Writes config.json, tables.npz and params.npz for the Python baseline."""
  config.save(weights_dir)
  _savez(os.path.join(weights_dir, TABLES_FILENAME), dict(tables))
  _savez(os.path.join(weights_dir, PARAMS_FILENAME), flatten_params(params))


def _savez(path: str, arrays: Mapping[str, np.ndarray]) -> None:
  """`np.savez` with a mapping (typed wrapper around the **kwargs API)."""
  savez: Any = np.savez
  with open(path, 'wb') as f:
    savez(f, **arrays)


def load_baseline_weights(
    weights_dir: str,
    *,
    tables: bool = True,
    params: bool = True,
) -> tuple[ModelConfig, list[np.ndarray], dict[str, Any]]:
  """Loads what `save_baseline_weights` wrote; tables are in feature order.

  Args:
    weights_dir: Directory written by `save_baseline_weights`.
    tables: Load the embedding tables (the multi-process server's device owner
      does not need them: its frontends do the lookups).
    params: Load the DCNv2 parameters (frontends do not need them).

  Returns:
    (config, tables in feature order or [], params or {}).
  """
  config = ModelConfig.load(weights_dir)
  table_list: list[np.ndarray] = []
  if tables:
    with np.load(os.path.join(weights_dir, TABLES_FILENAME)) as npz:
      table_list = [
          np.ascontiguousarray(npz[str(i)]) for i in range(config.num_tables)
      ]
  param_tree: dict[str, Any] = {}
  if params:
    with np.load(os.path.join(weights_dir, PARAMS_FILENAME)) as npz:
      param_tree = unflatten_params({k: npz[k] for k in npz.files})
  return config, table_list, param_tree


# --------------------------------------------------------------------------- #
# Statistics
# --------------------------------------------------------------------------- #


def latency_summary_ms(latencies_s: Sequence[float]) -> dict[str, float]:
  """p50/p90/p99/p99.9/max/mean in milliseconds."""
  if not len(latencies_s):  # pylint: disable=g-explicit-length-test
    return {
        k: float('nan') for k in ('p50', 'p90', 'p99', 'p999', 'max', 'mean')
    }
  arr = np.asarray(latencies_s, dtype=np.float64) * 1000.0
  p50, p90, p99, p999 = np.percentile(arr, [50, 90, 99, 99.9])
  return {
      'p50': float(p50),
      'p90': float(p90),
      'p99': float(p99),
      'p999': float(p999),
      'max': float(arr.max()),
      'mean': float(arr.mean()),
  }

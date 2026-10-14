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

"""Pure-Python JAX baseline server for the DLRM benchmark.

Serves exactly the same gRPC method and protos as XSE
(`PredictionService/Predict`, i.e. `common.PREDICT_METHOD`, with
`HostBufferProto` inputs) so the same load generator and parity checker can be
pointed at either server.

Request path (per RPC):
  1. parse `PredictRequest` -> dict of numpy arrays            (Python, GIL)
  2. SUM-pooled embedding gather for 26 tables with numpy into
     one (B, 26, D) array                                       (host)
  3. pad to the next allowed bucket, `device_put` (one transfer
     each for embeddings and dense), `jax.jit`-compiled DCNv2  (device)
  4. serialise `PredictResponse`                                (Python, GIL)

Modes, mirroring the XSE arms:
  * default:              one request == one model call (like XSE NO_BATCHING);
                          `--num_threads` gRPC handler threads do the work.
  * `--dynamic_batching`: a queue thread coalesces concurrent requests up to
    `--max_batch_size` or `--batch_timeout_micros`, pads to the next allowed
    bucket, runs lookup+model once and scatters results back (a hand-rolled
    equivalent of XSE `BatchOptions`, batching *before* the embedding gather
    like XSE's `PRE_PROCESSOR_AND_MODEL_FUNCTION`). `--num_batch_threads` do
    the work; the gRPC handler pool is widened (`--handler_threads`) because
    its threads only park on a future and capping them would cap the batch
    size. Admission control mirrors XSE's `max_enqueued_batches`: when more
    than `--max_enqueued_batches` x `--max_batch_size` examples are waiting
    a request fails fast with UNAVAILABLE instead of queueing.
  * `--num_processes=N` (N > 1): the multi-process variant that answers "can
    Python be made faster with several processes behind one port?". N
    re-executed copies of this program serve gRPC on the same port
    (`SO_REUSEPORT`) and do parsing + lookup each with their own GIL; this
    process owns the accelerator and runs the model on their behalf through
    shared memory. See multiprocess_server.py. Combines with
    `--dynamic_batching`: each frontend batches in front of its gather (same
    placement as the single-process arm) and the owner merges the batches
    that are already waiting for the device without adding any wait.

This is deliberately the "honest, reasonably well-written" Python server a team
would build first, not a research-grade one: `grpc.server` with a thread pool,
numpy, `jax.jit`, no custom C extensions.
"""

from __future__ import annotations

import collections
from concurrent import futures
import dataclasses
import functools
import os
import signal
import threading
import time
from typing import Any, Callable, Mapping, Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import jax
import numpy as np

from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from xla_saved_model.protos.grpc import predict_pb2

# Handler threads used by `--dynamic_batching` when `--handler_threads=0`.
# Must be >= the largest closed-loop concurrency of the sweep so the batcher
# can see every in-flight request.
DEFAULT_BATCHING_HANDLER_THREADS = 512

_JAX_PLATFORMS_FOR_DEVICE = {
    'cpu': 'cpu',
    'cuda': 'cuda,cpu',
    'tpu': 'tpu,cpu',
}


@dataclasses.dataclass(frozen=True)
class Flags:
  """Command-line flags; only registered when run as a binary."""

  port: flags.FlagHolder[int]
  weights_dir: flags.FlagHolder[str]
  model_name: flags.FlagHolder[str]
  device: flags.FlagHolder[str]
  num_threads: flags.FlagHolder[int]
  handler_threads: flags.FlagHolder[int]
  dynamic_batching: flags.FlagHolder[bool]
  max_batch_size: flags.FlagHolder[int]
  batch_timeout_micros: flags.FlagHolder[int]
  num_batch_threads: flags.FlagHolder[int]
  max_enqueued_batches: flags.FlagHolder[int]
  warmup: flags.FlagHolder[bool]
  max_message_mb: flags.FlagHolder[int]
  num_processes: flags.FlagHolder[int]
  ring_mb: flags.FlagHolder[int]
  ipc_dir: flags.FlagHolder[str | None]
  frontend_id: flags.FlagHolder[int | None]


def define_flags() -> Flags:
  """Registers flags.

  Defined lazily (not at import time) so that other scripts that import this
  module can define flags with the same names.

  Returns:
    Holders for all flags of this binary.
  """
  return Flags(
      port=flags.DEFINE_integer('port', 8501, 'gRPC port to listen on.'),
      weights_dir=flags.DEFINE_string(
          'weights_dir',
          None,
          'Directory written by export_benchmark_model.py '
          '--baseline_weights_dir.',
          required=True,
      ),
      model_name=flags.DEFINE_string(
          'model_name', common.DEFAULT_MODEL_NAME, 'Model name to accept.'
      ),
      device=flags.DEFINE_enum(
          'device', 'cpu', ['cpu', 'cuda', 'tpu'], 'Device for the DCNv2 model.'
      ),
      num_threads=flags.DEFINE_integer(
          'num_threads',
          8,
          'Threads that execute requests (compare with XSE '
          '--thread_pool_num_threads). Without --dynamic_batching these are '
          'the gRPC handler threads; with it, see --handler_threads. With '
          '--num_processes they are the handler threads of *each* frontend '
          'and the model-dispatch threads of the device owner.',
      ),
      handler_threads=flags.DEFINE_integer(
          'handler_threads',
          0,
          'gRPC handler threads. 0 = --num_threads, except with '
          '--dynamic_batching where it is max(--num_threads, '
          f'{DEFAULT_BATCHING_HANDLER_THREADS}): handler threads then only '
          'park on a future while --num_batch_threads do the work, and '
          'capping them would cap the batch size at --num_threads requests '
          '(XSE queues RPCs in its batch scheduler independently of its '
          'execution pool). With --num_processes the batching default is '
          'split evenly across the frontends.',
      ),
      dynamic_batching=flags.DEFINE_bool(
          'dynamic_batching',
          False,
          'Coalesce concurrent requests into batches (before the embedding '
          'lookup, like XSE PRE_PROCESSOR_AND_MODEL_FUNCTION).',
      ),
      max_batch_size=flags.DEFINE_integer(
          'max_batch_size',
          0,
          'Dynamic batching cap (0 = largest allowed bucket).',
      ),
      batch_timeout_micros=flags.DEFINE_integer(
          'batch_timeout_micros',
          1000,
          'Dynamic batching max wait for a full batch.',
      ),
      num_batch_threads=flags.DEFINE_integer(
          'num_batch_threads',
          0,
          'Dynamic batching worker threads (0 = --num_threads; give XSE the '
          'same value via export_benchmark_model --num_batch_threads). With '
          '--num_processes they are split evenly across the frontends (at '
          'least one each) and the device owner uses the full count.',
      ),
      max_enqueued_batches=flags.DEFINE_integer(
          'max_enqueued_batches',
          64,
          'Dynamic batching admission control, the equivalent of XSE '
          'max_enqueued_batches: a request is rejected with UNAVAILABLE when '
          'more than this many maximum-size batches worth of examples are '
          'already waiting (0 = unbounded). With --num_processes the budget '
          'is split evenly across the frontends.',
      ),
      warmup=flags.DEFINE_bool(
          'warmup',
          True,
          'Run one request per allowed batch size before serving.',
      ),
      max_message_mb=flags.DEFINE_integer(
          'max_message_mb', 64, 'gRPC max send/receive message size.'
      ),
      num_processes=flags.DEFINE_integer(
          'num_processes',
          1,
          'If > 1: start this many gRPC frontend processes sharing --port via '
          'SO_REUSEPORT (each parses requests and does the embedding lookup '
          'with its own GIL and its own copy of the tables) while this '
          'process owns the accelerator and runs the model for all of them.',
      ),
      ring_mb=flags.DEFINE_integer(
          'ring_mb',
          256,
          'Shared-memory ring (embeddings in, predictions out) per frontend '
          'process, in MiB. Request handlers block when it is full.',
      ),
      ipc_dir=flags.DEFINE_string(
          'ipc_dir',
          None,
          "Directory for the multi-process server's UNIX socket and ring "
          'files (default: a fresh directory under /dev/shm, else $TMPDIR).',
      ),
      frontend_id=flags.DEFINE_integer(
          'frontend_id',
          None,
          'Internal: set when this program is re-executed as a frontend of '
          'the multi-process server.',
      ),
  )


def restrict_jax_platforms(device: str) -> None:
  """Makes JAX initialise only the backends needed for `device`.

  Must run before the first `jax.devices()` / computation. JAX reads
  `JAX_PLATFORMS` from the environment at *import* time, so setting the
  variable afterwards has no effect; `jax.config.update` does. Without this,
  a helper process (exporter, parity check, a `--device=cpu` server) started
  on a GPU VM would initialise CUDA and pre-allocate most of the GPU memory
  while the server under test owns the device, and on a TPU VM it would try
  to open the chip a second process already holds.

  An explicit `JAX_PLATFORMS` in the environment is honoured as is.

  Args:
    device: `cpu`, `cuda` or `tpu`.
  """
  if os.environ.get('JAX_PLATFORMS'):
    return
  jax.config.update('jax_platforms', _JAX_PLATFORMS_FOR_DEVICE[device])


def _jax_device(name: str) -> jax.Device:
  platform = {'cpu': 'cpu', 'cuda': 'gpu', 'tpu': 'tpu'}[name]
  devices = jax.devices(platform)
  if not devices:
    raise RuntimeError(f'No JAX devices for platform {platform!r}')
  return devices[0]


class DlrmBaseline:
  """Loads baseline weights and serves predictions (numpy lookup + jax.jit).

  The request path is split in two so that the multi-process server can run
  the halves in different processes:

    * `lookup(features)`      host: numpy gather into one (B, 26, D) array;
    * `run_model(emb, dense)` device: pad to bucket, `device_put`, jitted DCNv2.

  `predict` is simply both.
  """

  def __init__(
      self, weights_dir: str, device: str = 'cpu', load_tables: bool = True
  ):
    t0 = time.perf_counter()
    self.config, self._tables, params = common.load_baseline_weights(
        weights_dir, tables=load_tables
    )
    self._device = _jax_device(device)
    self._params = jax.device_put(params, self._device)
    self._model = model_jce.build_model(
        embedding_size=self.config.embedding_size
    )
    self._apply = jax.jit(self._apply_stacked)
    self.load_time_s = time.perf_counter() - t0
    self.buckets = tuple(self.config.batch_sizes)
    logging.info(
        'Loaded baseline weights from %s on %s in %.2fs (%d tables, dim=%d)',
        weights_dir,
        self._device,
        self.load_time_s,
        len(self._tables),
        self.config.embedding_size,
    )

  @property
  def device(self) -> jax.Device:
    return self._device

  @property
  def tables(self) -> list[np.ndarray]:
    return self._tables

  def _apply_stacked(self, params, dense, emb):
    # One (B, T, D) input; the per-table dict the model wants is sliced out
    # inside the jitted function (XLA fuses the slices with the concat).
    return self._model.apply(params, dense, common.stacked_to_dict(emb))

  def lookup(
      self,
      features: Mapping[str, np.ndarray],
      out: np.ndarray | None = None,
  ) -> tuple[np.ndarray, np.ndarray]:
    """Host half: returns (embeddings (B, T, D) float32, dense (B, F))."""
    if not self._tables:
      raise RuntimeError('This DlrmBaseline was created without tables')
    emb = common.embedding_lookup_stacked(self._tables, features, out=out)
    return emb, features[common.DENSE_FEATURE_NAME]

  def run_model(
      self, emb: np.ndarray, dense: np.ndarray, batch: int
  ) -> np.ndarray:
    """Device half: pads to the next bucket, runs DCNv2, strips the padding.

    Args:
      emb: (rows, T, D) float32 pooled embeddings, rows >= batch; rows beyond
        `batch` are padding and may hold anything.
      dense: (rows, F) float32 dense features.
      batch: Number of real examples.

    Returns:
      (batch,) float32 predictions.
    """
    target = common.next_bucket(batch, self.buckets)
    if emb.shape[0] != target:
      emb = common.pad_rows(emb[:batch], target)
      dense = common.pad_rows(dense[:batch], target)
    # Explicit H2D so the transfer cost is attributed to this request, exactly
    # like XSE moving the preprocessor outputs to the device.
    emb_d = jax.device_put(emb, self._device)
    dense_d = jax.device_put(dense, self._device)
    out = self._apply(self._params, dense_d, emb_d)
    return np.asarray(out, dtype=np.float32).reshape(-1)[:batch]

  def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
    """Runs lookup + model for one (possibly concatenated) batch."""
    batch = common.validate_features(self.config, features)
    emb, dense = self.lookup(features)
    return self.run_model(emb, dense, batch)

  def warmup(self) -> dict[int, float]:
    """Compiles/caches every bucket; returns per-bucket first-call latency.

    Works without tables (device-owner process): random embeddings then stand
    in for the lookup, which is irrelevant for the compile cache.
    """
    rng = np.random.default_rng(self.config.seed + 1)
    timings = {}
    for b in self.buckets:
      t0 = time.perf_counter()
      if self._tables:
        self.predict(common.generate_features(self.config, b, rng))
      else:
        emb = rng.uniform(
            -0.5,
            0.5,
            size=(b, self.config.num_tables, self.config.embedding_size),
        ).astype(np.float32)
        dense = rng.uniform(
            0.0, 0.9, size=(b, self.config.num_dense_features)
        ).astype(np.float32)
        self.run_model(emb, dense, b)
      timings[b] = time.perf_counter() - t0
      logging.info('Warmup batch %d: %.1f ms', b, timings[b] * 1000)
    return timings


@dataclasses.dataclass
class _PendingRequest:
  payload: Any
  batch: int
  future: futures.Future[Any]
  enqueued_at: float


# run_fn(payloads, batch sizes) -> one result per payload, in order.
BatchRunFn = Callable[[list[Any], list[int]], Sequence[Any]]


class QueueFullError(Exception):
  """Raised by `DynamicBatcher.submit` when admission control rejects a request.

  Carries XSE's wording for the same condition (`SharedBatchScheduler`:
  "The batch scheduling queue to which this task was submitted is full").
  """


def split_across(total: int, parts: int) -> int:
  """Per-process share of a budget of `total` spread over `parts` processes.

  Rounds up so that the shares cover the budget; a budget of 0 stays 0
  (unbounded) and a positive budget never drops below 1 per part.

  Args:
    total: The budget (threads, batches, ...) for all parts together.
    parts: Number of parts (frontend processes).

  Returns:
    The share of one part.
  """
  if total <= 0:
    return 0
  return max(1, -(-total // max(1, parts)))


class DynamicBatcher:
  """Coalesces concurrent requests into one `run_fn` call.

  Semantics follow XSE `BatchOptions`: wait at most `timeout_s` for the batch
  to reach `max_batch_size` examples, never split a request across batches,
  and let `run_fn` pad to the next allowed bucket. Payloads are opaque: the
  single-process server and the multi-process frontends submit feature dicts,
  the multi-process device owner submits shared-memory row ranges, so all arms
  batch identically.

  Admission control follows XSE `max_enqueued_batches`: XSE rejects a task
  with UNAVAILABLE when `max_enqueued_batches` batches are already queued and
  the task does not fit into the last one; the example-count equivalent here
  is "more than `max_enqueued_batches * max_batch_size` examples would be
  waiting". 0 disables the bound.
  """

  def __init__(
      self,
      run_fn: BatchRunFn,
      max_batch_size: int,
      timeout_s: float,
      buckets: Sequence[int],
      num_threads: int = 1,
      max_enqueued_batches: int = 0,
  ):
    self._run_fn = run_fn
    self._max_batch_size = max_batch_size
    self._timeout_s = timeout_s
    self._buckets = tuple(buckets)
    self._max_enqueued_batches = max(0, max_enqueued_batches)
    self._max_queued_examples = self._max_enqueued_batches * max_batch_size
    self._cv = threading.Condition()
    self._queue: collections.deque[_PendingRequest] = collections.deque()
    self._queued_examples = 0
    self._stopped = False
    self.batches_run = 0
    self.examples_run = 0
    self.padded_examples = 0
    self.rejected = 0
    self._threads = [
        threading.Thread(target=self._loop, name=f'batcher-{i}', daemon=True)
        for i in range(max(1, num_threads))
    ]
    for t in self._threads:
      t.start()

  @property
  def max_enqueued_batches(self) -> int:
    return self._max_enqueued_batches

  @property
  def num_threads(self) -> int:
    return len(self._threads)

  @property
  def queued_examples(self) -> int:
    with self._cv:
      return self._queued_examples

  def submit(self, payload: Any, batch: int) -> futures.Future[Any]:
    """Queues one request; raises `QueueFullError` if admission control says no.

    Args:
      payload: Opaque request payload handed to `run_fn`.
      batch: Number of examples in the request.

    Returns:
      A future that resolves to this request's slice of the `run_fn` result.

    Raises:
      QueueFullError: More than `max_enqueued_batches * max_batch_size`
        examples would be waiting (XSE's "batch scheduling queue is full").
    """
    fut: futures.Future[Any] = futures.Future()
    with self._cv:
      if (
          self._max_queued_examples
          and self._queued_examples + batch > self._max_queued_examples
      ):
        self.rejected += 1
        raise QueueFullError(
            'The batch scheduling queue to which this request was submitted '
            f'is full; currently {self._queued_examples} examples '
            f'({-(-self._queued_examples // self._max_batch_size)} batches) '
            f'enqueued and max_enqueued_batches is {self._max_enqueued_batches}'
        )
      self._queue.append(
          _PendingRequest(payload, batch, fut, time.perf_counter())
      )
      self._queued_examples += batch
      self._cv.notify()
    return fut

  def stop(self) -> None:
    with self._cv:
      self._stopped = True
      self._cv.notify_all()
    for t in self._threads:
      t.join(timeout=5)

  def stats_line(self, prefix: str = 'Batcher stats') -> str:
    """One-line summary printed at shutdown (parsed by the tests)."""
    return (
        f'{prefix}: {self.batches_run} batches, {self.examples_run} examples, '
        f'{self.padded_examples} padded examples, {self.rejected} rejected'
    )

  def _take_batch(self) -> list[_PendingRequest]:
    """Blocks until a batch is ready (full or timed out) and dequeues it."""
    with self._cv:
      while True:
        while not self._queue and not self._stopped:
          self._cv.wait()
        if not self._queue:
          return []  # stopped and drained
        deadline = self._queue[0].enqueued_at + self._timeout_s
        while (
            self._queue
            and self._queued_examples < self._max_batch_size
            and not self._stopped
        ):
          remaining = deadline - time.perf_counter()
          if remaining <= 0:
            break
          self._cv.wait(timeout=remaining)
        if not self._queue:
          continue  # another worker took it; wait for new work
        taken: list[_PendingRequest] = []
        size = 0
        while self._queue:
          nxt = self._queue[0]
          if taken and size + nxt.batch > self._max_batch_size:
            break
          taken.append(self._queue.popleft())
          size += nxt.batch
        self._queued_examples -= size
        return taken

  def _loop(self) -> None:
    """Worker thread: take a batch, run it once, scatter results/errors."""
    while True:
      batch = self._take_batch()
      if not batch:
        return
      try:
        total = sum(p.batch for p in batch)
        results = self._run_fn(
            [p.payload for p in batch], [p.batch for p in batch]
        )
        self.batches_run += 1
        self.examples_run += total
        self.padded_examples += common.next_bucket(total, self._buckets) - total
        for p, r in zip(batch, results):
          p.future.set_result(r)
      except Exception as e:  # pylint: disable=broad-exception-caught
        for p in batch:
          if not p.future.done():
            p.future.set_exception(e)


def split_predictions(preds: np.ndarray, batches: Sequence[int]) -> list[Any]:
  """Slices a concatenated prediction vector back into per-request pieces."""
  out = []
  offset = 0
  for b in batches:
    out.append(preds[offset : offset + b])
    offset += b
  return out


class Predictor:
  """What `PredictionServicer` needs from a model: config + predict()."""

  config: common.ModelConfig

  def predict(self, features: Mapping[str, np.ndarray]) -> np.ndarray:
    raise NotImplementedError


class PredictionServicer:
  """Generic gRPC handler for `PredictionService/Predict`.

  Registered under `common.PREDICT_METHOD`, the same name the clients call and
  the real XSE server serves (the proto package differs between  and the
  OSS export, so the name is never spelled out here).
  """

  def __init__(
      self,
      model: DlrmBaseline | Predictor,
      model_name: str,
      batcher: DynamicBatcher | None = None,
  ):
    self._model = model
    self._model_name = model_name
    self._batcher = batcher
    self.port = 0
    self.handler_threads = 0
    self.requests = 0
    self.examples = 0
    self.rejected = 0

  def predict(
      self, request: predict_pb2.PredictRequest, context: grpc.ServicerContext
  ) -> predict_pb2.PredictResponse:
    """Handles one `Predict` RPC (directly or through the batcher)."""
    if request.model_spec.name and request.model_spec.name != self._model_name:
      context.abort(
          grpc.StatusCode.NOT_FOUND,
          f'Unknown model {request.model_spec.name!r}; serving '
          f'{self._model_name!r}',
      )
    try:
      features = common.parse_predict_request(request)
      batch = common.validate_features(self._model.config, features)
    except ValueError as e:
      context.abort(grpc.StatusCode.INVALID_ARGUMENT, str(e))
      raise AssertionError('unreachable') from e
    if self._batcher is not None:
      try:
        fut = self._batcher.submit(features, batch)
      except QueueFullError as e:
        # Same status XSE's batch scheduler returns when its queue is full;
        # the request has cost parsing and validation only at this point.
        self.rejected += 1
        context.abort(grpc.StatusCode.UNAVAILABLE, str(e))
        raise AssertionError('unreachable') from e
      preds = fut.result()
    else:
      preds = self._model.predict(features)
    self.requests += 1
    self.examples += batch
    return common.make_predict_response({common.OUTPUT_NAME: preds}, request)

  def generic_handler(self) -> grpc.GenericRpcHandler:
    service, method = common.PREDICT_METHOD.strip('/').rsplit('/', 1)
    return grpc.method_handlers_generic_handler(
        service,
        {
            method: grpc.unary_unary_rpc_method_handler(
                self.predict,
                request_deserializer=predict_pb2.PredictRequest.FromString,
                response_serializer=predict_pb2.PredictResponse.SerializeToString,
            )
        },
    )


def grpc_server_options(max_message_mb: int) -> list[tuple[str, int]]:
  return [
      ('grpc.max_receive_message_length', max_message_mb * 1024 * 1024),
      ('grpc.max_send_message_length', max_message_mb * 1024 * 1024),
  ]


def build_server(
    weights_dir: str,
    port: int,
    device: str = 'cpu',
    model_name: str = common.DEFAULT_MODEL_NAME,
    num_threads: int = 8,
    dynamic_batching: bool = False,
    max_batch_size: int = 0,
    batch_timeout_micros: int = 1000,
    num_batch_threads: int = 0,
    warmup: bool = True,
    max_message_mb: int = 64,
    handler_threads: int = 0,
    max_enqueued_batches: int = 64,
) -> tuple[grpc.Server, PredictionServicer, DynamicBatcher | None]:
  """Creates (but does not start) the single-process baseline server.

  Args:
    weights_dir: Directory written by `export_benchmark_model.py`.
    port: Port to bind (0 = ephemeral).
    device: `cpu`, `cuda` or `tpu`.
    model_name: `model_spec.name` to accept.
    num_threads: Request-executing threads (gRPC handlers without batching).
    dynamic_batching: Coalesce concurrent requests.
    max_batch_size: Dynamic batching cap in examples (0 = largest bucket).
    batch_timeout_micros: Max wait for a full batch.
    num_batch_threads: Dynamic batching worker threads (0 = `num_threads`).
    warmup: Run every bucket once before returning.
    max_message_mb: gRPC message size limit.
    handler_threads: gRPC handler threads; 0 = `num_threads`, or
      `max(num_threads, DEFAULT_BATCHING_HANDLER_THREADS)` with batching so the
      batcher can see every in-flight request.
    max_enqueued_batches: Batcher admission control (see `DynamicBatcher`);
      0 = unbounded.

  Returns:
    (server, servicer, batcher or None).
  """
  model = DlrmBaseline(weights_dir, device=device)
  if warmup:
    model.warmup()
  batcher = None
  if dynamic_batching:

    def run_batch(feature_list: list[Any], batches: list[int]):
      preds = model.predict(common.concat_features(feature_list))
      return split_predictions(preds, batches)

    batcher = DynamicBatcher(
        run_batch,
        max_batch_size=max_batch_size or model.buckets[-1],
        timeout_s=batch_timeout_micros / 1e6,
        buckets=model.buckets,
        num_threads=num_batch_threads or num_threads,
        max_enqueued_batches=max_enqueued_batches,
    )
  if handler_threads <= 0:
    handler_threads = (
        max(num_threads, DEFAULT_BATCHING_HANDLER_THREADS)
        if dynamic_batching
        else num_threads
    )
  servicer = PredictionServicer(model, model_name, batcher)
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=handler_threads),
      handlers=(servicer.generic_handler(),),
      options=grpc_server_options(max_message_mb),
  )
  servicer.port = common.add_local_port(server, port)
  servicer.handler_threads = handler_threads
  return server, servicer, batcher


def wait_for_termination(stop: threading.Event) -> None:
  """Blocks until SIGTERM/SIGINT (how run_benchmark.sh stops servers)."""
  for sig in (signal.SIGTERM, signal.SIGINT):
    signal.signal(sig, lambda *_: stop.set())
  stop.wait()


def _run_single_process(f: Flags) -> None:
  restrict_jax_platforms(f.device.value)
  server, servicer, batcher = build_server(
      weights_dir=f.weights_dir.value,
      port=f.port.value,
      device=f.device.value,
      model_name=f.model_name.value,
      num_threads=f.num_threads.value,
      dynamic_batching=f.dynamic_batching.value,
      max_batch_size=f.max_batch_size.value,
      batch_timeout_micros=f.batch_timeout_micros.value,
      num_batch_threads=f.num_batch_threads.value,
      warmup=f.warmup.value,
      max_message_mb=f.max_message_mb.value,
      handler_threads=f.handler_threads.value,
      max_enqueued_batches=f.max_enqueued_batches.value,
  )
  server.start()
  logging.info(
      'Python baseline listening on port %d (device=%s, threads=%d, '
      'handler_threads=%d, dynamic_batching=%s, max_enqueued_batches=%d, '
      'pid=%d)',
      servicer.port,
      f.device.value,
      f.num_threads.value,
      servicer.handler_threads,
      f.dynamic_batching.value,
      f.max_enqueued_batches.value,
      os.getpid(),
  )
  # Printed unbuffered so orchestration scripts can wait for readiness.
  print(f'READY port={servicer.port} pid={os.getpid()}', flush=True)
  # Shut down gracefully on SIGTERM so the statistics below are reported
  # instead of dying in the signal handler.
  try:
    wait_for_termination(threading.Event())
  finally:
    server.stop(grace=2)
    if batcher is not None:
      batcher.stop()
      msg = batcher.stats_line()
      logging.info(msg)
      print(msg, flush=True)
    msg = f'Served {servicer.requests} requests / {servicer.examples} examples'
    logging.info(msg)
    print(msg, flush=True)


def main(argv: Sequence[str], f: Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  # Imported here: multiprocess_server imports this module for the model,
  # batcher and servicer classes.
  from examples.dlrm_benchmark import multiprocess_server  # pylint: disable=g-import-not-at-top

  if f.frontend_id.value is not None:
    multiprocess_server.run_frontend(f)
  elif f.num_processes.value > 1:
    multiprocess_server.run_owner(f)
  else:
    _run_single_process(f)


if __name__ == '__main__':
  app.run(functools.partial(main, f=define_flags()))

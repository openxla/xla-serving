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

"""Multiprocess gRPC load generator for `PredictionService/Predict`.

Works unchanged against the XSE server and the Python baseline.

Two load patterns:
  * closed-loop (`--concurrency=C`): C independent clients, each sends the
    next request as soon as the previous one returns. Sweeping C traces the
    throughput/latency curve up to saturation.
  * open-loop (`--qps=Q`): Poisson arrivals at rate Q regardless of server
    speed. Used at a fraction of the measured peak to report latency under
    realistic (non-saturating) load.

Design choices that keep the *client* from being the bottleneck:
  * `--num_workers` separate processes (each with its own GIL), started by
    re-executing this program in worker mode and coordinated through a
    scratch directory (start barrier, shared stop time, results).
  * `--channels_per_worker` gRPC channels (= TCP connections) per worker.
    Matters for servers that balance *connections* over processes
    (`SO_REUSEPORT`, as the multi-process Python arm does): with one
    connection per worker, 8 workers could never keep 8 server processes
    busy. Applied identically to every arm.
  * Requests are generated once per worker (`--num_distinct_requests`) and
    pre-serialised; the response is not deserialised (only status + size).
  * Only timestamps are collected; percentiles are computed at the end.

Output: one JSON document (`--output`) with throughput, latency percentiles
and run metadata. `report.py` aggregates many of them.
"""

from __future__ import annotations

import collections
import dataclasses
import functools
import itertools
import json
import os
import platform
import random
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any, Sequence

from absl import app
from absl import flags
from absl import logging
import grpc
import numpy as np

from examples.dlrm_benchmark import common


@dataclasses.dataclass(frozen=True)
class _Flags:
  """Command-line flags; only registered when run as a binary."""

  server: flags.FlagHolder[str]
  model_name: flags.FlagHolder[str]
  weights_dir: flags.FlagHolder[str | None]
  request_batch_size: flags.FlagHolder[int]
  concurrency: flags.FlagHolder[int]
  qps: flags.FlagHolder[float]
  max_inflight: flags.FlagHolder[int]
  duration_s: flags.FlagHolder[float]
  warmup_s: flags.FlagHolder[float]
  num_workers: flags.FlagHolder[int]
  channels_per_worker: flags.FlagHolder[int]
  num_distinct_requests: flags.FlagHolder[int]
  timeout_s: flags.FlagHolder[float]
  seed: flags.FlagHolder[int]
  output: flags.FlagHolder[str | None]
  labels: flags.FlagHolder[list[str]]
  worker_dir: flags.FlagHolder[str | None]
  worker_id: flags.FlagHolder[int | None]


def _define_flags() -> _Flags:
  """Registers flags lazily so tests can import `loadgen` alongside other tools."""
  return _Flags(
      server=flags.DEFINE_string('server', 'localhost:8500', 'host:port.'),
      model_name=flags.DEFINE_string(
          'model_name',
          common.DEFAULT_MODEL_NAME,
          'PredictRequest.model_spec.name.',
      ),
      weights_dir=flags.DEFINE_string(
          'weights_dir',
          None,
          'Baseline weights dir (only config.json is read, to size the '
          'requests).',
      ),
      request_batch_size=flags.DEFINE_integer(
          'request_batch_size', 16, 'Examples per request.'
      ),
      concurrency=flags.DEFINE_integer(
          'concurrency',
          0,
          'Closed-loop: total in-flight requests (0 = open-loop).',
      ),
      qps=flags.DEFINE_float(
          'qps', 0.0, 'Open-loop: Poisson arrival rate in requests/s.'
      ),
      max_inflight=flags.DEFINE_integer(
          'max_inflight',
          4096,
          'Open-loop: requests in flight before new arrivals are counted as '
          'overload errors instead of being sent.',
      ),
      duration_s=flags.DEFINE_float('duration_s', 30.0, 'Measured window.'),
      warmup_s=flags.DEFINE_float(
          'warmup_s',
          5.0,
          'Traffic sent before the measured window (discarded).',
      ),
      num_workers=flags.DEFINE_integer('num_workers', 4, 'Client processes.'),
      channels_per_worker=flags.DEFINE_integer(
          'channels_per_worker',
          1,
          'gRPC channels (TCP connections) per worker process. Closed loop: '
          'client slot s of a worker uses channel s mod k for its lifetime '
          '(so at most min(k, slots) are opened); open loop: requests '
          'round-robin over k. Servers that hash connections to processes '
          '(SO_REUSEPORT) need several per worker to be loaded evenly.',
      ),
      num_distinct_requests=flags.DEFINE_integer(
          'num_distinct_requests',
          64,
          'Pre-generated request pool size per worker.',
      ),
      timeout_s=flags.DEFINE_float('timeout_s', 30.0, 'Per-RPC deadline.'),
      seed=flags.DEFINE_integer('seed', 1234, 'Request content seed.'),
      output=flags.DEFINE_string('output', None, 'Write JSON results here.'),
      labels=flags.DEFINE_list(
          'labels',
          [],
          'key=value pairs copied into the result JSON (e.g. '
          'arm=xse-batch,platform=cuda).',
      ),
      worker_dir=flags.DEFINE_string('worker_dir', None, 'Internal.'),
      worker_id=flags.DEFINE_integer('worker_id', None, 'Internal.'),
  )


@dataclasses.dataclass
class WorkerResult:
  """Raw per-process measurements, written as JSON for the parent."""

  worker: int
  # Wall-clock completion timestamps and latencies (s) for *all* successful
  # requests, including warmup and drain; the parent slices `[t_measure_start,
  # t_measure_end]`.
  completed_at: list[float]
  latencies_s: list[float]
  # Wall-clock timestamps of failed RPCs (windowed the same way) and the
  # gRPC status code name of each (`error_codes[i]` belongs to `error_at[i]`).
  error_at: list[float]
  errors: int
  overload_drops: int
  sent: int
  cpu_s: float
  error_samples: list[str]
  connections: int = 1
  error_codes: list[str] = dataclasses.field(default_factory=list)
  overload_drops_total: int = 0


def _status_name(exc: Exception) -> str:
  """`CANCELLED`, `UNAVAILABLE`, ... for a `grpc.RpcError`; else the type."""
  code = getattr(exc, 'code', None)
  if callable(code):
    try:
      return code().name
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return type(exc).__name__


def _build_request_pool(
    config: common.ModelConfig,
    batch_size: int,
    count: int,
    seed: int,
    model_name: str,
) -> list[bytes]:
  """Pre-serialises `count` seeded `PredictRequest`s of `batch_size` rows."""
  rng = np.random.default_rng(seed)
  pool = []
  for _ in range(count):
    feats = common.generate_features(config, batch_size, rng)
    pool.append(
        common.make_predict_request(feats, model_name).SerializeToString()
    )
  return pool


def _identity(x):
  return x


def _make_stub(server: str):
  """Opens a channel and returns `(channel, raw-bytes unary stub)`."""
  channel = common.make_channel(
      server,
      options=[
          ('grpc.max_receive_message_length', 64 * 1024 * 1024),
          ('grpc.max_send_message_length', 64 * 1024 * 1024),
          # One HTTP/2 connection per channel object: without this, channels
          # with identical target/args in one process share a subchannel and
          # `--channels_per_worker` would not open more connections.
          ('grpc.use_local_subchannel_pool', 1),
      ],
  )
  stub = channel.unary_unary(
      common.PREDICT_METHOD,
      request_serializer=_identity,
      response_deserializer=_identity,
  )
  return channel, stub


_READY_FILE = 'ready.{worker}'
_GO_FILE = 'go.json'
_RESULT_FILE = 'result.{worker}.json'


def _default_worker_command() -> list[str]:
  """Command prefix that launches `loadgen` in worker mode."""
  main_mod = sys.modules.get('__main__')
  main_file = getattr(main_mod, '__file__', None)
  if main_file and os.path.abspath(main_file) == os.path.abspath(__file__):
    return common.self_command()
  if sys.executable and __spec__ is not None and __spec__.name:
    if '.' in __spec__.name:
      return [sys.executable, '-m', __spec__.name]
    return [sys.executable, os.path.abspath(__file__)]
  return common.self_command()


def _worker(worker_id: int, work_dir: str) -> None:
  """Entry point of each load generator process.

  Protocol with the parent (all via files in `work_dir`):
    1. read `args.json`, build the request pool, connect;
    2. touch `ready.<id>`;
    3. poll for `go.json` = {"measure_start": <wall>, "stop_at": <wall>} and
       start immediately;
    4. run until `stop_at`, then write `result.<id>.json`.

  Args:
    worker_id: Index of this worker.
    work_dir: Directory shared with the parent.
  """
  with open(os.path.join(work_dir, 'args.json')) as f:
    args = json.load(f)
  config = common.ModelConfig.from_json(args['config_json'])
  pool = _build_request_pool(
      config,
      args['request_batch_size'],
      args['num_distinct_requests'],
      args['seed'] + worker_id * 7919,
      args['model_name'],
  )
  slots = args['slots'][worker_id]
  num_channels = args['channels_per_worker']
  if args['concurrency'] > 0:
    num_channels = max(1, min(num_channels, slots))
  channels = []
  stubs = []
  for _ in range(num_channels):
    channel, stub = _make_stub(args['server'])
    channels.append(channel)
    stubs.append(stub)
  for channel in channels:
    grpc.channel_ready_future(channel).result(timeout=60)
  timeout_s = args['timeout_s']

  completed_at: list[float] = []
  latencies: list[float] = []
  error_at: list[float] = []
  error_codes: list[str] = []
  errors = 0
  overload_drops = 0
  overload_drops_total = 0
  sent = 0
  error_samples: list[str] = []
  lock = threading.Lock()

  def record(t0: float, exc: Exception | None) -> None:
    nonlocal errors
    t1 = time.perf_counter()
    t_wall = time.time()
    with lock:
      if exc is None:
        completed_at.append(t_wall)
        latencies.append(t1 - t0)
      else:
        errors += 1
        error_at.append(t_wall)
        error_codes.append(_status_name(exc))
        if len(error_samples) < 5:
          error_samples.append(repr(exc)[:200])

  with open(os.path.join(work_dir, _READY_FILE.format(worker=worker_id)), 'w'):
    pass
  go_path = os.path.join(work_dir, _GO_FILE)
  while not os.path.exists(go_path):
    time.sleep(0.005)
  with open(go_path) as f:
    go_data = json.load(f)
  # Convert the shared wall-clock window into this process's monotonic clock.
  now_wall = time.time()
  now_perf = time.perf_counter()
  stop_at_wall = go_data['stop_at']
  measure_start_wall = go_data.get('measure_start', stop_at_wall)
  stop_at = now_perf + (stop_at_wall - now_wall)
  measure_start = now_perf + (measure_start_wall - now_wall)
  cpu0 = time.process_time()

  if args['concurrency'] > 0:
    # Closed loop: `slots` threads, each a sequential client on one channel.
    def client_loop(slot: int) -> None:
      nonlocal sent
      stub = stubs[slot % len(stubs)]
      i = slot
      while time.perf_counter() < stop_at:
        payload = pool[i % len(pool)]
        i += 1
        t0 = time.perf_counter()
        try:
          stub(payload, timeout=timeout_s)
          record(t0, None)
        except grpc.RpcError as e:
          record(t0, e)
        with lock:
          sent += 1

    threads = [
        threading.Thread(target=client_loop, args=(s,), daemon=True)
        for s in range(slots)
    ]
    for t in threads:
      t.start()
    for t in threads:
      t.join()
  else:
    # Open loop: Poisson arrivals, fire-and-forget with completion callbacks,
    # round-robin over the channels.
    rate = args['worker_qps']
    rng = random.Random(args['seed'] + worker_id)
    inflight = 0
    max_inflight = args['max_inflight']
    next_send = time.perf_counter()
    i = 0

    def on_done(t0: float):
      def _cb(fut):
        nonlocal inflight
        try:
          fut.result()
          record(t0, None)
        except grpc.RpcError as e:
          record(t0, e)
        with lock:
          inflight -= 1

      return _cb

    while True:
      now = time.perf_counter()
      if now >= stop_at:
        break
      if now < next_send:
        time.sleep(min(next_send - now, 0.002))
        continue
      next_send += rng.expovariate(rate)
      with lock:
        if inflight >= max_inflight:
          overload_drops_total += 1
          if now >= measure_start:
            overload_drops += 1
          continue
        inflight += 1
        sent += 1
      payload = pool[i % len(pool)]
      stub = stubs[i % len(stubs)]
      i += 1
      t0 = time.perf_counter()
      fut = stub.future(payload, timeout=timeout_s)
      fut.add_done_callback(on_done(t0))

    # Drain.
    deadline = time.perf_counter() + timeout_s + 1
    while time.perf_counter() < deadline:
      with lock:
        if inflight == 0:
          break
      time.sleep(0.01)

  cpu_s = time.process_time() - cpu0
  for channel in channels:
    channel.close()
  result = WorkerResult(
      worker=worker_id,
      completed_at=completed_at,
      latencies_s=latencies,
      error_at=error_at,
      errors=errors,
      overload_drops=overload_drops,
      sent=sent,
      cpu_s=cpu_s,
      error_samples=error_samples,
      connections=num_channels,
      error_codes=error_codes,
      overload_drops_total=overload_drops_total,
  )
  out = os.path.join(work_dir, _RESULT_FILE.format(worker=worker_id))
  with open(out + '.tmp', 'w') as f:
    json.dump(dataclasses.asdict(result), f)
  os.replace(out + '.tmp', out)


def _run_workers(
    *,
    work_dir: str,
    num_workers: int,
    warmup_s: float,
    duration_s: float,
    timeout_s: float,
    describe: str,
    worker_cmd: Sequence[str] | None = None,
    worker_cwd: str | None = None,
) -> tuple[float, float, float, float, list[WorkerResult]]:
  """Launches the workers, releases them together and collects results.

  Args:
    work_dir: Scratch directory holding `args.json`.
    num_workers: Number of worker processes.
    warmup_s: Seconds of traffic before the measured window.
    duration_s: Seconds of measured traffic.
    timeout_s: Per-RPC deadline (bounds the drain time).
    describe: Human readable description for the log line.
    worker_cmd: Optional argv prefix to invoke `loadgen` in worker mode.
    worker_cwd: Optional working directory for worker processes.

  Returns:
    (t_start, t_measure_start, t_measure_end, t_end, per-worker results).
  """
  cmd = (
      list(worker_cmd) if worker_cmd is not None else _default_worker_command()
  )
  procs = [
      subprocess.Popen(
          cmd + [f'--worker_dir={work_dir}', f'--worker_id={w}'],
          cwd=worker_cwd,
          stdin=subprocess.DEVNULL,
      )
      for w in range(num_workers)
  ]
  try:
    # All workers are connected and have built their request pools when their
    # ready files exist; only then start the clock.
    ready = [
        os.path.join(work_dir, _READY_FILE.format(worker=w))
        for w in range(num_workers)
    ]
    deadline = time.time() + 300
    while not all(os.path.exists(r) for r in ready):
      for w, p in enumerate(procs):
        if p.poll() is not None:
          raise RuntimeError(
              f'loadgen worker {w} exited early ({p.returncode})'
          )
      if time.time() > deadline:
        raise RuntimeError('loadgen workers did not become ready in 300s')
      time.sleep(0.02)
    t_start = time.time()
    t_measure_start = t_start + warmup_s
    t_measure_end = t_measure_start + duration_s
    go_tmp = os.path.join(work_dir, _GO_FILE + '.tmp')
    with open(go_tmp, 'w') as f:
      json.dump(
          {'measure_start': t_measure_start, 'stop_at': t_measure_end}, f
      )
    os.replace(go_tmp, os.path.join(work_dir, _GO_FILE))
    logging.info(
        'Load started: %s workers=%d warmup=%.0fs duration=%.0fs',
        describe,
        num_workers,
        warmup_s,
        duration_s,
    )
    results: list[WorkerResult] = []
    for w, p in enumerate(procs):
      p.wait(timeout=warmup_s + duration_s + timeout_s + 120)
      path = os.path.join(work_dir, _RESULT_FILE.format(worker=w))
      if p.returncode != 0 or not os.path.exists(path):
        raise RuntimeError(f'loadgen worker {w} failed (exit {p.returncode})')
      with open(path) as f:
        results.append(WorkerResult(**json.load(f)))
    return t_start, t_measure_start, t_measure_end, time.time(), results
  finally:
    for p in procs:
      if p.poll() is None:
        p.kill()


def run_load(
    *,
    server: str,
    config: common.ModelConfig,
    request_batch_size: int,
    concurrency: int,
    qps: float,
    duration_s: float,
    warmup_s: float,
    num_workers: int,
    num_distinct_requests: int,
    timeout_s: float,
    max_inflight: int,
    seed: int,
    model_name: str,
    labels: dict[str, str],
    channels_per_worker: int = 1,
    worker_cmd: Sequence[str] | None = None,
    worker_cwd: str | None = None,
) -> dict[str, Any]:
  """Runs one load configuration and returns the result dict."""
  if duration_s <= 0:
    raise ValueError(f'duration_s must be > 0, got {duration_s}')
  if warmup_s < 0:
    raise ValueError(f'warmup_s must be >= 0, got {warmup_s}')
  if num_workers < 1:
    raise ValueError(f'num_workers must be >= 1, got {num_workers}')
  if num_distinct_requests < 1:
    raise ValueError(
        f'num_distinct_requests must be >= 1, got {num_distinct_requests}'
    )
  if timeout_s <= 0:
    raise ValueError(f'timeout_s must be > 0, got {timeout_s}')
  if concurrency < 0 or qps < 0:
    raise ValueError(
        f'concurrency and qps must be >= 0, got concurrency={concurrency}, '
        f'qps={qps}'
    )
  if (concurrency > 0) == (qps > 0):
    raise ValueError('Specify exactly one of concurrency > 0 or qps > 0.')
  if concurrency == 0 and max_inflight < 1:
    raise ValueError(f'max_inflight must be >= 1, got {max_inflight}')
  if request_batch_size < 1 or request_batch_size > config.batch_sizes[-1]:
    raise ValueError(
        f'request_batch_size={request_batch_size} must be in '
        f'[1, {config.batch_sizes[-1]}]'
    )
  if concurrency > 0:
    num_workers = max(1, min(num_workers, concurrency))
  slots = [
      concurrency // num_workers + (1 if w < concurrency % num_workers else 0)
      for w in range(num_workers)
  ]
  work_dir = tempfile.mkdtemp(prefix='loadgen_')
  with open(os.path.join(work_dir, 'args.json'), 'w') as f:
    json.dump(
        dict(
            config_json=config.to_json(),
            request_batch_size=request_batch_size,
            num_distinct_requests=num_distinct_requests,
            seed=seed,
            model_name=model_name,
            server=server,
            timeout_s=timeout_s,
            concurrency=concurrency,
            slots=slots,
            worker_qps=qps / num_workers,
            max_inflight=max_inflight,
            channels_per_worker=max(1, channels_per_worker),
        ),
        f,
    )
  try:
    t_start, t_measure_start, t_measure_end, t_end, results = _run_workers(
        work_dir=work_dir,
        num_workers=num_workers,
        warmup_s=warmup_s,
        duration_s=duration_s,
        timeout_s=timeout_s,
        describe=(
            f'{server} B={request_batch_size} '
            + (f'concurrency={concurrency}' if concurrency else f'qps={qps:g}')
        ),
        worker_cmd=worker_cmd,
        worker_cwd=worker_cwd,
    )
  finally:
    shutil.rmtree(work_dir, ignore_errors=True)

  completed = np.concatenate(
      [np.asarray(r.completed_at) for r in results] or [np.zeros(0)]
  )
  lat = np.concatenate(
      [np.asarray(r.latencies_s) for r in results] or [np.zeros(0)]
  )
  # Slice completions strictly to `[t_measure_start, t_measure_end]` so
  # post-deadline drain completions neither inflate `n / duration_s` nor mix
  # drain-down latencies into steady-state percentiles.
  measured = (completed >= t_measure_start) & (completed <= t_measure_end)
  lat_measured = lat[measured]
  n = int(measured.sum())
  error_at = np.concatenate(
      [np.asarray(r.error_at) for r in results] or [np.zeros(0)]
  )
  errors = int(
      ((error_at >= t_measure_start) & (error_at <= t_measure_end)).sum()
  )
  errors_total = sum(r.errors for r in results)
  # gRPC status codes of the failed RPCs in the measured window, most
  # frequent first. Tells server-side admission control (UNAVAILABLE) apart
  # from gRPC core shedding an overloaded server (CANCELLED) and from client
  # deadlines (DEADLINE_EXCEEDED).
  code_counts: collections.Counter[str] = collections.Counter()
  for r in results:
    for code, t in zip(r.error_codes, r.error_at):
      if t_measure_start <= t <= t_measure_end:
        code_counts[code] += 1
  error_codes = dict(code_counts.most_common())
  drops = sum(r.overload_drops for r in results)
  drops_total = sum(r.overload_drops_total for r in results)
  total_sent = sum(r.sent for r in results)
  client_cpu_s = sum(r.cpu_s for r in results)
  achieved_qps = n / duration_s
  result = {
      'server': server,
      'mode': 'closed_loop' if concurrency > 0 else 'open_loop',
      'request_batch_size': request_batch_size,
      'concurrency': concurrency,
      'target_qps': qps,
      'duration_s': duration_s,
      'warmup_s': warmup_s,
      'num_workers': num_workers,
      'channels_per_worker': max(1, channels_per_worker),
      # Total TCP connections to the server (what SO_REUSEPORT balances).
      'connections': sum(r.connections for r in results),
      'requests': n,
      'requests_sent_total': total_sent,
      # Failed RPCs inside the measured window; `errors_total` also counts
      # warmup and drain.
      'errors': errors,
      'errors_total': errors_total,
      'error_codes': error_codes,
      'overload_drops': drops,
      'overload_drops_total': drops_total,
      'qps': achieved_qps,
      'examples_per_s': achieved_qps * request_batch_size,
      'latency_ms': common.latency_summary_ms(lat_measured.tolist()),
      'client_cpu_s': client_cpu_s,
      'client_cpu_per_1k_requests_s': (
          client_cpu_s / max(1, total_sent) * 1000
      ),
      'wall_s': t_end - t_start,
      'error_samples': list(
          itertools.chain.from_iterable(r.error_samples for r in results)
      )[:5],
      'embedding_size': config.embedding_size,
      'batch_sizes': list(config.batch_sizes),
      'client_host': socket.gethostname(),
      'client_platform': platform.platform(),
      'client_cpu_affinity': (
          sorted(os.sched_getaffinity(0))
          if hasattr(os, 'sched_getaffinity')
          else None
      ),
      'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S%z'),
      'labels': labels,
  }
  return result


def _parse_labels(items: Sequence[str]) -> dict[str, str]:
  labels = {}
  for item in items:
    if '=' not in item:
      raise app.UsageError(f'--labels entries must be key=value, got {item!r}')
    k, v = item.split('=', 1)
    labels[k.strip()] = v.strip()
  return labels


def main(argv: Sequence[str], f: _Flags) -> None:
  if len(argv) > 1:
    raise app.UsageError(f'Unexpected arguments: {argv[1:]}')
  if f.worker_dir.value is not None:
    if f.worker_id.value is None:
      raise app.UsageError('--worker_id is required with --worker_dir')
    _worker(f.worker_id.value, f.worker_dir.value)
    return
  if not f.weights_dir.value:
    raise app.UsageError('--weights_dir is required (reads config.json).')
  if f.concurrency.value < 0 or f.qps.value < 0:
    raise app.UsageError('--concurrency and --qps must be >= 0.')
  if (f.concurrency.value > 0) == (f.qps.value > 0):
    raise app.UsageError('Specify exactly one of --concurrency or --qps.')
  if f.duration_s.value <= 0:
    raise app.UsageError('--duration_s must be > 0.')
  if f.warmup_s.value < 0:
    raise app.UsageError('--warmup_s must be >= 0.')
  if f.num_workers.value < 1:
    raise app.UsageError('--num_workers must be >= 1.')
  if f.num_distinct_requests.value < 1:
    raise app.UsageError('--num_distinct_requests must be >= 1.')
  if f.timeout_s.value <= 0:
    raise app.UsageError('--timeout_s must be > 0.')
  if f.concurrency.value == 0 and f.max_inflight.value < 1:
    raise app.UsageError('--max_inflight must be >= 1.')
  config = common.ModelConfig.load(f.weights_dir.value)
  if (
      f.request_batch_size.value < 1
      or f.request_batch_size.value > config.batch_sizes[-1]
  ):
    raise app.UsageError(
        f'--request_batch_size={f.request_batch_size.value} must be in '
        f'[1, {config.batch_sizes[-1]}]'
    )
  result = run_load(
      server=f.server.value,
      config=config,
      request_batch_size=f.request_batch_size.value,
      concurrency=f.concurrency.value,
      qps=f.qps.value,
      duration_s=f.duration_s.value,
      warmup_s=f.warmup_s.value,
      num_workers=f.num_workers.value,
      channels_per_worker=f.channels_per_worker.value,
      num_distinct_requests=f.num_distinct_requests.value,
      timeout_s=f.timeout_s.value,
      max_inflight=f.max_inflight.value,
      seed=f.seed.value,
      model_name=f.model_name.value,
      labels=_parse_labels(f.labels.value),
  )
  lat = result['latency_ms']
  codes = ','.join(f'{k}:{v}' for k, v in result['error_codes'].items())
  errors = f"errors={result['errors']}" + (f' ({codes})' if codes else '')
  summary = (
      f"{result['mode']} B={result['request_batch_size']}"
      f" {'C=' + str(result['concurrency']) if result['concurrency'] else 'qps=' + format(result['target_qps'], 'g')}:"
      f" {result['qps']:.1f} req/s, {result['examples_per_s']:.0f} ex/s,"
      f" p50={lat['p50']:.2f}ms p90={lat['p90']:.2f}ms p99={lat['p99']:.2f}ms"
      f" max={lat['max']:.1f}ms {errors} drops={result['overload_drops']}"
  )
  logging.info(summary)
  print(summary, flush=True)
  if result['error_samples']:
    logging.warning('Error samples: %s', result['error_samples'])
  if f.output.value:
    os.makedirs(
        os.path.dirname(os.path.abspath(f.output.value)), exist_ok=True
    )
    with open(f.output.value, 'w') as out_f:
      json.dump(result, out_f, indent=2, sort_keys=True)
    logging.info('Wrote %s', f.output.value)
  if result['requests'] == 0:
    sys.exit(2)


if __name__ == '__main__':
  app.run(functools.partial(main, f=_define_flags()))

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

"""Multi-process Python baseline (`python_server --num_processes=N`).

Answers the obvious objection to the single-process Python arm: "the GIL caps
it at ~2 cores; what if several Python processes shared one gRPC port?". This
is the strongest such design that still works on a single accelerator:

  * one accelerator chip can be opened by exactly one process (libtpu holds a
    process-wide lock on the chip; a second CUDA context would compete for
    the GPU's memory), so the model must run in one place;
  * everything *else* -- gRPC parsing/serialisation and the 26-table numpy
    embedding gather, i.e. all the GIL-bound work -- is spread over N
    processes.

Layout: N frontends + 1 device owner, all started from one command line.

  owner (this process, owns the device):
    1. create the IPC dir, one shared-memory ring file per frontend and a
       UNIX SOCK_SEQPACKET listener;
    2. re-exec itself N times as `python_server ... --frontend_id=i`;
    3. load params, init JAX, warm up (frontends load tables meanwhile);
    4. wait for N x HELLO + LOADED; BIND frontend 0 on --port, read the port
       it got back (so --port=0 works) and BIND the rest on that port;
    5. print `READY port=... pid=...`;
    6. one reader thread per frontend: REQ(req_id, off, rows, batch) ->
       run_model on ring.emb[off:off+rows] (a --num_threads pool), or with
       --dynamic_batching merger.submit(...) (see below) -> write
       ring.out[off:off+batch] -> DONE(req_id);
    7. SIGTERM: STOP every frontend, collect STATS, print per-frontend and
       total request/example (and batcher) counts, clean up.

  frontend i (re-exec'd `python_server --frontend_id=i`):
    1. attach ring i, connect to the listener, HELLO; load its own private
       copy of the tables, LOADED;
    2. wait for BIND(port) -> grpc.server(...).add_local_port with
       SO_REUSEPORT -> start -> STARTED(port);
    3. handler thread per request: parse -> validate -> either (no batching)
       reserve a bucket of rows in the ring -> numpy gather into
       ring.emb/ring.dense -> REQ -> wait -> copy ring.out -> reply; or
       (--dynamic_batching) batcher.submit(features) and park on the future;
       the frontend's batcher thread(s) then do exactly the above once per
       *batch* (one gather for all coalesced requests) and scatter the
       predictions back to the parked handlers;
    4. dispatcher thread: DONE(req_id) -> wake the waiting thread;
    5. STOP -> server.stop(grace) -> STATS -> exit.

Why it looks the way it does:

  * "sharing one acceptor" in gRPC means `SO_REUSEPORT`: N processes bind the
    same port and the kernel hashes each *connection* to one of them. Load is
    therefore balanced per connection, not per request, which is why the load
    generator opens several connections per worker (`--channels_per_worker`)
    and why the frontends start only after *all* of them are bound (otherwise
    every persistent client connection would land on the first one up).
  * Frontends are re-executed copies rather than forks: gRPC starts threads at
    import time, so forked children cannot use it. Each therefore loads its own
    copy of the tables (N x 3.2 GB at full size). They could share one
    read-only mapping, but a file-backed mapping loses the transparent huge
    pages numpy's anonymous allocation gets, which measurably slows the random
    gather -- an unfair handicap, so the RAM is spent instead.
  * Data crosses processes through a per-frontend shared-memory ring
    (embeddings and dense features in, predictions out); only fixed-size
    records go over the socket. The frontend always reserves a whole bucket
    and zeroes the padding rows, so the owner can `device_put` the padded
    view with no copy.
  * Batching happens in the frontends, *before* the gather -- the same
    placement as the single-process `--dynamic_batching` arm and as XSE's
    `PRE_PROCESSOR_AND_MODEL_FUNCTION` (one lookup per batch, not per
    request). It reuses `python_server.DynamicBatcher` verbatim, so timeout,
    cap, no-splitting, pad-up and the `max_enqueued_batches` admission
    control are the same as in the single-process arm; the per-frontend
    batcher threads and queue budget are `--num_batch_threads` and
    `--max_enqueued_batches` divided by N. Because handler threads only parse
    and park while one or two batcher threads per process run the NumPy
    gathers, gRPC-Python's single GIL-bound completion thread is never starved
    under overload.
  * The owner still has a `DynamicBatcher`, but with `timeout_s=0`: it merges
    whatever frontend batches are *already* waiting for the device into one
    launch (up to `--max_batch_size`) and never waits for more. A frontend
    sees only 1/N of the traffic, so at request batch 1 its batches would be
    tiny and the accelerator's launch rate, not its throughput, would bind;
    XSE's single scheduler sees all the traffic at once, this merger restores
    that without adding latency. A batch from a single frontend runs on its
    (already padded) ring rows directly; merging several copies them into one
    padded array first.
"""

from __future__ import annotations

import collections
from concurrent import futures
import contextlib
import dataclasses
import json
import mmap
import os
import shutil
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from absl import logging
import grpc
import numpy as np

from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import python_server

SOCKET_NAME = 'owner.sock'
RING_NAME = 'ring.{frontend}'

# Record kinds on the SEQPACKET socket (one datagram per record).
_HELLO = 1  # frontend -> owner: <BI  frontend_id
_LOADED = 2  # frontend -> owner: <B
_BIND = 3  # owner -> frontend: <BI  port (0 = ephemeral)
_STARTED = 4  # frontend -> owner: <BI  bound port (0 = failed)
_REQ = 5  # frontend -> owner: <BIIII req_id, row offset, rows, batch
_DONE = 6  # owner -> frontend: <BII req_id, status (0 = ok)
_STOP = 7  # owner -> frontend: <B
_STATS = 8  # frontend -> owner: <B + json

_REQ_FMT = '<BIIII'
_DONE_FMT = '<BII'
_ID_FMT = '<BI'
_MAX_RECORD = 1 << 16


def _send(sock: socket.socket, lock: threading.Lock, data: bytes) -> None:
  with lock:
    sock.sendall(data)


def _recv(sock: socket.socket) -> bytes:
  """Receives one record; b'' when the peer is gone."""
  try:
    return sock.recv(_MAX_RECORD)
  except (ConnectionResetError, OSError):
    return b''


# --------------------------------------------------------------------------- #
# Shared-memory ring
# --------------------------------------------------------------------------- #


def _bytes_per_row(num_tables: int, dim: int, num_dense: int) -> int:
  return (num_tables * dim + num_dense + 1) * 4


class Ring:
  """Row-indexed shared arrays: emb (rows, T, D), dense (rows, F), out (rows,).

  Backed by one file (on /dev/shm by default) mapped by the owner and by one
  frontend; both see the same pages. Allocation policy lives in
  `RowAllocator` (frontend side only, single producer).
  """

  def __init__(
      self,
      path: str,
      rows: int,
      num_tables: int,
      dim: int,
      num_dense: int,
      create: bool,
  ):
    self.rows = rows
    emb_bytes = rows * num_tables * dim * 4
    dense_bytes = rows * num_dense * 4
    out_bytes = rows * 4
    size = emb_bytes + dense_bytes + out_bytes
    # O_TRUNC (not O_EXCL): a stale file left by a crashed run in a
    # user-supplied --ipc_dir must not prevent the next run from starting.
    flags = os.O_RDWR | (os.O_CREAT | os.O_TRUNC if create else 0)
    fd = os.open(path, flags, 0o600)
    try:
      if create:
        os.ftruncate(fd, size)
      self._mm = mmap.mmap(fd, size)
    finally:
      os.close(fd)
    self.emb = np.ndarray(
        (rows, num_tables, dim), np.float32, buffer=self._mm, offset=0
    )
    self.dense = np.ndarray(
        (rows, num_dense), np.float32, buffer=self._mm, offset=emb_bytes
    )
    self.out = np.ndarray(
        (rows,), np.float32, buffer=self._mm, offset=emb_bytes + dense_bytes
    )

  @classmethod
  def rows_for(cls, config: common.ModelConfig, ring_mb: int) -> int:
    rows = (
        ring_mb
        * 2**20
        // _bytes_per_row(
            config.num_tables, config.embedding_size, config.num_dense_features
        )
    )
    # At least two of the largest bucket so one batch can be formed while
    # another is on the device.
    return max(rows, 2 * config.batch_sizes[-1])

  @classmethod
  def for_config(
      cls, path: str, config: common.ModelConfig, rows: int, create: bool
  ) -> 'Ring':
    return cls(
        path,
        rows,
        config.num_tables,
        config.embedding_size,
        config.num_dense_features,
        create,
    )


@dataclasses.dataclass
class _Segment:
  off: int
  rows: int
  done: bool = False


class RowAllocator:
  """Contiguous row ranges from a circular buffer; frees in any order.

  Single producer process (handler threads of one frontend), so a plain lock
  and condition variable are enough. `reserve` blocks while the ring is full
  (backpressure on the gRPC handler); `release` marks a segment done and
  advances the tail over the done prefix, so space is reclaimed in FIFO order
  even when completions arrive out of order (several device threads).
  """

  def __init__(self, capacity: int):
    if capacity <= 0:
      raise ValueError('capacity must be positive')
    self._capacity = capacity
    self._cv = threading.Condition()
    self._live: collections.deque[_Segment] = collections.deque()
    self._by_off: dict[int, _Segment] = {}
    self._head = 0

  @property
  def capacity(self) -> int:
    return self._capacity

  def _fit(self, rows: int) -> int | None:
    """Returns a contiguous offset for `rows` if available, else `None`."""
    if not self._live:
      self._head = 0
      return 0
    tail = self._live[0].off
    head = self._head
    if head == tail:
      return None  # full (live region is the whole ring)
    if head > tail:  # live = [tail, head); free = [head, cap) + [0, tail)
      if head + rows <= self._capacity:
        return head
      if rows < tail:
        return 0  # wrap; the rows at the end stay unused until tail wraps
      return None
    # live = [tail, cap) + [0, head); free = [head, tail). Strictly less so
    # that head == tail always means "full".
    if head + rows < tail:
      return head
    return None

  def reserve(self, rows: int, timeout_s: float | None = None) -> int:
    """Returns the first row of a contiguous range of `rows`; blocks if full."""
    if rows <= 0 or rows > self._capacity:
      raise ValueError(
          f'Cannot reserve {rows} rows in a {self._capacity}-row ring'
      )
    deadline = None if timeout_s is None else time.monotonic() + timeout_s
    with self._cv:
      while True:
        off = self._fit(rows)
        if off is not None:
          seg = _Segment(off, rows)
          self._live.append(seg)
          self._by_off[off] = seg
          self._head = (off + rows) % self._capacity
          return off
        remaining = None if deadline is None else deadline - time.monotonic()
        if remaining is not None and remaining <= 0:
          raise TimeoutError('shared-memory ring is full')
        self._cv.wait(timeout=remaining)

  def release(self, off: int) -> None:
    with self._cv:
      seg = self._by_off.pop(off)
      seg.done = True
      while self._live and self._live[0].done:
        self._live.popleft()
      self._cv.notify_all()

  @property
  def live_segments(self) -> int:
    with self._cv:
      return len(self._live)


# --------------------------------------------------------------------------- #
# Frontend
# --------------------------------------------------------------------------- #


class _RemoteModel(python_server.Predictor):
  """`predict()` = local numpy lookup into the ring + round trip to the owner.

  Plugged into the unchanged `PredictionServicer` (directly, or as the
  `run_fn` of a frontend-side `DynamicBatcher`), so request parsing,
  validation, error mapping and counting are exactly those of the
  single-process server. `features` may be one request or the concatenation
  of a whole batch; either way a full bucket of ring rows is reserved and the
  padding rows are zeroed so the owner can run the view without copying.
  """

  def __init__(
      self,
      config: common.ModelConfig,
      tables: list[np.ndarray],
      ring: Ring,
      sock: socket.socket,
      timeout_s: float = 120.0,
  ):
    self.config = config
    self._tables = tables
    self._ring = ring
    self._alloc = RowAllocator(ring.rows)
    self._sock = sock
    self._send_lock = threading.Lock()
    self._buckets = tuple(config.batch_sizes)
    self._timeout_s = timeout_s
    self._pending_lock = threading.Lock()
    self._pending: dict[int, tuple[threading.Event, list[int]]] = {}
    self._next_id = 0

  def complete(self, req_id: int, status: int) -> None:
    with self._pending_lock:
      entry = self._pending.pop(req_id, None)
    if entry is not None:
      entry[1][0] = status
      entry[0].set()

  def predict_validated(self, features, batch: int) -> np.ndarray:
    rows = common.next_bucket(batch, self._buckets)
    off = self._alloc.reserve(rows, timeout_s=self._timeout_s)
    try:
      emb = self._ring.emb[off : off + rows]
      dense = self._ring.dense[off : off + rows]
      common.embedding_lookup_stacked(self._tables, features, out=emb[:batch])
      dense[:batch] = features[common.DENSE_FEATURE_NAME]
      if rows > batch:  # whole-bucket reservation: zero the padding rows
        emb[batch:] = 0
        dense[batch:] = 0
      event = threading.Event()
      status = [1]
      with self._pending_lock:
        req_id = self._next_id
        self._next_id = (self._next_id + 1) & 0xFFFFFFFF
        self._pending[req_id] = (event, status)
      _send(
          self._sock,
          self._send_lock,
          struct.pack(_REQ_FMT, _REQ, req_id, off, rows, batch),
      )
      if not event.wait(self._timeout_s):
        with self._pending_lock:
          self._pending.pop(req_id, None)
        raise RuntimeError(
            f'device owner did not answer within {self._timeout_s}s'
        )
      if status[0] != 0:
        raise RuntimeError('device owner failed to run the model')
      return self._ring.out[off : off + batch].copy()
    finally:
      self._alloc.release(off)

  def predict(self, features) -> np.ndarray:
    return self.predict_validated(features, common.batch_size_of(features))


def run_frontend(f: python_server.Flags) -> None:
  """Entry point of a frontend process (`--frontend_id` set by the owner)."""
  fid = f.frontend_id.value
  ipc_dir = f.ipc_dir.value
  if fid is None or not ipc_dir:
    raise ValueError('--frontend_id requires --ipc_dir')
  # Frontends never touch the accelerator (and on a TPU VM must not try to).
  python_server.restrict_jax_platforms('cpu')
  t0 = time.perf_counter()
  config, tables, _ = common.load_baseline_weights(
      f.weights_dir.value, params=False
  )
  logging.info(
      'frontend %d: loaded %d tables in %.1fs (pid %d)',
      fid,
      len(tables),
      time.perf_counter() - t0,
      os.getpid(),
  )
  ring_path = os.path.join(ipc_dir, RING_NAME.format(frontend=fid))
  rows = os.path.getsize(ring_path) // _bytes_per_row(
      config.num_tables, config.embedding_size, config.num_dense_features
  )
  ring = Ring.for_config(ring_path, config, rows, create=False)

  sock = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
  sock.connect(os.path.join(ipc_dir, SOCKET_NAME))
  sock.sendall(struct.pack(_ID_FMT, _HELLO, fid))

  model = _RemoteModel(config, tables, ring, sock)
  n = max(1, f.num_processes.value)
  batcher = None
  if f.dynamic_batching.value:

    def run_batch(feature_list: list[Any], batches: list[int]):
      # One gather + one owner round trip for the whole batch (the same
      # `run_fn` shape as python_server.build_server).
      preds = model.predict_validated(
          common.concat_features(feature_list), sum(batches)
      )
      return python_server.split_predictions(preds, batches)

    batcher = python_server.DynamicBatcher(
        run_batch,
        max_batch_size=f.max_batch_size.value or config.batch_sizes[-1],
        timeout_s=f.batch_timeout_micros.value / 1e6,
        buckets=config.batch_sizes,
        # This frontend's share of the batch threads and of the queue budget
        # (XSE has one scheduler with the full budget; N frontends get 1/N
        # each so the totals match).
        num_threads=python_server.split_across(
            f.num_batch_threads.value or f.num_threads.value, n
        ),
        max_enqueued_batches=python_server.split_across(
            f.max_enqueued_batches.value, n
        ),
    )
  servicer = python_server.PredictionServicer(
      model, f.model_name.value, batcher
  )
  handler_threads = f.handler_threads.value
  if handler_threads <= 0:
    handler_threads = f.num_threads.value
    if f.dynamic_batching.value:
      # Same total parked-handler budget as the single-process batched arm,
      # split across the frontends (see python_server --handler_threads).
      # Handlers only parse and park here, so the width is harmless to the
      # gRPC event thread.
      per_frontend = python_server.split_across(
          python_server.DEFAULT_BATCHING_HANDLER_THREADS, n
      )
      handler_threads = max(handler_threads, per_frontend)
  server = grpc.server(
      futures.ThreadPoolExecutor(max_workers=handler_threads),
      handlers=(servicer.generic_handler(),),
      options=python_server.grpc_server_options(f.max_message_mb.value)
      + [('grpc.so_reuseport', 1)],
  )
  sock.sendall(struct.pack('<B', _LOADED))

  # The owner releases the frontends together once all have loaded.
  msg = _recv(sock)
  if len(msg) != struct.calcsize(_ID_FMT) or msg[0] != _BIND:
    logging.error('frontend %d: expected BIND, got %r; exiting', fid, msg)
    sys.exit(1)
  port = struct.unpack(_ID_FMT, msg)[1]
  bound = common.add_local_port(server, port)
  if not bound:
    sock.sendall(struct.pack(_ID_FMT, _STARTED, 0))
    logging.error('frontend %d: could not bind port %d', fid, port)
    sys.exit(1)

  stop = threading.Event()

  def dispatch() -> None:
    while True:
      msg = _recv(sock)
      if not msg:
        if not stop.is_set():
          logging.error('frontend %d: owner went away; exiting', fid)
          os._exit(1)  # pylint: disable=protected-access
        return
      kind = msg[0]
      if kind == _DONE:
        _, req_id, status = struct.unpack(_DONE_FMT, msg)
        model.complete(req_id, status)
      elif kind == _STOP:
        stop.set()
        return

  # Start listening for DONE/STOP before opening the gRPC listener so early
  # requests can complete immediately.
  threading.Thread(target=dispatch, name='dispatch', daemon=True).start()
  server.start()
  sock.sendall(struct.pack(_ID_FMT, _STARTED, bound))
  batcher_info = ''
  if batcher is not None:
    batcher_info = (
        f', {batcher.num_threads} batcher threads, '
        f'max_enqueued_batches={batcher.max_enqueued_batches}'
    )
  logging.info(
      'frontend %d: serving on port %d with %d handler threads%s (pid %d)',
      fid,
      bound,
      handler_threads,
      batcher_info,
      os.getpid(),
  )
  # Normally stopped by the owner's STOP record; a SIGINT/SIGTERM to the
  # whole process group (Ctrl-C) should end the same way.
  python_server.wait_for_termination(stop)
  server.stop(grace=2).wait()
  stats_dict: dict[str, int] = {
      'requests': servicer.requests,
      'examples': servicer.examples,
  }
  if batcher is not None:
    batcher.stop()
    stats_dict.update(
        batches=batcher.batches_run,
        examples_run=batcher.examples_run,
        padded_examples=batcher.padded_examples,
        rejected=batcher.rejected,
    )
  stats = json.dumps(stats_dict).encode()
  with contextlib.suppress(OSError):
    sock.sendall(struct.pack('<B', _STATS) + stats)
  sock.close()


# --------------------------------------------------------------------------- #
# Device owner
# --------------------------------------------------------------------------- #


@dataclasses.dataclass
class _Frontend:
  index: int
  proc: subprocess.Popen[bytes]
  ring: Ring
  sock: socket.socket | None = None
  send_lock: threading.Lock = dataclasses.field(default_factory=threading.Lock)
  stats: dict[str, int] | None = None

  def send(self, data: bytes) -> None:
    assert self.sock is not None
    _send(self.sock, self.send_lock, data)


@dataclasses.dataclass(frozen=True)
class _Work:
  """One frontend request as seen by the owner (rows live in `fe.ring`)."""

  fe: _Frontend
  req_id: int
  off: int
  rows: int


def _default_ipc_dir() -> str:
  # /dev/shm keeps the rings in RAM; /tmp is the fallback (the rings are
  # ordinary files, page-cache shared, so any filesystem works). Both are
  # short, which matters: AF_UNIX socket paths are limited to 108 bytes and
  # $TMPDIR in a test sandbox can be longer than that.
  for base in ('/dev/shm', '/tmp', tempfile.gettempdir()):
    if os.path.isdir(base) and os.access(base, os.W_OK):
      return tempfile.mkdtemp(prefix=f'dlrm_mp_{os.getpid()}_', dir=base)
  return tempfile.mkdtemp(prefix=f'dlrm_mp_{os.getpid()}_')


class _Owner:
  """Owns the accelerator; runs the model for N frontends."""

  _model: python_server.DlrmBaseline

  def __init__(self, f: python_server.Flags):
    self._f = f
    self._n = f.num_processes.value
    self._batched = f.dynamic_batching.value
    self._stop = threading.Event()
    self._failed: bool = False
    self._frontends: list[_Frontend] = []
    self._readers: list[threading.Thread] = []
    self._ipc_dir = ''
    self._own_ipc_dir: bool = False
    self._listener: socket.socket | None = None
    self._pool: futures.ThreadPoolExecutor | None = None
    self._batcher: python_server.DynamicBatcher | None = None
    # Frontend batches the merger has run (vs. its `batches_run` launches).
    self._stats_lock = threading.Lock()
    self._merged_items = 0

  # ---- startup ---------------------------------------------------------- #

  def start(self) -> int:
    """Starts everything; returns the bound port."""
    f = self._f
    config = common.ModelConfig.load(f.weights_dir.value)

    # IPC first and frontends right after, so that their table loads overlap
    # with this process's JAX initialisation and warm-up.
    if f.ipc_dir.value:
      self._ipc_dir = f.ipc_dir.value
      os.makedirs(self._ipc_dir, exist_ok=True)
      with contextlib.suppress(FileNotFoundError):
        os.unlink(os.path.join(self._ipc_dir, SOCKET_NAME))
    else:
      self._ipc_dir = _default_ipc_dir()
      self._own_ipc_dir = True
    self._listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    self._listener.bind(os.path.join(self._ipc_dir, SOCKET_NAME))
    self._listener.listen(self._n)

    rows = Ring.rows_for(config, f.ring_mb.value)
    logging.info(
        'owner: %d frontends, ring %d rows (%.0f MiB) each, ipc_dir=%s',
        self._n,
        rows,
        rows
        * _bytes_per_row(
            config.num_tables, config.embedding_size, config.num_dense_features
        )
        / 2**20,
        self._ipc_dir,
    )
    cmd = common.self_command() + sys.argv[1:] + [f'--ipc_dir={self._ipc_dir}']
    for i in range(self._n):
      ring = Ring.for_config(
          os.path.join(self._ipc_dir, RING_NAME.format(frontend=i)),
          config,
          rows,
          create=True,
      )
      proc = subprocess.Popen(
          cmd + [f'--frontend_id={i}'], stdin=subprocess.DEVNULL
      )
      self._frontends.append(_Frontend(i, proc, ring))

    python_server.restrict_jax_platforms(f.device.value)
    self._model = python_server.DlrmBaseline(
        f.weights_dir.value, device=f.device.value, load_tables=False
    )
    if f.warmup.value:
      self._model.warmup()

    # HELLO + LOADED from everyone (they arrive as each frontend finishes
    # loading its tables), then release them together.
    self._listener.settimeout(0.5)
    deadline = time.monotonic() + 1800
    connected = 0
    while connected < self._n:
      self._check_children_alive()
      if time.monotonic() > deadline:
        raise RuntimeError('frontends did not start within 30 minutes')
      try:
        conn, _ = self._listener.accept()
      except socket.timeout:
        continue
      hello = _recv(conn)
      if len(hello) != struct.calcsize(_ID_FMT) or hello[0] != _HELLO:
        raise RuntimeError(f'bad HELLO from a frontend: {hello!r}')
      fe = self._frontends[struct.unpack(_ID_FMT, hello)[1]]
      fe.sock = conn
      loaded = _recv(conn)
      if not loaded or loaded[0] != _LOADED:
        raise RuntimeError(f'frontend {fe.index} failed before LOADED')
      connected += 1

    # Prepare the execution backend before any frontend opens the gRPC port,
    # and start each frontend's IPC reader immediately once it reports STARTED.
    if self._batched:
      # Opportunistic merger (see the module docstring): the frontends have
      # already batched with the real timeout, so this one never waits; it
      # just runs together whatever frontend batches are queued right now.
      # No admission bound: the frontends' batchers already applied it.
      self._batcher = python_server.DynamicBatcher(
          self._run_batch,
          max_batch_size=f.max_batch_size.value or config.batch_sizes[-1],
          timeout_s=0.0,
          buckets=config.batch_sizes,
          num_threads=f.num_batch_threads.value or f.num_threads.value,
      )
    else:
      self._pool = futures.ThreadPoolExecutor(
          max_workers=f.num_threads.value, thread_name_prefix='device'
      )

    # Frontend 0 binds first (possibly an ephemeral port), the others join
    # that port; SO_REUSEPORT lets them all listen on it.
    port = self._bind_and_start_reader(self._frontends[0], f.port.value)
    for fe in self._frontends[1:]:
      self._bind_and_start_reader(fe, port)

    threading.Thread(
        target=self._watchdog, name='watchdog', daemon=True
    ).start()
    return port

  def _bind_and_start_reader(self, fe: _Frontend, port: int) -> int:
    bound = self._bind(fe, port)
    t = threading.Thread(
        target=self._reader,
        args=(fe,),
        name=f'reader-{fe.index}',
        daemon=True,
    )
    t.start()
    self._readers.append(t)
    return bound

  def _bind(self, fe: _Frontend, port: int) -> int:
    fe.send(struct.pack(_ID_FMT, _BIND, port))
    assert fe.sock is not None
    fe.sock.settimeout(60)
    msg = _recv(fe.sock)
    fe.sock.settimeout(None)
    if len(msg) != struct.calcsize(_ID_FMT) or msg[0] != _STARTED:
      raise RuntimeError(f'frontend {fe.index} did not start: {msg!r}')
    bound = struct.unpack(_ID_FMT, msg)[1]
    if not bound:
      raise RuntimeError(f'frontend {fe.index} could not bind port {port}')
    return bound

  def _check_children_alive(self) -> None:
    for fe in self._frontends:
      if fe.proc.poll() is not None:
        raise RuntimeError(
            f'frontend {fe.index} exited with {fe.proc.returncode} during'
            ' startup'
        )

  def _watchdog(self) -> None:
    while not self._stop.wait(0.5):
      for fe in self._frontends:
        if fe.proc.poll() is not None:
          logging.error(
              'frontend %d died (exit %d); shutting down',
              fe.index,
              fe.proc.returncode,
          )
          self._failed = True
          self._stop.set()
          return

  # ---- request path ------------------------------------------------------ #

  def _reader(self, fe: _Frontend) -> None:
    """Reads request and shutdown-stats records from one frontend socket."""
    assert fe.sock is not None
    req_size = struct.calcsize(_REQ_FMT)
    while True:
      msg = _recv(fe.sock)
      if not msg:
        if not self._stop.is_set():
          logging.error('frontend %d closed its socket', fe.index)
          self._failed = True
          self._stop.set()
        return
      kind = msg[0]
      if kind == _REQ and len(msg) == req_size:
        _, req_id, off, rows, batch = struct.unpack(_REQ_FMT, msg)
        work = _Work(fe, req_id, off, rows)
        if self._batcher is not None:
          self._batcher.submit(work, batch)
        else:
          assert self._pool is not None
          self._pool.submit(self._run_one, work, batch)
      elif kind == _STATS:
        fe.stats = json.loads(msg[1:].decode())
      else:
        logging.warning('frontend %d: unexpected record %r', fe.index, msg[:8])

  def _run_one(self, work: _Work, batch: int) -> None:
    """Executes one unmerged request slice directly out of `work.fe.ring`."""
    ring = work.fe.ring
    status = 0
    try:
      preds = self._model.run_model(
          ring.emb[work.off : work.off + work.rows],
          ring.dense[work.off : work.off + work.rows],
          batch,
      )
      ring.out[work.off : work.off + batch] = preds
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('model call failed')
      status = 1
    work.fe.send(struct.pack(_DONE_FMT, _DONE, work.req_id, status))

  def _run_batch(self, items: list[Any], batches: list[int]) -> list[None]:
    """Merger `run_fn`: run one frontend batch in place, or merge several.

    Args:
      items: `_Work` records, one per frontend batch (bucket-padded rows).
      batches: Real example count of each item.

    Returns:
      One `None` per item (the predictions go back through the ring).
    """
    with self._stats_lock:
      self._merged_items += len(items)
    if len(items) == 1:
      self._run_one(items[0], batches[0])
      return [None]
    config = self._model.config
    total = sum(batches)
    target = common.next_bucket(total, config.batch_sizes)
    status = 0
    try:
      emb = np.empty(
          (target, config.num_tables, config.embedding_size), np.float32
      )
      dense = np.empty((target, config.num_dense_features), np.float32)
      o = 0
      for work, b in zip(items, batches):
        emb[o : o + b] = work.fe.ring.emb[work.off : work.off + b]
        dense[o : o + b] = work.fe.ring.dense[work.off : work.off + b]
        o += b
      if target > total:
        emb[total:] = 0
        dense[total:] = 0
      preds = self._model.run_model(emb, dense, total)
      o = 0
      for work, b in zip(items, batches):
        work.fe.ring.out[work.off : work.off + b] = preds[o : o + b]
        o += b
    except Exception:  # pylint: disable=broad-exception-caught
      logging.exception('batched model call failed')
      status = 1
    for work in items:
      work.fe.send(struct.pack(_DONE_FMT, _DONE, work.req_id, status))
    return [None] * len(items)

  # ---- shutdown ----------------------------------------------------------- #

  def wait(self) -> None:
    python_server.wait_for_termination(self._stop)

  def shutdown(self) -> bool:
    """Stops frontends, prints statistics; returns True if all went well."""
    self._stop.set()
    for fe in self._frontends:
      if fe.proc.poll() is None:
        if fe.sock is not None:
          with contextlib.suppress(OSError):
            fe.send(struct.pack('<B', _STOP))
        else:
          fe.proc.terminate()
    deadline = time.monotonic() + 15
    for fe in self._frontends:
      remaining = max(0.1, deadline - time.monotonic())
      try:
        fe.proc.wait(timeout=remaining)
      except subprocess.TimeoutExpired:
        fe.proc.kill()
        fe.proc.wait()
    # Readers return on the frontend's EOF, after its STATS record.
    for t in self._readers:
      t.join(timeout=max(0.1, deadline - time.monotonic()))
    if self._batcher is not None:
      self._batcher.stop()
    if self._pool is not None:
      self._pool.shutdown(wait=False)

    def emit(msg: str) -> None:
      logging.info(msg)
      print(msg, flush=True)

    total_req = total_ex = 0
    batch_keys = ('batches', 'examples_run', 'padded_examples', 'rejected')
    totals = dict.fromkeys(batch_keys, 0)
    for fe in self._frontends:
      s = fe.stats or {'requests': 0, 'examples': 0}
      total_req += s['requests']
      total_ex += s['examples']
      emit(
          f"Frontend {fe.index} served {s['requests']} requests / "
          f"{s['examples']} examples"
      )
      if self._batched and 'batches' in s:
        for k in batch_keys:
          totals[k] += s.get(k, 0)
        emit(
            f"Frontend {fe.index} batcher: {s['batches']} batches, "
            f"{s['examples_run']} examples, {s['padded_examples']} padded "
            f"examples, {s.get('rejected', 0)} rejected"
        )
    if self._batcher is not None:
      # Summed over the frontends: the same line (and meaning) as the
      # single-process batched arm prints.
      emit(
          f"Batcher stats: {totals['batches']} batches, "
          f"{totals['examples_run']} examples, "
          f"{totals['padded_examples']} padded examples, "
          f"{totals['rejected']} rejected"
      )
      emit(
          f'Owner merged {self._merged_items} frontend batches into '
          f'{self._batcher.batches_run} device batches '
          f'({self._batcher.examples_run} examples, '
          f'{self._batcher.padded_examples} padded examples)'
      )
    emit(f'Served {total_req} requests / {total_ex} examples')
    self._cleanup()
    return not self._failed

  def _cleanup(self) -> None:
    """Closes sockets and removes IPC ring files."""
    if self._listener is not None:
      self._listener.close()
    for fe in self._frontends:
      if fe.sock is not None:
        fe.sock.close()
    if self._own_ipc_dir:
      shutil.rmtree(self._ipc_dir, ignore_errors=True)
    else:
      for name in [SOCKET_NAME] + [
          RING_NAME.format(frontend=i) for i in range(self._n)
      ]:
        with contextlib.suppress(OSError):
          os.unlink(os.path.join(self._ipc_dir, name))


def run_owner(f: python_server.Flags) -> None:
  """Entry point of `python_server --num_processes=N` (N > 1)."""
  owner = _Owner(f)
  ok = False
  try:
    port = owner.start()
    logging.info(
        'Python multi-process baseline listening on port %d (device=%s, '
        '%d frontends x %d threads, dynamic_batching=%s, '
        'max_enqueued_batches=%d, pid=%d)',
        port,
        f.device.value,
        f.num_processes.value,
        f.num_threads.value,
        f.dynamic_batching.value,
        f.max_enqueued_batches.value,
        os.getpid(),
    )
    # Printed unbuffered, and only once every frontend is bound, so that the
    # first client connections are spread over all of them.
    print(f'READY port={port} pid={os.getpid()}', flush=True)
    owner.wait()
  finally:
    ok = owner.shutdown()
  if not ok:
    sys.exit(1)

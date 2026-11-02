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

"""Tests for the multi-process Python server (ring allocator + e2e, CPU)."""

from concurrent import futures
import json
import os
import shutil
import signal
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
from typing import Any

from absl import flags
from absl.testing import absltest
from absl.testing import parameterized
import grpc
import numpy as np

from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as export
from examples.dlrm_benchmark import multiprocess_server
from examples.dlrm_benchmark import python_server
from xla_saved_model.protos.grpc import predict_pb2

FLAGS = flags.FLAGS

_BATCH_SIZES = (1, 4, 16)


class RowAllocatorTest(absltest.TestCase):

  def test_wraps_around_and_frees_fifo(self):
    a = multiprocess_server.RowAllocator(10)
    self.assertEqual(a.reserve(4), 0)
    self.assertEqual(a.reserve(4), 4)
    # 2 rows left at the end: a 3-row request must wrap, which needs the
    # head segment freed first.
    with self.assertRaises(TimeoutError):
      a.reserve(3, timeout_s=0.05)
    a.release(0)
    self.assertEqual(a.reserve(3), 0)  # wrapped
    self.assertEqual(a.live_segments, 2)
    a.release(4)
    a.release(0)
    self.assertEqual(a.live_segments, 0)
    self.assertEqual(a.reserve(10), 0)  # empty ring: full capacity again

  def test_out_of_order_release_reclaims_prefix_only(self):
    a = multiprocess_server.RowAllocator(8)
    offs = [a.reserve(2) for _ in range(4)]  # 0, 2, 4, 6: full
    self.assertEqual(offs, [0, 2, 4, 6])
    a.release(2)  # not the oldest: nothing reclaimable yet
    with self.assertRaises(TimeoutError):
      a.reserve(1, timeout_s=0.05)
    a.release(0)  # now [0, 4) is free
    self.assertEqual(a.reserve(3), 0)

  def test_reserve_blocks_until_release(self):
    a = multiprocess_server.RowAllocator(4)
    a.reserve(4)
    got = []

    def waiter():
      got.append(a.reserve(2, timeout_s=5))

    t = threading.Thread(target=waiter)
    t.start()
    time.sleep(0.1)
    self.assertEqual(got, [])
    a.release(0)
    t.join(timeout=5)
    self.assertEqual(got, [0])

  def test_rejects_impossible_sizes(self):
    a = multiprocess_server.RowAllocator(4)
    with self.assertRaises(ValueError):
      a.reserve(5)
    with self.assertRaises(ValueError):
      a.reserve(0)


class RingTest(absltest.TestCase):

  def test_two_mappings_share_pages(self):
    path = os.path.join(self.create_tempdir().full_path, 'ring')
    config = common.make_model_config(embedding_size=4, max_vocab_size=3)
    a = multiprocess_server.Ring.for_config(path, config, rows=5, create=True)
    b = multiprocess_server.Ring.for_config(path, config, rows=5, create=False)
    a.emb[3, 1, 2] = 42.0
    a.dense[4, 0] = 1.5
    b.out[2] = -1.0
    self.assertEqual(b.emb[3, 1, 2], 42.0)
    self.assertEqual(b.dense[4, 0], 1.5)
    self.assertEqual(a.out[2], -1.0)
    rows = multiprocess_server.Ring.rows_for(config, ring_mb=1)
    self.assertGreaterEqual(rows, 2 * config.batch_sizes[-1])


class MultiprocessServerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.tmpdir = tempfile.mkdtemp(prefix='dlrm_mp_server_test_')
    cls.addClassCleanup(shutil.rmtree, cls.tmpdir, True)
    cls.weights_dir = os.path.join(cls.tmpdir, 'weights')
    cls.pysrc_dir = os.path.join(cls.tmpdir, 'pysrc')
    os.makedirs(cls.pysrc_dir, exist_ok=True)
    examples_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    os.symlink(examples_dir, os.path.join(cls.pysrc_dir, 'examples'))
    cls.config = export.export_model(
        embedding_size=8,
        max_vocab_size=20,
        batch_sizes=_BATCH_SIZES,
        baseline_weights_dir=cls.weights_dir,
    )
    python_server.restrict_jax_platforms('cpu')
    cls.reference = python_server.DlrmBaseline(cls.weights_dir, device='cpu')

  def _server_command(self) -> tuple[list[str], str | None]:
    pkg = multiprocess_server.__package__
    test_srcdir = (
        FLAGS.test_srcdir
        if FLAGS.is_parsed()
        else os.environ.get('TEST_SRCDIR', '')
    )
    if test_srcdir and pkg:
      binary = os.path.join(test_srcdir, *pkg.split('.'), 'python_server')
      if os.path.isfile(binary) and os.access(binary, os.X_OK):
        return [binary], None
    # OSS pytest: run from an isolated import root holding only `examples` so
    # the checkout's `xla_saved_model/` tree cannot shadow the installed wheel.
    return [sys.executable, '-m', f'{pkg}.python_server'], self.pysrc_dir

  def _start(self, dynamic_batching: bool, *extra_flags: str):
    """Starts `python_server --num_processes=2`; returns (proc, port, lines)."""
    cmd, cwd = self._server_command()
    # No --ipc_dir: the server picks /dev/shm or /tmp itself. The sandbox
    # tmpdir can be longer than the 108-byte AF_UNIX socket path limit.
    argv = [
        *cmd,
        f'--weights_dir={self.weights_dir}',
        '--port=0',
        '--device=cpu',
        '--num_processes=2',
        '--num_threads=2',
        '--ring_mb=1',
        '--dynamic_batching' if dynamic_batching else '--nodynamic_batching',
        '--batch_timeout_micros=20000',
        *extra_flags,
    ]
    proc = subprocess.Popen(
        argv,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    assert proc.stdout is not None
    lines: list[str] = []
    port = None
    deadline = time.monotonic() + 240
    while time.monotonic() < deadline:
      line = proc.stdout.readline()
      if not line:
        if proc.poll() is not None:
          break
        continue
      lines.append(line)
      if line.startswith('READY port='):
        port = int(line.split()[1].split('=')[1])
        break
    if port is None:
      proc.kill()
      self.fail('server did not become ready:\n' + ''.join(lines[-40:]))

    def drain():
      for line in proc.stdout:
        lines.append(line)

    threading.Thread(target=drain, daemon=True).start()

    def cleanup():
      if proc.poll() is None:
        proc.kill()
        proc.wait()

    self.addCleanup(cleanup)
    return proc, port, lines

  def _stub(self, port: int, index: int = 0):
    channel = common.make_channel(
        f'localhost:{port}',
        # Each channel = its own TCP connection, so that the kernel's
        # SO_REUSEPORT hashing can spread them over both frontends.
        options=[('grpc.use_local_subchannel_pool', 1), ('test.index', index)],
    )
    self.addCleanup(channel.close)
    return channel.unary_unary(
        common.PREDICT_METHOD,
        request_serializer=predict_pb2.PredictRequest.SerializeToString,
        response_deserializer=predict_pb2.PredictResponse.FromString,
    )

  def _stop_and_collect(self, proc, lines) -> str:
    proc.send_signal(signal.SIGTERM)
    proc.wait(timeout=60)
    time.sleep(0.2)  # let the drain thread catch the last lines
    return ''.join(lines)

  @parameterized.named_parameters(('no_batching', False), ('batching', True))
  def test_parity_concurrency_and_shutdown(self, dynamic_batching):
    proc, port, lines = self._start(dynamic_batching)

    # Parity vs the in-process reference, including a non-bucket size.
    rng = np.random.default_rng(1)
    stub = self._stub(port)
    for batch in (1, 3, 4, 16):
      feats = common.generate_features(self.config, batch, rng)
      resp = stub(common.make_predict_request(feats), timeout=60)
      self.assertEqual(resp.model_spec.name, common.DEFAULT_MODEL_NAME)
      np.testing.assert_allclose(
          common.predictions_from_response(resp),
          self.reference.predict(feats),
          atol=1e-5,
      )

    # Invalid requests are rejected by the frontend, before any IPC.
    bad = common.generate_features(self.config, 2, rng)
    del bad['0']
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(bad), timeout=60)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )

    # Many connections, concurrent requests: every answer must be correct
    # and both frontends must have seen traffic.
    stubs = [self._stub(port, i) for i in range(16)]
    feats = [common.generate_features(self.config, 4, rng) for _ in range(64)]
    with futures.ThreadPoolExecutor(max_workers=32) as pool:
      resps = list(
          pool.map(
              lambda i: stubs[i % 16](
                  common.make_predict_request(feats[i]), timeout=60
              ),
              range(64),
          )
      )
    for f, r in zip(feats, resps):
      np.testing.assert_allclose(
          common.predictions_from_response(r),
          self.reference.predict(f),
          atol=1e-5,
      )

    log = self._stop_and_collect(proc, lines)
    self.assertEqual(proc.returncode, 0, log)
    served = {}
    for line in log.splitlines():
      if line.startswith('Frontend ') and ' served ' in line:
        parts = line.split()
        served[int(parts[1])] = int(parts[3])
    self.assertEqual(sorted(served), [0, 1], log)
    self.assertEqual(sum(served.values()), 4 + 64, log)
    self.assertTrue(all(n > 0 for n in served.values()), log)
    self.assertIn('Served 68 requests / 280 examples', log)
    if dynamic_batching:
      self.assertIn('Batcher stats:', log)

  def test_batching_coalesces_requests_in_frontends(self):
    proc, port, lines = self._start(dynamic_batching=True)
    rng = np.random.default_rng(2)
    stubs = [self._stub(port, i) for i in range(8)]
    feats = [common.generate_features(self.config, 1, rng) for _ in range(32)]
    with futures.ThreadPoolExecutor(max_workers=32) as pool:
      list(
          pool.map(
              lambda i: stubs[i % 8](
                  common.make_predict_request(feats[i]), timeout=60
              ),
              range(32),
          )
      )
    log = self._stop_and_collect(proc, lines)
    self.assertEqual(proc.returncode, 0, log)
    stats = [l for l in log.splitlines() if l.startswith('Batcher stats:')]
    self.assertLen(stats, 1, log)
    batches = int(stats[0].split()[2])
    # 32 single-example requests with a 20 ms window and max batch 16 must
    # have been coalesced into far fewer gathers than requests -- in the
    # frontends, before the lookup, like the single-process batched arm.
    self.assertLess(batches, 16, log)
    self.assertIn(', 32 examples, ', stats[0])
    self.assertIn(', 0 rejected', stats[0])
    per_frontend = [
        l for l in log.splitlines() if l.startswith('Frontend ') and
        ' batcher: ' in l
    ]
    self.assertLen(per_frontend, 2, log)
    self.assertEqual(
        sum(int(l.split(' batcher: ')[1].split()[0]) for l in per_frontend),
        batches,
        log,
    )
    # The owner never waits, it only merges what is already queued: it runs
    # at most as many device batches as it received frontend batches.
    merged = [l for l in log.splitlines() if l.startswith('Owner merged ')]
    self.assertLen(merged, 1, log)
    words = merged[0].split()
    self.assertEqual(int(words[2]), batches, log)  # frontend batches
    self.assertLessEqual(int(words[6]), batches, log)  # device batches

  def test_queue_full_is_unavailable_and_counted(self):
    # One maximum-size batch of queue budget in total (split 1 + 1 over the
    # two frontends, 16 examples each); 32 concurrent full-size requests per
    # frontend cannot all be admitted while one batcher thread drains them.
    proc, port, lines = self._start(
        True, '--max_enqueued_batches=1', '--batch_timeout_micros=200000'
    )
    rng = np.random.default_rng(3)
    stubs = [self._stub(port, i) for i in range(8)]
    feats = [common.generate_features(self.config, 16, rng) for _ in range(64)]
    reqs = [common.make_predict_request(f) for f in feats]
    # One burst, as fast as the client can issue it.
    futs = [stubs[i % 8].future(reqs[i], timeout=60) for i in range(64)]
    results = []
    for fut in futs:
      try:
        results.append(fut.result())
      except grpc.RpcError as e:
        results.append(e)
    ok = [r for r in results if not isinstance(r, grpc.RpcError)]
    failed = [r for r in results if isinstance(r, grpc.RpcError)]
    self.assertNotEmpty(failed)
    self.assertNotEmpty(ok)
    for e in failed:
      self.assertEqual(getattr(e, 'code')(), grpc.StatusCode.UNAVAILABLE)
      self.assertIn('batch scheduling queue', getattr(e, 'details')())
    for i, r in enumerate(results):
      if not isinstance(r, grpc.RpcError):
        np.testing.assert_allclose(
            common.predictions_from_response(r),
            self.reference.predict(feats[i]),
            atol=1e-5,
        )
    log = self._stop_and_collect(proc, lines)
    self.assertEqual(proc.returncode, 0, log)
    stats = [l for l in log.splitlines() if l.startswith('Batcher stats:')]
    self.assertLen(stats, 1, log)
    self.assertIn(f', {len(failed)} rejected', stats[0])
    self.assertIn(f'Served {len(ok)} requests / {16 * len(ok)} examples', log)

  def test_recovers_from_stale_socket_in_user_ipc_dir(self):
    ipc_dir = tempfile.mkdtemp(prefix='dlrm_ipc_test_', dir='/tmp')
    self.addCleanup(shutil.rmtree, ipc_dir, True)
    with open(os.path.join(ipc_dir, multiprocess_server.SOCKET_NAME), 'w'):
      pass
    proc, port, lines = self._start(False, f'--ipc_dir={ipc_dir}')
    feats = common.generate_features(
        self.config, 1, np.random.default_rng(9)
    )
    resp = self._stub(port)(common.make_predict_request(feats), timeout=60)
    np.testing.assert_allclose(
        common.predictions_from_response(resp),
        self.reference.predict(feats),
        atol=1e-5,
    )
    log = self._stop_and_collect(proc, lines)
    self.assertEqual(proc.returncode, 0, log)
    self.assertTrue(os.path.isdir(ipc_dir))

  def test_remote_model_in_process_roundtrip_error_and_timeout(self):
    ring_path = os.path.join(self.create_tempdir().full_path, 'ring.0')
    ring = multiprocess_server.Ring.for_config(
        ring_path, self.config, rows=32, create=True
    )
    ring.emb[:] = 99.0
    ring.dense[:] = 99.0
    fe_sock, owner_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET
    )
    self.addCleanup(fe_sock.close)
    self.addCleanup(owner_sock.close)

    remote = multiprocess_server._RemoteModel(
        self.config, self.reference.tables, ring, fe_sock, timeout_s=0.15
    )
    # Completing an unknown request ID is a safe no-op.
    remote.complete(999999, 0)

    def dispatch_replies():
      while True:
        msg = multiprocess_server._recv(fe_sock)
        if not msg:
          return
        _, req_id, status = struct.unpack(multiprocess_server._DONE_FMT, msg)
        remote.complete(req_id, status)

    dispatch_thread = threading.Thread(target=dispatch_replies, daemon=True)
    dispatch_thread.start()

    seen: list[tuple[int, int, int, int, bool]] = []

    def fake_owner():
      # 1st request: normal execution (batch=3 padded to bucket 4).
      msg0 = multiprocess_server._recv(owner_sock)
      _, req_id0, off0, rows0, batch0 = struct.unpack(
          multiprocess_server._REQ_FMT, msg0
      )
      pad_zeroed = bool(
          np.all(ring.emb[off0 + batch0 : off0 + rows0] == 0.0)
          and np.all(ring.dense[off0 + batch0 : off0 + rows0] == 0.0)
      )
      seen.append((req_id0, off0, rows0, batch0, pad_zeroed))
      preds0 = self.reference.run_model(
          ring.emb[off0 : off0 + rows0],
          ring.dense[off0 : off0 + rows0],
          batch0,
      )
      ring.out[off0 : off0 + batch0] = preds0
      owner_sock.sendall(
          struct.pack(
              multiprocess_server._DONE_FMT,
              multiprocess_server._DONE,
              req_id0,
              0,
          )
      )

      # 2nd request: simulate device owner failure (status=1).
      msg1 = multiprocess_server._recv(owner_sock)
      _, req_id1, off1, rows1, batch1 = struct.unpack(
          multiprocess_server._REQ_FMT, msg1
      )
      seen.append((req_id1, off1, rows1, batch1, True))
      owner_sock.sendall(
          struct.pack(
              multiprocess_server._DONE_FMT,
              multiprocess_server._DONE,
              req_id1,
              1,
          )
      )
      # 3rd request intentionally receives no reply so timeout_s triggers.
      multiprocess_server._recv(owner_sock)

    owner_thread = threading.Thread(target=fake_owner, daemon=True)
    owner_thread.start()

    rng = np.random.default_rng(17)
    feats3 = common.generate_features(self.config, 3, rng)
    preds = remote.predict(feats3)
    np.testing.assert_allclose(
        preds, self.reference.predict(feats3), atol=1e-6
    )

    feats1 = common.generate_features(self.config, 1, rng)
    with self.assertRaisesRegex(RuntimeError, 'failed to run'):
      remote.predict(feats1)

    with self.assertRaisesRegex(RuntimeError, 'did not answer'):
      remote.predict(feats1)
    self.assertEqual(remote._alloc.live_segments, 0)

    owner_thread.join(timeout=5)
    self.assertEqual(seen, [(0, 0, 4, 3, True), (1, 0, 1, 1, True)])

  def test_run_frontend_in_process_with_dynamic_batching(self):
    ipc_dir = tempfile.mkdtemp(prefix='dlrm_fe_', dir='/tmp')
    self.addCleanup(shutil.rmtree, ipc_dir, True)

    missing_fv = flags.FlagValues()
    missing_f = python_server.define_flags(missing_fv)
    with self.assertRaises(flags.IllegalFlagValueError):
      missing_fv(['python_server'])
    missing_fv(['python_server', f'--weights_dir={self.weights_dir}'])
    self.assertFalse(missing_f.dynamic_batching.value)
    self.assertTrue(missing_f.warmup.value)
    with self.assertRaisesRegex(ValueError, '--frontend_id requires --ipc_dir'):
      multiprocess_server.run_frontend(missing_f)

    rows = multiprocess_server.Ring.rows_for(self.config, ring_mb=1)
    ring_path = os.path.join(
        ipc_dir, multiprocess_server.RING_NAME.format(frontend=0)
    )
    owner_ring = multiprocess_server.Ring.for_config(
        ring_path, self.config, rows=rows, create=True
    )
    # Sentinel on the last row verifies `run_frontend` opens with create=False
    # (create=True would O_TRUNC the shared file and zero it out).
    owner_ring.out[-1] = 123.0

    listener = socket.socket(socket.AF_UNIX, socket.SOCK_SEQPACKET)
    self.addCleanup(listener.close)
    listener.bind(os.path.join(ipc_dir, multiprocess_server.SOCKET_NAME))
    listener.listen(1)

    orig_int = signal.getsignal(signal.SIGINT)
    orig_term = signal.getsignal(signal.SIGTERM)
    self.addCleanup(signal.signal, signal.SIGINT, orig_int)
    self.addCleanup(signal.signal, signal.SIGTERM, orig_term)

    fv = flags.FlagValues()
    f = python_server.define_flags(fv)
    fv([
        'python_server',
        f'--weights_dir={self.weights_dir}',
        f'--ipc_dir={ipc_dir}',
        '--frontend_id=0',
        '--num_processes=2',
        '--num_threads=1',
        '--dynamic_batching',
        '--batch_timeout_micros=10000',
    ])

    rng = np.random.default_rng(21)
    feats8_a = common.generate_features(self.config, 8, rng)
    feats8_b = common.generate_features(self.config, 8, rng)
    feats3 = common.generate_features(self.config, 3, rng)
    state: dict[str, Any] = {'error': None}

    def owner_driver():
      conn: socket.socket | None = None
      ch: grpc.Channel | None = None
      try:
        conn, _ = listener.accept()
        state['hello'] = multiprocess_server._recv(conn)
        state['loaded'] = multiprocess_server._recv(conn)
        conn.sendall(
            struct.pack(
                multiprocess_server._ID_FMT, multiprocess_server._BIND, 0
            )
        )
        started = multiprocess_server._recv(conn)
        _, bound_port = struct.unpack(multiprocess_server._ID_FMT, started)
        state['bound_port'] = bound_port

        ch = common.make_channel(f'localhost:{bound_port}')
        stub = ch.unary_unary(
            common.PREDICT_METHOD,
            request_serializer=predict_pb2.PredictRequest.SerializeToString,
            response_deserializer=predict_pb2.PredictResponse.FromString,
        )

        # Two concurrent 8-row RPCs coalesce immediately into one 16-row batch
        # (also proving handler_threads was widened past --num_threads=1).
        fut_a = stub.future(common.make_predict_request(feats8_a), timeout=15)
        fut_b = stub.future(common.make_predict_request(feats8_b), timeout=15)
        req_msg = multiprocess_server._recv(conn)
        _, req_id, off, req_rows, batch = struct.unpack(
            multiprocess_server._REQ_FMT, req_msg
        )
        owner_ring.out[off : off + batch] = self.reference.run_model(
            owner_ring.emb[off : off + req_rows],
            owner_ring.dense[off : off + req_rows],
            batch,
        )
        conn.sendall(
            struct.pack(
                multiprocess_server._DONE_FMT,
                multiprocess_server._DONE,
                req_id,
                0,
            )
        )
        state['resp_a'] = common.predictions_from_response(
            fut_a.result(timeout=15)
        )
        state['resp_b'] = common.predictions_from_response(
            fut_b.result(timeout=15)
        )

        # Single 3-row RPC exercises batch timeout + bucket padding (3 -> 4).
        fut_c = stub.future(common.make_predict_request(feats3), timeout=15)
        req_msg2 = multiprocess_server._recv(conn)
        _, req_id2, off2, req_rows2, batch2 = struct.unpack(
            multiprocess_server._REQ_FMT, req_msg2
        )
        owner_ring.out[off2 : off2 + batch2] = self.reference.run_model(
            owner_ring.emb[off2 : off2 + req_rows2],
            owner_ring.dense[off2 : off2 + req_rows2],
            batch2,
        )
        conn.sendall(
            struct.pack(
                multiprocess_server._DONE_FMT,
                multiprocess_server._DONE,
                req_id2,
                0,
            )
        )
        state['resp_c'] = common.predictions_from_response(
            fut_c.result(timeout=15)
        )
        ch.close()
        ch = None

        conn.sendall(struct.pack('<B', multiprocess_server._STOP))
        conn.settimeout(2.0)
        stats_msg = multiprocess_server._recv(conn)
        state['stats_kind'] = stats_msg[0]
        state['stats'] = json.loads(stats_msg[1:].decode())
        state['eof'] = multiprocess_server._recv(conn)
      except BaseException as exc:  # pylint: disable=broad-exception-caught
        state['error'] = exc
      finally:
        if ch is not None:
          ch.close()
        if conn is not None:
          try:
            conn.sendall(struct.pack('<B', multiprocess_server._STOP))
          except OSError:
            pass
          conn.close()

    driver = threading.Thread(target=owner_driver, daemon=True)
    driver.start()
    multiprocess_server.run_frontend(f)
    driver.join(timeout=10)

    self.assertIsNone(state['error'])
    self.assertEqual(
        state['hello'],
        struct.pack(multiprocess_server._ID_FMT, multiprocess_server._HELLO, 0),
    )
    self.assertEqual(
        state['loaded'], struct.pack('<B', multiprocess_server._LOADED)
    )
    self.assertGreater(state['bound_port'], 0)
    self.assertEqual(owner_ring.out[-1], 123.0)
    np.testing.assert_allclose(
        state['resp_a'], self.reference.predict(feats8_a), atol=1e-6
    )
    np.testing.assert_allclose(
        state['resp_b'], self.reference.predict(feats8_b), atol=1e-6
    )
    np.testing.assert_allclose(
        state['resp_c'], self.reference.predict(feats3), atol=1e-6
    )
    self.assertEqual(state['stats_kind'], multiprocess_server._STATS)
    self.assertEqual(state['eof'], b'')
    self.assertEqual(
        state['stats'],
        {
            'requests': 3,
            'examples': 19,
            'batches': 2,
            'examples_run': 19,
            'padded_examples': 1,
            'rejected': 0,
        },
    )

  def test_owner_merges_batches_and_reads_frontend_records_in_process(self):
    fv = flags.FlagValues()
    f = python_server.define_flags(fv)
    fv([
        'python_server',
        f'--weights_dir={self.weights_dir}',
        '--num_processes=2',
        '--num_threads=2',
        '--dynamic_batching',
    ])
    owner = multiprocess_server._Owner(f)
    owner._model = self.reference
    owner._batcher = python_server.DynamicBatcher(
        owner._run_batch,
        max_batch_size=16,
        timeout_s=0.0,
        buckets=self.config.batch_sizes,
        num_threads=1,
    )
    self.addCleanup(owner._batcher.stop)

    tmp = self.create_tempdir().full_path
    ring0 = multiprocess_server.Ring.for_config(
        os.path.join(tmp, 'r0'), self.config, rows=16, create=True
    )
    ring1 = multiprocess_server.Ring.for_config(
        os.path.join(tmp, 'r1'), self.config, rows=16, create=True
    )
    fe0_sock, own0_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET
    )
    fe1_sock, own1_sock = socket.socketpair(
        socket.AF_UNIX, socket.SOCK_SEQPACKET
    )
    for s in (fe0_sock, own0_sock, fe1_sock, own1_sock):
      self.addCleanup(s.close)

    dummy_proc: Any = None
    fe0 = multiprocess_server._Frontend(0, dummy_proc, ring0, sock=own0_sock)
    fe1 = multiprocess_server._Frontend(1, dummy_proc, ring1, sock=own1_sock)

    rng = np.random.default_rng(31)
    feats0 = common.generate_features(self.config, 3, rng)
    feats1 = common.generate_features(self.config, 2, rng)
    _, dense0 = self.reference.lookup(feats0, out=ring0.emb[:3])
    ring0.dense[:3] = dense0
    _, dense1 = self.reference.lookup(feats1, out=ring1.emb[:2])
    ring1.dense[:2] = dense1

    # Merge two frontend batches (3 + 2 = 5 -> bucket 16) in one model call.
    res = owner._run_batch(
        [
            multiprocess_server._Work(fe0, req_id=10, off=0, rows=4),
            multiprocess_server._Work(fe1, req_id=20, off=0, rows=4),
        ],
        [3, 2],
    )
    self.assertEqual(res, [None, None])
    self.assertEqual(
        multiprocess_server._recv(fe0_sock),
        struct.pack(
            multiprocess_server._DONE_FMT, multiprocess_server._DONE, 10, 0
        ),
    )
    self.assertEqual(
        multiprocess_server._recv(fe1_sock),
        struct.pack(
            multiprocess_server._DONE_FMT, multiprocess_server._DONE, 20, 0
        ),
    )
    np.testing.assert_allclose(
        ring0.out[:3], self.reference.predict(feats0), atol=1e-6
    )
    np.testing.assert_allclose(
        ring1.out[:2], self.reference.predict(feats1), atol=1e-6
    )

    # Also drive `_reader` + single-item `_run_one` + `_STATS` + clean EOF.
    reader_thread = threading.Thread(
        target=owner._reader, args=(fe0,), daemon=True
    )
    reader_thread.start()
    ring0.out[:3] = 0.0
    fe0_sock.sendall(
        struct.pack(
            multiprocess_server._REQ_FMT, multiprocess_server._REQ, 11, 0, 4, 3
        )
    )
    self.assertEqual(
        multiprocess_server._recv(fe0_sock),
        struct.pack(
            multiprocess_server._DONE_FMT, multiprocess_server._DONE, 11, 0
        ),
    )
    np.testing.assert_allclose(
        ring0.out[:3], self.reference.predict(feats0), atol=1e-6
    )
    # 16-byte JSON makes total datagram length 17 == struct.calcsize(_REQ_FMT),
    # verifying `_reader` dispatches on `kind` rather than datagram size alone.
    raw_stats = b'{"requests":  1}'
    self.assertEqual(
        1 + len(raw_stats), struct.calcsize(multiprocess_server._REQ_FMT)
    )
    fe0_sock.sendall(struct.pack('<B', multiprocess_server._STATS) + raw_stats)
    # Wait briefly for STATS to be processed before closing the socket cleanly.
    deadline = time.monotonic() + 5
    while fe0.stats is None and time.monotonic() < deadline:
      time.sleep(0.01)
    owner._stop.set()
    fe0_sock.close()
    reader_thread.join(timeout=5)
    self.assertEqual(fe0.stats, {'requests': 1})

    # `_cleanup()` with default `_own_ipc_dir == False` unlinks IPC files while
    # preserving a user-supplied directory.
    user_ipc = os.path.join(tmp, 'user_ipc')
    os.makedirs(user_ipc)
    stale_sock = os.path.join(user_ipc, multiprocess_server.SOCKET_NAME)
    with open(stale_sock, 'w'):
      pass
    owner._ipc_dir = user_ipc
    owner._cleanup()
    self.assertTrue(os.path.isdir(user_ipc))
    self.assertFalse(os.path.exists(stale_sock))


if __name__ == '__main__':
  absltest.main()

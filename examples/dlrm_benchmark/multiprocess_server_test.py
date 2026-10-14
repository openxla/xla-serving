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
import os
import signal
import subprocess
import threading
import time

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
_SERVER_BIN = (
    '/third_party/neptune/examples/dlrm_benchmark/python_server'
)


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
    cls.weights_dir = os.path.join(
        absltest.get_default_test_tmpdir(), 'weights'
    )
    cls.config = export.export_model(
        embedding_size=8,
        max_vocab_size=20,
        batch_sizes=_BATCH_SIZES,
        baseline_weights_dir=cls.weights_dir,
    )
    python_server.restrict_jax_platforms('cpu')
    cls.reference = python_server.DlrmBaseline(cls.weights_dir, device='cpu')

  def _start(self, dynamic_batching: bool, *extra_flags: str):
    """Starts `python_server --num_processes=2`; returns (proc, port)."""
    binary = os.path.join(FLAGS.test_srcdir, _SERVER_BIN)
    # No --ipc_dir: the server picks /dev/shm or /tmp itself. The sandbox
    # tmpdir can be longer than the 108-byte AF_UNIX socket path limit.
    argv = [
        binary,
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


if __name__ == '__main__':
  absltest.main()

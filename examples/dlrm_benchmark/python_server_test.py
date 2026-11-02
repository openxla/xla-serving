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

"""End-to-end tests for the pure-Python baseline server (CPU only)."""

from concurrent import futures
import math
import os
import shutil
import sys
import tempfile
import threading
import time
import types
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import jax
import numpy as np

from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as export
from examples.dlrm_benchmark import python_server
from xla_saved_model.protos.grpc import predict_pb2

_BATCH_SIZES = (1, 4, 16)


class PythonServerTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    tmpdir = tempfile.mkdtemp(prefix='dlrm_py_server_test_')
    cls.addClassCleanup(shutil.rmtree, tmpdir, True)
    cls.weights_dir = os.path.join(tmpdir, 'weights')
    cls.config = export.export_model(
        embedding_size=8,
        max_vocab_size=20,
        batch_sizes=_BATCH_SIZES,
        baseline_weights_dir=cls.weights_dir,
    )
    cls.model = python_server.DlrmBaseline(cls.weights_dir, device='cpu')

  def _start(self, dynamic_batching: bool, **kwargs):
    server, servicer, batcher = python_server.build_server(
        weights_dir=self.weights_dir,
        port=0,
        device='cpu',
        num_threads=4,
        dynamic_batching=dynamic_batching,
        warmup=False,
        **kwargs,
    )
    server.start()
    self.addCleanup(server.stop, 0)
    if batcher is not None:
      self.addCleanup(batcher.stop)
    channel = common.make_channel(f'localhost:{servicer.port}')
    self.addCleanup(channel.close)
    stub = channel.unary_unary(
        common.PREDICT_METHOD,
        request_serializer=predict_pb2.PredictRequest.SerializeToString,
        response_deserializer=predict_pb2.PredictResponse.FromString,
    )
    return stub, servicer, batcher

  def test_predict_method_follows_proto_package(self):
    # Derive via the message descriptor (an independent path from
    # `predict_pb2.DESCRIPTOR`) and verify the service path matches.
    package = predict_pb2.PredictRequest.DESCRIPTOR.file.package
    self.assertNotEmpty(package)
    self.assertEqual(common.PREDICT_SERVICE, f'{package}.PredictionService')
    self.assertEqual(
        common.PREDICT_METHOD, f'/{package}.PredictionService/Predict'
    )

  def test_predict_pads_to_bucket_and_matches_reference(self):
    rng = np.random.default_rng(0)
    for batch in (1, 3, 4, 9, 16):
      feats = common.generate_features(self.config, batch, rng)
      preds = self.model.predict(feats)
      self.assertEqual(preds.shape, (batch,))
      ref = self.model.predict(common.pad_features(feats, 16))[:batch]
      np.testing.assert_allclose(preds, ref, atol=1e-6)

  def test_stacked_lookup_equals_dict_reference(self):
    feats = common.generate_features(self.config, 5, np.random.default_rng(7))
    ref = common.embedding_lookup_numpy(self.model.tables, feats)
    stacked = common.embedding_lookup_stacked(self.model.tables, feats)
    self.assertEqual(stacked.shape, (5, self.config.num_tables, 8))
    self.assertEqual(stacked.dtype, np.float32)
    for i in range(self.config.num_tables):
      np.testing.assert_array_equal(stacked[:, i, :], ref[str(i)])
    # Same numbers when gathering into a strided view of caller memory
    # (what the multi-process frontends do with their shared-memory ring).
    buf = np.full((8, self.config.num_tables, 8), np.nan, np.float32)
    out = common.embedding_lookup_stacked(
        self.model.tables, feats, out=buf[2:7]
    )
    self.assertIs(out.base, buf)
    np.testing.assert_array_equal(buf[2:7], stacked)
    self.assertTrue(np.isnan(buf[:2]).all() and np.isnan(buf[7:]).all())
    with self.assertRaisesRegex(ValueError, 'out must be'):
      common.embedding_lookup_stacked(self.model.tables, feats, out=buf[:3])

  def test_run_model_pads_and_ignores_padding_rows(self):
    feats = common.generate_features(self.config, 3, np.random.default_rng(8))
    emb, dense = self.model.lookup(feats)
    ref = self.model.run_model(emb, dense, 3)  # pads 3 -> 4 internally
    self.assertEqual(ref.shape, (3,))
    # Caller-padded to the bucket with garbage in the padding rows: the
    # per-example model must give identical real rows.
    emb4 = np.concatenate([emb, np.full((1,) + emb.shape[1:], 7.0, np.float32)])
    dense4 = np.concatenate(
        [dense, np.full((1, dense.shape[1]), 7.0, np.float32)]
    )
    np.testing.assert_allclose(self.model.run_model(emb4, dense4, 3), ref)
    np.testing.assert_allclose(self.model.predict(feats), ref)
    self.assertEqual(
        common.pad_rows(emb, 3).shape, (3, self.config.num_tables, 8)
    )
    with self.assertRaises(ValueError):
      common.pad_rows(emb4, 3)

  def test_batcher_generic_payloads(self):
    seen = []

    def run_fn(payloads, batches):
      seen.append((list(payloads), list(batches)))
      return [p * 10 for p in payloads]

    batcher = python_server.DynamicBatcher(
        run_fn, max_batch_size=8, timeout_s=0.2, buckets=(1, 4, 8)
    )
    self.addCleanup(batcher.stop)
    futs = [batcher.submit(1, 3), batcher.submit(2, 3)]
    self.assertEqual([f.result(timeout=5) for f in futs], [10, 20])
    self.assertEqual(seen, [([1, 2], [3, 3])])
    self.assertEqual(batcher.examples_run, 6)
    self.assertEqual(batcher.padded_examples, 2)  # 6 -> bucket 8
    self.assertEqual(batcher.rejected, 0)
    self.assertIn('0 rejected', batcher.stats_line())
    self.assertEqual(
        python_server.split_predictions(np.arange(6), [2, 4])[1].tolist(),
        [2, 3, 4, 5],
    )

  def _blocked_batcher(self, result_fn=None, **kwargs):
    """A batcher whose single worker is parked inside `run_fn` on `gate`."""
    gate = threading.Event()
    entered = threading.Event()
    result_fn = result_fn or (lambda payloads, batches: list(payloads))

    def run_fn(payloads, batches):
      entered.set()
      gate.wait(10)
      return result_fn(payloads, batches)

    batcher = python_server.DynamicBatcher(
        run_fn, timeout_s=0.0, buckets=(1, 4, 8), num_threads=1, **kwargs
    )
    self.addCleanup(batcher.stop)
    self.addCleanup(gate.set)
    # Occupy the only worker so that everything submitted next stays queued.
    first = batcher.submit('busy', 1)
    self.assertTrue(entered.wait(5))
    return batcher, gate, first

  def test_batcher_rejects_when_queue_is_full(self):
    # max_enqueued_batches=2 x max_batch_size=4 = 8 queued examples at most.
    batcher, gate, first = self._blocked_batcher(
        max_batch_size=4, max_enqueued_batches=2
    )
    queued = [batcher.submit(i, 3) for i in range(2)]  # 6 examples queued
    self.assertEqual(batcher.queued_examples, 6)
    with self.assertRaisesRegex(
        python_server.QueueFullError, 'queue .* is full; currently 6 examples'
    ) as cm:
      batcher.submit('too-many', 3)  # 9 > 8
    self.assertIn('max_enqueued_batches is 2', str(cm.exception))
    queued.append(batcher.submit('fits', 2))  # exactly 8: admitted
    self.assertEqual(batcher.rejected, 1)
    gate.set()
    self.assertEqual(first.result(timeout=5), 'busy')
    self.assertEqual(
        [f.result(timeout=5) for f in queued], [0, 1, 'fits']
    )
    self.assertEqual(batcher.queued_examples, 0)
    self.assertIn(', 1 rejected', batcher.stats_line())

  def test_batcher_unbounded_by_default(self):
    batcher, gate, _ = self._blocked_batcher(max_batch_size=4)
    self.assertEqual(batcher.max_enqueued_batches, 0)
    futs = [batcher.submit(i, 4) for i in range(50)]  # 200 examples waiting
    self.assertEqual(batcher.rejected, 0)
    gate.set()
    self.assertEqual([f.result(timeout=10) for f in futs], list(range(50)))

  def test_servicer_maps_queue_full_to_unavailable(self):
    class StubPredictor(python_server.Predictor):
      config = self.config

    batcher, gate, _ = self._blocked_batcher(
        result_fn=lambda payloads, batches: [
            np.full((b,), 0.5, np.float32) for b in batches
        ],
        max_batch_size=4,
        max_enqueued_batches=1,
    )
    servicer = python_server.PredictionServicer(
        StubPredictor(), common.DEFAULT_MODEL_NAME, batcher
    )
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=4),
        handlers=(servicer.generic_handler(),),
    )
    port = common.add_local_port(server, 0)
    server.start()
    self.addCleanup(server.stop, 0)
    channel = common.make_channel(f'localhost:{port}')
    self.addCleanup(channel.close)
    stub = channel.unary_unary(
        common.PREDICT_METHOD,
        request_serializer=predict_pb2.PredictRequest.SerializeToString,
        response_deserializer=predict_pb2.PredictResponse.FromString,
    )
    rng = np.random.default_rng(5)
    req4 = common.make_predict_request(
        common.generate_features(self.config, 4, rng)
    )
    # Fills the queue (4 of 4 examples) and parks behind the busy worker.
    parked = stub.future(req4, timeout=30)
    deadline = time.monotonic() + 5
    while batcher.queued_examples < 4 and time.monotonic() < deadline:
      time.sleep(0.01)
    self.assertEqual(batcher.queued_examples, 4)
    with self.assertRaises(grpc.RpcError) as cm:
      stub(req4, timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.UNAVAILABLE
    )
    self.assertIn(
        'batch scheduling queue', getattr(cm.exception, 'details')()
    )
    self.assertEqual(servicer.rejected, 1)
    self.assertEqual(servicer.requests, 0)
    # Invalid requests are still rejected before touching the batcher.
    bad = common.generate_features(self.config, 2, rng)
    del bad['0']
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(bad), timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )
    self.assertEqual(batcher.rejected, 1)
    # The admitted request completes once the worker is released.
    gate.set()
    preds = common.predictions_from_response(parked.result(timeout=30))
    np.testing.assert_array_equal(preds, np.full((4,), 0.5, np.float32))
    self.assertEqual(servicer.requests, 1)
    self.assertEqual(servicer.examples, 4)

  def test_split_across(self):
    self.assertEqual(python_server.split_across(512, 8), 64)
    self.assertEqual(python_server.split_across(8, 8), 1)
    self.assertEqual(python_server.split_across(8, 3), 3)
    self.assertEqual(python_server.split_across(2, 8), 1)
    self.assertEqual(python_server.split_across(0, 8), 0)
    self.assertEqual(python_server.split_across(64, 1), 64)

  def test_warmup_covers_all_buckets(self):
    timings = self.model.warmup()
    self.assertEqual(sorted(timings), list(_BATCH_SIZES))

  def test_warmup_without_tables_uses_run_model(self):
    owner_side = python_server.DlrmBaseline(
        self.weights_dir, device='cpu', load_tables=False
    )
    self.assertEqual(owner_side.tables, [])
    self.assertEqual(sorted(owner_side.warmup()), list(_BATCH_SIZES))
    with self.assertRaises(RuntimeError):
      owner_side.lookup(
          common.generate_features(self.config, 1, np.random.default_rng(0))
      )

  @parameterized.named_parameters(('no_batching', False), ('batching', True))
  def test_grpc_roundtrip(self, dynamic_batching):
    stub, servicer, _ = self._start(dynamic_batching)
    rng = np.random.default_rng(1)
    for batch in (1, 4, 5):
      feats = common.generate_features(self.config, batch, rng)
      resp = stub(common.make_predict_request(feats), timeout=30)
      preds = common.predictions_from_response(resp)
      self.assertEqual(resp.model_spec.name, common.DEFAULT_MODEL_NAME)
      np.testing.assert_allclose(preds, self.model.predict(feats), atol=1e-6)
    self.assertEqual(servicer.requests, 3)
    self.assertEqual(servicer.examples, 10)

  def test_dynamic_batching_coalesces_concurrent_requests(self):
    stub, _, batcher = self._start(
        True, max_batch_size=16, batch_timeout_micros=200_000
    )
    rng = np.random.default_rng(2)
    feats = [common.generate_features(self.config, 4, rng) for _ in range(4)]
    with futures.ThreadPoolExecutor(max_workers=4) as pool:
      resps = list(
          pool.map(
              lambda f: stub(common.make_predict_request(f), timeout=30), feats
          )
      )
    for f, r in zip(feats, resps):
      np.testing.assert_allclose(
          common.predictions_from_response(r), self.model.predict(f), atol=1e-6
      )
    self.assertEqual(batcher.examples_run, 16)
    # 4 concurrent requests of 4 with a generous timeout must have been merged
    # into fewer than 4 model calls.
    self.assertLess(batcher.batches_run, 4)

  def test_invalid_request_is_rejected(self):
    stub, _, _ = self._start(False)
    rng = np.random.default_rng(3)
    feats = common.generate_features(self.config, 2, rng)
    del feats['0']
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(feats), timeout=30)
    # The raised object is a `grpc.Call`; the `RpcError` stub lacks `.code()`.
    status = getattr(cm.exception, 'code')()
    self.assertEqual(status, grpc.StatusCode.INVALID_ARGUMENT)

    # Same key count (one missing, one unexpected) still reports names mismatch.
    substituted = common.generate_features(self.config, 2, rng)
    substituted['unexpected'] = substituted.pop('5')
    with self.assertRaisesRegex(ValueError, 'Input names mismatch'):
      common.validate_features(self.config, substituted)

    # Extra unexpected key alongside all expected keys is rejected.
    with_extra = common.generate_features(self.config, 2, rng)
    with_extra['extra'] = np.zeros((2, 1), dtype=np.int32)
    with self.assertRaisesRegex(ValueError, 'Input names mismatch'):
      common.validate_features(self.config, with_extra)

    # Wrong dense dtype (int32 instead of float32) is rejected as
    # INVALID_ARGUMENT.
    bad_dense = common.generate_features(self.config, 2, rng)
    bad_dense[common.DENSE_FEATURE_NAME] = bad_dense[
        common.DENSE_FEATURE_NAME
    ].astype(np.int32)
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(bad_dense), timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )

    # Inconsistent batch size across features is rejected.
    mismatched = common.generate_features(self.config, 2, rng)
    mismatched['3'] = mismatched['3'][:1]
    with self.assertRaisesRegex(ValueError, 'Inconsistent batch dimension'):
      common.validate_features(self.config, mismatched)

  def test_batcher_wakes_sibling_worker_when_queue_has_residual_items(self):
    # Item 0 (batch=2) enters timeout wait on worker 0; item 1 (batch=4) makes
    # queued_examples=6 >= max_batch_size=4, waking worker 0. Worker 0 can only
    # take item 0 (2 + 4 > 4) and must notify worker 1 so item 1 runs without
    # waiting for worker 0's run_fn to finish.
    first_entered = threading.Event()
    release_first = threading.Event()
    second_ran = threading.Event()

    def run_fn(payloads, batches):
      del batches
      if payloads == ['first']:
        first_entered.set()
        release_first.wait(5)
      elif payloads == ['second']:
        second_ran.set()
      return list(payloads)

    batcher = python_server.DynamicBatcher(
        run_fn,
        max_batch_size=4,
        timeout_s=0.5,
        buckets=(1, 2, 4),
        num_threads=2,
    )
    self.addCleanup(batcher.stop)
    self.addCleanup(release_first.set)

    fut1 = batcher.submit('first', 2)
    time.sleep(0.02)  # let worker 0 enter the batch-fill wait
    fut2 = batcher.submit('second', 4)
    self.assertTrue(first_entered.wait(5))
    self.assertTrue(second_ran.wait(2))
    self.assertEqual(fut2.result(timeout=2), 'second')
    release_first.set()
    self.assertEqual(fut1.result(timeout=2), 'first')

  def test_handler_threads_default_depends_on_batching(self):
    _, plain, _ = self._start(False)
    self.assertEqual(plain.handler_threads, 4)
    _, batched, batcher = self._start(True)
    self.assertEqual(
        batched.handler_threads,
        python_server.DEFAULT_BATCHING_HANDLER_THREADS,
    )
    self.assertEqual(batcher.max_enqueued_batches, 64)  # exporter default
    _, explicit, batcher = self._start(
        True, handler_threads=7, max_enqueued_batches=3
    )
    self.assertEqual(explicit.handler_threads, 7)
    self.assertEqual(batcher.max_enqueued_batches, 3)

  def test_batching_coalesces_more_requests_than_num_threads(self):
    # 12 concurrent single-example requests against 4 compute threads: with
    # the widened handler pool the batcher sees all of them at once.
    stub, _, batcher = self._start(
        True, max_batch_size=16, batch_timeout_micros=200_000
    )
    rng = np.random.default_rng(4)
    feats = [common.generate_features(self.config, 1, rng) for _ in range(12)]
    with futures.ThreadPoolExecutor(max_workers=12) as pool:
      list(
          pool.map(
              lambda f: stub(common.make_predict_request(f), timeout=30), feats
          )
      )
    self.assertEqual(batcher.examples_run, 12)
    self.assertLessEqual(batcher.batches_run, 2)

  def test_restrict_jax_platforms_honours_environment(self):
    with mock.patch.dict(os.environ, {'JAX_PLATFORMS': 'cpu'}):
      with mock.patch.object(jax.config, 'update') as update:
        python_server.restrict_jax_platforms('cuda')
    update.assert_not_called()
    with mock.patch.dict(os.environ, {'JAX_PLATFORMS': ''}):
      with mock.patch.object(jax.config, 'update') as update:
        python_server.restrict_jax_platforms('cuda')
    update.assert_called_once_with('jax_platforms', 'cuda,cpu')

  @parameterized.named_parameters(('no_batching', False), ('batching', True))
  def test_oversized_zero_and_malformed_requests_return_invalid_argument(
      self, dynamic_batching
  ):
    stub, _, _ = self._start(dynamic_batching)
    rng = np.random.default_rng(9)

    # Batch larger than max_batch_size (16) -> INVALID_ARGUMENT.
    oversized = common.generate_features(self.config, 32, rng)
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(oversized), timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )

    # Empty 0-row batch -> INVALID_ARGUMENT.
    empty = common.generate_features(self.config, 0, rng)
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(empty), timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )

    # Negative dim in HostBufferProto -> INVALID_ARGUMENT (not NumPy wildcard).
    malformed = common.make_predict_request(
        common.generate_features(self.config, 2, rng)
    )
    malformed.inputs[common.DENSE_FEATURE_NAME].shape.dims[:] = [-1, 13]
    with self.assertRaises(grpc.RpcError) as cm:
      stub(malformed, timeout=30)
    self.assertEqual(
        getattr(cm.exception, 'code')(), grpc.StatusCode.INVALID_ARGUMENT
    )

  def test_batcher_submit_after_stop_raises_queue_full_error(self):
    entered = threading.Event()
    completed: list[str] = []

    def slow_run(payloads, batches):
      del batches
      entered.set()
      time.sleep(0.05)
      completed.append('done')
      return list(payloads)

    batcher = python_server.DynamicBatcher(
        slow_run,
        max_batch_size=4,
        timeout_s=0.0,
        buckets=(1, 4),
    )
    fut = batcher.submit('work', 1)
    self.assertTrue(entered.wait(5))
    batcher.stop()
    # `stop()` must join in-flight worker threads before returning.
    self.assertEqual(completed, ['done'])
    self.assertEqual(fut.result(timeout=1), 'work')
    with self.assertRaisesRegex(python_server.QueueFullError, 'stopped'):
      batcher.submit('late', 1)

  @parameterized.named_parameters(
      ('localhost_port', 'localhost:8500', True),
      ('ipv4_loopback_port', '127.0.0.1:8500', True),
      ('ipv6_bracketed_port', '[::1]:8500', True),
      ('empty_host_port', ':8500', True),
      ('bare_localhost', 'localhost', True),
      ('bare_ipv6_loopback', '::1', True),
      ('empty_string', '', True),
      ('remote_ipv4_port', '10.0.0.1:8500', False),
      ('remote_hostname_port', 'example.com:8500', False),
      ('bare_remote_hostname', 'example.com', False),
  )
  def test_is_loopback_target(self, target: str, expected: bool):
    self.assertEqual(common.is_loopback_target(target), expected)

  def test_make_channel_uses_local_credentials_for_loopback_and_insecure_remote(
      self,
  ):
    _, servicer, _ = self._start(False)
    with mock.patch.object(
        grpc,
        'insecure_channel',
        autospec=True,
        spec_set=True,
        side_effect=NotImplementedError('insecure disallowed'),
    ) as insecure_mock:
      loopback_ch = common.make_channel(f'localhost:{servicer.port}')
      self.addCleanup(loopback_ch.close)
      stub = loopback_ch.unary_unary(
          common.PREDICT_METHOD,
          request_serializer=predict_pb2.PredictRequest.SerializeToString,
          response_deserializer=predict_pb2.PredictResponse.FromString,
      )
      feats = common.generate_features(
          self.config, 2, np.random.default_rng(12)
      )
      resp = stub(common.make_predict_request(feats), timeout=30)
      np.testing.assert_allclose(
          common.predictions_from_response(resp),
          self.model.predict(feats),
          atol=1e-6,
      )
      insecure_mock.assert_not_called()

    opts = [('grpc.max_send_message_length', 1024)]
    sentinel = mock.create_autospec(grpc.Channel, instance=True, spec_set=True)
    with mock.patch.object(
        grpc,
        'insecure_channel',
        autospec=True,
        spec_set=True,
        return_value=sentinel,
    ) as insecure_mock:
      got = common.make_channel('example.com:8500', options=opts)
    self.assertIs(got, sentinel)
    insecure_mock.assert_called_once_with('example.com:8500', options=opts)

  def test_make_model_config_uncapped_vocab_and_batch_normalization(self):
    cfg = common.make_model_config(
        embedding_size=8, batch_sizes=(16, 1, 4, 1), seed=7
    )
    self.assertEqual(cfg.vocab_sizes, tuple(model_jce.VOCAB_SIZES))
    self.assertEqual(cfg.valency, tuple(model_jce.VALENCY))
    self.assertEqual(cfg.batch_sizes, (1, 4, 16))
    self.assertEqual(cfg.seed, 7)

  @parameterized.named_parameters(
      (
          'module_spec',
          '/usr/bin/python3',
          types.SimpleNamespace(name='pkg.tool'),
          '/app/tool.py',
          ['/usr/bin/python3', '-m', 'pkg.tool'],
      ),
      (
          'empty_spec_name_script',
          '/usr/bin/python3',
          types.SimpleNamespace(name=''),
          '/app/tool.py',
          ['/usr/bin/python3', '/app/tool.py'],
      ),
      (
          'no_spec_script',
          '/usr/bin/python3',
          None,
          '/app/tool.py',
          ['/usr/bin/python3', '/app/tool.py'],
      ),
      (
          'standalone_binary_with_python',
          '/usr/bin/python3',
          None,
          '/opt/bin/server_bin',
          ['/opt/bin/server_bin'],
      ),
      (
          'empty_executable_with_spec',
          '',
          types.SimpleNamespace(name='pkg.tool'),
          '/opt/bin/server_bin',
          ['/opt/bin/server_bin'],
      ),
      (
          'empty_executable_with_py_argv',
          '',
          None,
          '/app/tool.py',
          ['/app/tool.py'],
      ),
  )
  def test_self_command(
      self, executable: str, spec: object, argv0: str, expected: list[str]
  ):
    fake_main = types.ModuleType('__main__')
    fake_main.__spec__ = spec  # type: ignore[attr-defined]
    with (
        mock.patch.dict(sys.modules, {'__main__': fake_main}),
        mock.patch.object(sys, 'executable', executable),
        mock.patch.object(sys, 'argv', [argv0, '--flag=1']),
    ):
      self.assertEqual(common.self_command(), expected)

  def test_latency_summary_ms_empty_and_populated(self):
    empty = common.latency_summary_ms([])
    self.assertEqual(
        set(empty), {'p50', 'p90', 'p99', 'p999', 'max', 'mean'}
    )
    for key, val in empty.items():
      self.assertTrue(math.isnan(val), msg=key)

    stats = common.latency_summary_ms([0.01, 0.02, 0.03, 0.04])
    self.assertAlmostEqual(stats['p50'], 25.0)
    self.assertAlmostEqual(stats['max'], 40.0)
    self.assertAlmostEqual(stats['mean'], 25.0)
    self.assertGreater(stats['p90'], stats['p50'])
    self.assertGreater(stats['p99'], stats['p90'])
    self.assertGreater(stats['p999'], stats['p99'])

  def test_grpc_server_options(self):
    mb = 1024 * 1024
    self.assertEqual(
        python_server.grpc_server_options(),
        [
            ('grpc.max_receive_message_length', 64 * mb),
            ('grpc.max_send_message_length', 64 * mb),
        ],
    )
    self.assertEqual(
        python_server.grpc_server_options(1),
        [
            ('grpc.max_receive_message_length', mb),
            ('grpc.max_send_message_length', mb),
        ],
    )
    with self.assertRaisesRegex(ValueError, 'max_message_mb'):
      python_server.grpc_server_options(0)

  def test_predictions_from_response_and_existing_dir_save(self):
    multi = common.make_predict_response({
        common.OUTPUT_NAME: np.array([1.5, 2.5], dtype=np.float32),
        'aux_logits': np.array([9.0, 9.0], dtype=np.float32),
    })
    np.testing.assert_array_equal(
        common.predictions_from_response(multi),
        np.array([1.5, 2.5], dtype=np.float32),
    )

    single_custom = common.make_predict_response({
        'score': np.array([[3.25]], dtype=np.float32),
    })
    np.testing.assert_array_equal(
        common.predictions_from_response(single_custom),
        np.array([3.25], dtype=np.float32),
    )

    ambiguous = common.make_predict_response({
        'a': np.array([1.0], dtype=np.float32),
        'b': np.array([2.0], dtype=np.float32),
    })
    with self.assertRaisesRegex(ValueError, 'Cannot find predictions'):
      common.predictions_from_response(ambiguous)

    existing_dir = self.create_tempdir().full_path
    self.config.save(existing_dir)
    self.assertEqual(common.ModelConfig.load(existing_dir), self.config)

    feats3 = common.generate_features(
        self.config, 3, np.random.default_rng(41)
    )
    padded8 = common.pad_features(feats3, 8)
    self.assertEqual(padded8['0'].shape, (8, self.config.valency[0]))
    np.testing.assert_array_equal(padded8['0'][:3], feats3['0'])
    self.assertTrue(np.all(padded8['0'][3:] == 0))
    same3 = common.pad_features(feats3, 3)
    self.assertEqual(same3['0'].shape, (3, self.config.valency[0]))
    with self.assertRaisesRegex(ValueError, 'Cannot pad batch'):
      common.pad_features(feats3, 2)


if __name__ == '__main__':
  absltest.main()

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
import os
import threading
import time
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import grpc
import jax
import numpy as np

from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as export
from examples.dlrm_benchmark import python_server
from xla_saved_model.protos.grpc import predict_pb2

_BATCH_SIZES = (1, 4, 16)


class PythonServerTest(parameterized.TestCase):

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
    # The proto package is `jsv.serving` in  and `xla_serving.serving`
    # in the OSS export; the method path must follow it or the real XSE server
    # answers every call with UNIMPLEMENTED. Go through the message descriptor
    # rather than `predict_pb2.DESCRIPTOR` so this is an independent route.
    package = predict_pb2.PredictRequest.DESCRIPTOR.file.package
    self.assertEqual(common.PREDICT_SERVICE, f'{package}.PredictionService')
    self.assertEqual(
        common.PREDICT_METHOD, f'/{package}.PredictionService/Predict'
    )
    self.assertNotIn('jsv', common.PREDICT_METHOD.replace(package, ''))

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
    feats = common.generate_features(self.config, 2, np.random.default_rng(3))
    del feats['0']
    with self.assertRaises(grpc.RpcError) as cm:
      stub(common.make_predict_request(feats), timeout=30)
    # The raised object is a `grpc.Call`; the `RpcError` stub lacks `.code()`.
    status = getattr(cm.exception, 'code')()
    self.assertEqual(status, grpc.StatusCode.INVALID_ARGUMENT)

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


if __name__ == '__main__':
  absltest.main()

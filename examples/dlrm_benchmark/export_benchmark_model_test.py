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

"""Tests the benchmark export: batch>1 through JCE + numeric parity."""

import os
import shutil
import tempfile
from unittest import mock

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np

from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as export
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model_runner import model_runner

_EMBEDDING_SIZE = 16
_MAX_VOCAB = 50
_BATCH_SIZES = (1, 4, 16)


class ExportBenchmarkModelTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.root = tempfile.mkdtemp(prefix='dlrm_export_test_')
    cls.addClassCleanup(shutil.rmtree, cls.root, True)
    cls.batched_dir = os.path.join(cls.root, 'xse_batched')
    cls.unbatched_dir = os.path.join(cls.root, 'xse_unbatched')
    cls.weights_dir = os.path.join(cls.root, 'baseline_weights')
    cls.config = export.export_model(
        embedding_size=_EMBEDDING_SIZE,
        max_vocab_size=_MAX_VOCAB,
        batch_sizes=_BATCH_SIZES,
        platforms=['cpu'],
        batched_export_path=cls.batched_dir,
        unbatched_export_path=cls.unbatched_dir,
        baseline_weights_dir=cls.weights_dir,
    )

  def _reference(self, features):
    config, tables, params = common.load_baseline_weights(self.weights_dir)
    self.assertEqual(config, self.config)
    model = model_jce.build_model(embedding_size=config.embedding_size)
    embs = common.embedding_lookup_numpy(tables, features)
    return np.asarray(
        model.apply(params, features[common.DENSE_FEATURE_NAME], embs)
    )

  def test_artifacts_exist(self):
    for d in (self.batched_dir, self.unbatched_dir):
      orch_path = os.path.join(d, 'orchestration.pb')
      self.assertTrue(os.path.isfile(orch_path))
      with open(orch_path, 'rb') as f:
        pipelines = orchestration_pb2.Pipelines.FromString(f.read())
      pipeline = pipelines.name_to_pipeline[common.DEFAULT_SIGNATURE]
      expected_bc = (
          orchestration_pb2.BatchOptions.PRE_PROCESSOR_AND_MODEL_FUNCTION
          if d == self.batched_dir
          else orchestration_pb2.BatchOptions.NO_BATCHING
      )
      self.assertEqual(pipeline.batch_options.batch_component, expected_bc)
      input_dims = {
          nt.name: nt.tensor_type.shape.shape_with_known_rank.dimension_sizes
          for nt in pipeline.named_signature.inputs.named_tensor_types
      }
      for i, valency in enumerate(self.config.valency):
        dims = input_dims[str(i)]
        self.assertLen(dims, 2)
        self.assertFalse(dims[0].HasField('size'))
        self.assertEqual(dims[1].size, valency)
      dense_dims = input_dims[common.DENSE_FEATURE_NAME]
      self.assertLen(dense_dims, 2)
      self.assertFalse(dense_dims[0].HasField('size'))
      self.assertEqual(dense_dims[1].size, self.config.num_dense_features)

      warmup_path = os.path.join(
          d, 'assets.extra', 'xla_serving_warmup_requests'
      )
      self.assertTrue(os.path.isfile(warmup_path))
      self.assertGreater(os.path.getsize(warmup_path), 0)
      self.assertTrue(
          os.path.isfile(
              os.path.join(d, 'assets.extra', common.CONFIG_FILENAME)
          )
      )
    for name in (
        common.CONFIG_FILENAME,
        common.TABLES_FILENAME,
        common.PARAMS_FILENAME,
    ):
      self.assertTrue(os.path.isfile(os.path.join(self.weights_dir, name)))
    self.assertEqual(self.config.batch_sizes, _BATCH_SIZES)
    self.assertEqual(self.config.vocab_sizes[0], _MAX_VOCAB)
    _, tables, _ = common.load_baseline_weights(self.weights_dir, params=False)
    self.assertGreaterEqual(float(tables[0].min()), -0.05)
    self.assertLess(float(tables[0].min()), 0.0)
    self.assertGreater(float(tables[0].max()), 0.0)
    self.assertLessEqual(float(tables[0].max()), 0.05)

  @parameterized.product(
      batch_size=_BATCH_SIZES, export_kind=('batched', 'unbatched')
  )
  def test_xse_matches_numpy_reference(self, batch_size, export_kind):
    export_dir = (
        self.batched_dir if export_kind == 'batched' else self.unbatched_dir
    )
    runner = model_runner.ModelRunner(model_path=export_dir)
    signature = runner.signature(common.DEFAULT_SIGNATURE)
    rng = np.random.default_rng(batch_size)
    features = common.generate_features(self.config, batch_size, rng)

    outputs = signature.run_with_dict_return(**features)
    self.assertIn(common.OUTPUT_NAME, outputs)
    preds = np.asarray(outputs[common.OUTPUT_NAME]).reshape(-1)
    self.assertEqual(preds.shape, (batch_size,))
    self.assertEqual(preds.dtype, np.float32)

    np.testing.assert_allclose(preds, self._reference(features), atol=1e-5)

  def test_batched_export_pads_odd_batch_size(self):
    # 3 is not an allowed batch size: the scheduler must PAD_UP to 4 and strip
    # the padding from the output.
    runner = model_runner.ModelRunner(model_path=self.batched_dir)
    signature = runner.signature(common.DEFAULT_SIGNATURE)
    features = common.generate_features(
        self.config, 3, np.random.default_rng(3)
    )
    outputs = signature.run_with_dict_return(**features)
    preds = np.asarray(outputs[common.OUTPUT_NAME]).reshape(-1)
    self.assertEqual(preds.shape, (3,))
    np.testing.assert_allclose(preds, self._reference(features), atol=1e-5)

  def test_batch_options_carry_num_batch_threads_and_buckets(self):
    opts = export._batch_options(
        self.config,
        batched=True,
        batch_component='PRE_PROCESSOR_AND_MODEL_FUNCTION',
        batch_timeout_micros=1000,
        num_batch_threads=7,
        max_enqueued_batches=64,
    )
    self.assertEqual(opts.num_batch_threads, 7)
    self.assertEqual(list(opts.allowed_batch_sizes), list(_BATCH_SIZES))
    self.assertEqual(opts.max_batch_size, _BATCH_SIZES[-1])
    self.assertEqual(opts.batch_timeout_micros, 1000)
    self.assertEqual(opts.max_enqueued_batches, 64)
    self.assertEqual(opts.batch_padding_policy, opts.PAD_UP)

    unbatched = export._batch_options(
        self.config,
        batched=False,
        batch_component='PRE_PROCESSOR_AND_MODEL_FUNCTION',
        batch_timeout_micros=1000,
        num_batch_threads=7,
        max_enqueued_batches=64,
    )
    self.assertEqual(unbatched.batch_component, unbatched.NO_BATCHING)

  def test_default_batch_sizes_are_powers_of_two(self):
    # Power-of-two buckets ensure PAD_UP never wastes more than 2x.
    sizes = common.DEFAULT_BATCH_SIZES
    self.assertEqual(sizes[0], 1)
    self.assertEqual(sizes[-1], 4096)
    for a, b in zip(sizes, sizes[1:]):
      self.assertEqual(b, 2 * a)

  def test_jce_max_cpu_slots_forwarded_to_resource(self):
    custom_dir = os.path.join(self.root, 'xse_slots_2')
    seen_slots = []
    real_cls = export.cpu_embedding_module.CpuEmbeddingResource

    def spy_resource(*args, **kwargs):
      seen_slots.append(kwargs.get('max_cpu_slots'))
      return real_cls(*args, **kwargs)

    with mock.patch.object(
        export.cpu_embedding_module,
        'CpuEmbeddingResource',
        side_effect=spy_resource,
    ):
      export.export_model(
          embedding_size=_EMBEDDING_SIZE,
          max_vocab_size=_MAX_VOCAB,
          batch_sizes=_BATCH_SIZES,
          platforms=['cpu'],
          unbatched_export_path=custom_dir,
          jce_max_cpu_slots=2,
      )
    self.assertEqual(seen_slots, [2])
    runner = model_runner.ModelRunner(model_path=custom_dir)
    features = common.generate_features(
        self.config, 4, np.random.default_rng(11)
    )
    preds = np.asarray(
        runner.signature(common.DEFAULT_SIGNATURE).run_with_dict_return(
            **features
        )[common.OUTPUT_NAME]
    ).reshape(-1)
    np.testing.assert_allclose(preds, self._reference(features), atol=1e-5)

  def test_invalid_export_parameters_rejected(self):
    out_dir = os.path.join(self.root, 'invalid_export')
    with self.assertRaisesRegex(ValueError, 'Nothing to export'):
      export.export_model()
    with self.assertRaisesRegex(ValueError, 'jce_max_cpu_slots'):
      export.export_model(
          unbatched_export_path=out_dir, jce_max_cpu_slots=0
      )
    with self.assertRaisesRegex(ValueError, 'num_batch_threads'):
      export.export_model(batched_export_path=out_dir, num_batch_threads=0)
    with self.assertRaisesRegex(ValueError, 'embedding_size'):
      common.make_model_config(embedding_size=0)
    with self.assertRaisesRegex(ValueError, 'batch_sizes'):
      common.make_model_config(embedding_size=8, batch_sizes=())
    with self.assertRaisesRegex(ValueError, 'batch_sizes'):
      common.make_model_config(embedding_size=8, batch_sizes=(0, 4))


if __name__ == '__main__':
  absltest.main()

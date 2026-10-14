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

"""Tests the TF-preprocessor export: both lookups, batched/unbatched, parity."""

import os

from absl.testing import absltest
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from examples.dlrm import model_jce
from examples.dlrm_benchmark import common
from examples.dlrm_benchmark import export_benchmark_model as jce_export
from examples.dlrm_benchmark import export_tf_benchmark_model as tf_export
from xla_saved_model_runner import model_runner

_EMBEDDING_SIZE = 16
_MAX_VOCAB = 50
_BATCH_SIZES = (1, 4, 16)


class ExportTfBenchmarkModelTest(parameterized.TestCase):

  @classmethod
  def setUpClass(cls):
    super().setUpClass()
    cls.root = absltest.get_default_test_tmpdir()
    cls.weights_dir = os.path.join(cls.root, 'baseline_weights')
    # Weights come from the JCE exporter, exactly as in run_benchmark.sh.
    cls.config = jce_export.export_model(
        embedding_size=_EMBEDDING_SIZE,
        max_vocab_size=_MAX_VOCAB,
        batch_sizes=_BATCH_SIZES,
        platforms=['cpu'],
        baseline_weights_dir=cls.weights_dir,
    )
    cls.export_dirs = {}
    for impl in tf_export.TF_LOOKUPS:
      batched = os.path.join(cls.root, f'xse_tf_{impl}_batched')
      unbatched = os.path.join(cls.root, f'xse_tf_{impl}_unbatched')
      config = tf_export.export_tf_model(
          weights_dir=cls.weights_dir,
          batched_export_path=batched,
          unbatched_export_path=unbatched,
          tf_lookup=impl,
          platforms=['cpu'],
      )
      assert config == cls.config, (config, cls.config)
      cls.export_dirs[(impl, 'batched')] = batched
      cls.export_dirs[(impl, 'unbatched')] = unbatched
    _, cls.tables, cls.params = common.load_baseline_weights(cls.weights_dir)

  def _reference(self, features):
    model = model_jce.build_model(embedding_size=self.config.embedding_size)
    embs = common.embedding_lookup_numpy(self.tables, features)
    return np.asarray(
        model.apply(self.params, features[common.DENSE_FEATURE_NAME], embs)
    )

  def test_artifacts_exist(self):
    for d in self.export_dirs.values():
      self.assertTrue(os.path.isfile(os.path.join(d, 'orchestration.pb')))
      # The preprocessor is a TF SavedModel next to the XSE manifest.
      self.assertTrue(
          os.path.isfile(
              os.path.join(d, tf_export.TF_SAVED_MODEL_SUBDIR, 'saved_model.pb')
          )
      )
      self.assertTrue(
          os.path.isfile(
              os.path.join(d, 'assets.extra', 'xla_serving_warmup_requests')
          )
      )
      self.assertTrue(
          os.path.isfile(
              os.path.join(d, 'assets.extra', common.CONFIG_FILENAME)
          )
      )

  @parameterized.product(
      impl=tf_export.TF_LOOKUPS,
      batch_size=_BATCH_SIZES,
      export_kind=('batched', 'unbatched'),
  )
  def test_xse_matches_numpy_reference(self, impl, batch_size, export_kind):
    runner = model_runner.ModelRunner(
        model_path=self.export_dirs[(impl, export_kind)]
    )
    signature = runner.signature(common.DEFAULT_SIGNATURE)
    rng = np.random.default_rng(batch_size)
    features = common.generate_features(self.config, batch_size, rng)

    outputs = signature.run_with_dict_return(**features)
    self.assertIn(common.OUTPUT_NAME, outputs)
    preds = np.asarray(outputs[common.OUTPUT_NAME]).reshape(-1)
    self.assertEqual(preds.shape, (batch_size,))
    self.assertEqual(preds.dtype, np.float32)
    np.testing.assert_allclose(preds, self._reference(features), atol=1e-5)

  @parameterized.parameters(*tf_export.TF_LOOKUPS)
  def test_batched_export_pads_odd_batch_size(self, impl):
    # 3 is not an allowed batch size: PAD_UP to 4 through the TF preprocessor
    # (HostBuffer in / HostBuffer out) and strip the padding again.
    runner = model_runner.ModelRunner(
        model_path=self.export_dirs[(impl, 'batched')]
    )
    signature = runner.signature(common.DEFAULT_SIGNATURE)
    features = common.generate_features(
        self.config, 3, np.random.default_rng(3)
    )
    outputs = signature.run_with_dict_return(**features)
    preds = np.asarray(outputs[common.OUTPUT_NAME]).reshape(-1)
    self.assertEqual(preds.shape, (3,))
    np.testing.assert_allclose(preds, self._reference(features), atol=1e-5)

  @parameterized.product(impl=tf_export.TF_LOOKUPS, batch_size=(1, 5, 32))
  def test_tf_lookup_matches_numpy_for_every_valency(self, impl, batch_size):
    # Eager check of the lookup itself, covering V=1 (plain gather), the
    # multi-valent features and V=100 (feature 20) at an odd batch size.
    self.assertIn(1, self.config.valency)
    self.assertIn(100, self.config.valency)
    module = tf_export.TfEmbeddingTables(self.tables)
    features = common.generate_features(
        self.config, batch_size, np.random.default_rng(7)
    )
    tf_features = {
        k: tf.constant(v) for k, v in features.items() if k.isdigit()
    }
    got = tf_export.tf_embedding_lookup(module.tables, tf_features, impl)
    want = common.embedding_lookup_numpy(self.tables, features)
    self.assertEqual(set(got), set(want))
    for name in want:
      self.assertEqual(got[name].dtype, tf.float32)
      np.testing.assert_allclose(
          got[name].numpy(), want[name], atol=1e-6, err_msg=f'feature {name}'
      )

  def test_fused_and_gather_traced_graphs_use_the_expected_kernels(self):
    module = tf_export.TfEmbeddingTables(self.tables)
    signature = tf_export.tf_input_signature(self.config)

    def op_types(impl):
      fn = tf_export.make_tf_preprocessor(module, self.config, impl)
      graph = fn.get_concrete_function(signature).graph
      return {op.type for op in graph.get_operations()}

    fused = op_types(tf_export.TF_LOOKUP_FUSED)
    self.assertTrue(
        fused & {'SparseSegmentSum', 'SparseSegmentSumWithNumSegments'},
        fused,
    )
    # No [B, V, D] intermediate: the only reductions are the fused kernels.
    self.assertNotIn('Sum', fused)

    gather = op_types(tf_export.TF_LOOKUP_GATHER)
    self.assertIn('ResourceGather', gather)
    self.assertIn('Sum', gather)
    self.assertFalse(
        gather & {'SparseSegmentSum', 'SparseSegmentSumWithNumSegments'},
        gather,
    )

  def test_rejects_unknown_lookup(self):
    with self.assertRaises(ValueError):
      tf_export.export_tf_model(
          weights_dir=self.weights_dir,
          batched_export_path=os.path.join(self.root, 'never'),
          tf_lookup='bogus',
      )
    with self.assertRaises(ValueError):
      tf_export.export_tf_model(weights_dir=self.weights_dir)


if __name__ == '__main__':
  absltest.main()

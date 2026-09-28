# Copyright 2026 Google LLC
# Copyright 2026 Arm Ltd. and affiliates
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

"""Tests for XLA SavedModel export of DLRM model."""

import os
from unittest import mock

from absl.testing import absltest
import numpy as np

from examples.dlrm import benchmark_model_runner
from examples.dlrm import export_jce as export
from examples.dlrm import model_jce as model_lib
from xla_saved_model_runner import model_runner


class ExportTest(absltest.TestCase):

  # A seeded generator must reproduce dense and sparse inputs within bounds.
  def test_synthetic_inputs_are_reproducible(self):
    batches = [export.get_synthetic_batch(
        4, valency=[2, 3], vocab_sizes=[7, 11],
        rng=np.random.default_rng(7),
    ) for _ in range(2)]
    np.testing.assert_equal(*batches)
    self.assertEqual(batches[0]['dense_features'].dtype, np.float32)
    self.assertEqual(batches[0]['dense_features'].shape, (4, 13))
    for name, width, vocab_size in [('0', 2, 7), ('1', 3, 11)]:
      values = batches[0]['sparse_features'][name]
      self.assertEqual(values.dtype, np.int32)
      self.assertEqual(values.shape, (4, width))
      self.assertTrue(((values >= 0) & (values < vocab_size)).all())

  # Reject an existing benchmark archive before starting model export.
  def test_existing_benchmark_archive_is_not_overwritten(self):
    contents = b'existing inputs'
    path = self.create_tempfile(content=contents, mode='wb').full_path
    with mock.patch.object(export.main_lib, 'save') as save:
      with self.assertRaises(FileExistsError):
        export.export_model('unused', vocab_size_cap=32, benchmark_inputs=path)
      save.assert_not_called()
    with open(path, 'rb') as stream:
      self.assertEqual(stream.read(), contents)

  # Native inference must match the saved reference for every input batch.
  def test_load_and_run_inference(self):
    export_path = self.create_tempdir().full_path
    inputs_path = os.path.join(self.create_tempdir().full_path, 'inputs.npz')
    export.export_model(
        export_path, vocab_size_cap=32, global_batch_size=4,
        benchmark_inputs=inputs_path, num_input_batches=2,
    )

    runner = model_runner.ModelRunner(model_path=export_path)
    self.assertIn('serving_default', runner.signature_names())
    signature = runner.signature('serving_default')

    batches, expected = benchmark_model_runner.load_inputs(inputs_path)
    self.assertEqual(expected.shape, (2, 4))
    self.assertLen(batches, len(expected))
    for inputs, predictions in zip(batches, expected):
      outputs = signature.run_with_dict_return(**inputs)
      benchmark_model_runner.check_predictions(outputs['predictions'], predictions)

  # Default exports must preserve output shape and bind shuffled inputs by name.
  def test_load_and_run_inference_shuffled_order(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, use_capped_vocab=True)
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

    runner = model_runner.ModelRunner(model_path=export_path)
    signature = runner.signature('serving_default')

    batch = export.get_synthetic_batch(
        1, vocab_sizes=[min(v, 100) for v in model_lib.VOCAB_SIZES],
        rng=np.random.default_rng(7),
    )
    inputs = {**batch['sparse_features'], 'dense_features': batch['dense_features']}

    outputs_original = signature.run(**inputs)
    self.assertLen(outputs_original, 1)
    predictions = np.asarray(outputs_original[0])
    self.assertEqual(predictions.shape, (1,))
    self.assertEqual(predictions.dtype, np.float32)

    # Reverse insertion order; input names must still determine argument binding.
    outputs_shuffled = signature.run(**dict(reversed(inputs.items())))

    np.testing.assert_array_equal(
        predictions, np.asarray(outputs_shuffled[0])
    )


if __name__ == '__main__':
  absltest.main()

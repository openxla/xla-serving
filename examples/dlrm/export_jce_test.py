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

"""Tests for XLA SavedModel export of DLRM model."""

import os

from absl.testing import absltest
import numpy as np

from examples.dlrm import export_jce as export
from examples.dlrm import model_jce as model_lib
from xla_saved_model_runner import model_runner


class ExportTest(absltest.TestCase):

  def test_export_model(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, use_capped_vocab=True)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

  def test_load_and_run_inference(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, use_capped_vocab=True)

    # Load exported model
    runner = model_runner.ModelRunner(model_path=export_path)
    self.assertIn('serving_default', runner.signature_names())
    signature = runner.signature('serving_default')

    # Prepare inputs
    batch_size = 1
    vocab_sizes = [min(v, 100) for v in model_lib.VOCAB_SIZES]
    inputs = {}
    for i, rename_me in enumerate(model_lib.VALENCY):
      inputs[str(i)] = np.random.randint(
          0, vocab_sizes[i], size=(batch_size, rename_me), dtype=np.int32
      )
    inputs['dense_features'] = np.random.uniform(
        0.0, 1.0, size=(batch_size, 13)
    ).astype(np.float32)

    # Run full preprocessor pre-execution + model inference in C++ serving!
    outputs = signature.run(**inputs)
    self.assertLen(outputs, 1)
    predictions = np.array(outputs[0])
    self.assertEqual(predictions.shape, (batch_size,))
    self.assertEqual(predictions.dtype, np.float32)

  def test_load_and_run_inference_shuffled_order(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, use_capped_vocab=True)

    runner = model_runner.ModelRunner(model_path=export_path)
    signature = runner.signature('serving_default')

    batch_size = 1
    vocab_sizes = [min(v, 100) for v in model_lib.VOCAB_SIZES]
    inputs = {}
    for i, rename_me in enumerate(model_lib.VALENCY):
      inputs[str(i)] = np.random.randint(
          0, vocab_sizes[i], size=(batch_size, rename_me), dtype=np.int32
      )
    inputs['dense_features'] = np.random.uniform(
        0.0, 1.0, size=(batch_size, 13)
    ).astype(np.float32)

    # Run inference with original (forward) insertion order
    outputs_original = signature.run(**inputs)

    # Shuffled insertion order: Insert dense_features first, then even, then odd indices
    shuffled_inputs = {}
    shuffled_inputs['dense_features'] = inputs['dense_features']
    for i in list(range(0, 26, 2)) + list(range(1, 26, 2)):
      shuffled_inputs[str(i)] = inputs[str(i)]

    # Run inference with jumbled/shuffled insertion order
    outputs_shuffled = signature.run(**shuffled_inputs)

    # Verify predictions are 100% IDENTICAL
    np.testing.assert_array_equal(
        np.array(outputs_original[0]), np.array(outputs_shuffled[0])
    )


if __name__ == '__main__':
  absltest.main()

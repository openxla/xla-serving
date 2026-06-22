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

"""Tests for Neptune export of DLRM model."""

import os

from absl.testing import absltest
import numpy as np

from examples.dlrm import export_jce as export
from examples.dlrm import model_jce as model_lib
from .third_party.neptune.runtime.python import orbax_model_runner


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
    runner = orbax_model_runner.ModelRunner(model_path=export_path)
    self.assertIn('serving_default', runner.signature_names())
    signature = runner.signature('serving_default')

    # Prepare inputs
    batch_size = 2
    inputs = {}
    for i, rename_me in enumerate(model_lib.VALENCY):
      inputs[str(i)] = np.random.randint(
          0, 100, size=(batch_size, rename_me), dtype=np.int32
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


if __name__ == '__main__':
  absltest.main()

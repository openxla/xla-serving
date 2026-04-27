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

"""Tests for Neptune export of NNX model with JAX preprocessor."""

import os
from absl.testing import absltest
import numpy as np
from .third_party.neptune.examples.nnx_jax_preprocessor import export
from .third_party.neptune.runtime.python import orbax_model_runner


class ExportTest(absltest.TestCase):

  def test_export_model(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(os.path.isdir(os.path.join(export_path, 'checkpoint')))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )
    self.assertTrue(
        os.path.isdir(os.path.join(export_path, 'preprocessor_ckpt'))
    )

    # Run the model using Orbax ModelRunner
    runner = orbax_model_runner.ModelRunner(model_path=export_path)
    inputs = {
        'feature_1': np.zeros((2, 1), dtype=np.int32),
        'feature_2': np.zeros((2, 1), dtype=np.int32),
        'feature_3': np.zeros((2, 1), dtype=np.int32),
        'feature_4': np.zeros((2, 1), dtype=np.int32),
    }
    outputs = runner.signature('serving_default').run_with_dict_return(**inputs)
    self.assertIn('pred', outputs)
    self.assertEqual(outputs['pred'].shape, (2, 1, 1))


if __name__ == '__main__':
  absltest.main()

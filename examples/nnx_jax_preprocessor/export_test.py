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

"""Tests for XLA SavedModel export of NNX model with JAX preprocessor."""

import os
from absl.testing import absltest
import numpy as np
from examples.nnx_jax_preprocessor import export


def _model_runner(model_path: str):
  from xla_saved_model_runner import model_runner  # pylint: disable=g-import-not-at-top

  return model_runner.ModelRunner(model_path=model_path)


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
        os.path.isdir(
            os.path.join(export_path, 'checkpoint', 'embedding_layer')
        )
    )
    self.assertTrue(
        os.path.isdir(os.path.join(export_path, 'checkpoint', 'jax_model'))
    )

    runner = _model_runner(export_path)

    inputs = {
        'feature_1': np.ones((3, 1), dtype=np.int32),
        'feature_2': np.ones((3, 1), dtype=np.int32),
        'feature_3': np.ones((3, 1), dtype=np.int32),
        'feature_4': np.ones((3, 1), dtype=np.int32),
    }
    outputs = runner.signature('serving_default').run_with_dict_return(**inputs)

    self.assertIn('pred', outputs)
    self.assertEqual(outputs['pred'].shape, (3, 1, 1))


if __name__ == '__main__':
  absltest.main()

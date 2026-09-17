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

"""Tests for Neptune export of NNX model."""

import os
from absl import logging
from absl.testing import absltest
from absl.testing import parameterized
import jax.numpy as jnp
import numpy as np
from examples.nnx import export


def _model_runner(model_path: str):
  try:
    from xla_saved_model_runner import model_runner  # pylint: disable=g-import-not-at-top

    return model_runner.ModelRunner(model_path=model_path)
  except ImportError:
    return None


class ExportTest(parameterized.TestCase):

  @parameterized.named_parameters(
      ('float32', jnp.float32),
      ('bfloat16', jnp.bfloat16),
  )
  def test_export_model(self, dtype):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path, dtype=dtype)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(os.path.isdir(os.path.join(export_path, 'checkpoint')))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )
    runner = _model_runner(export_path)
    if runner is None:
      logging.warning(
          'ModelRunner not found, skipping running the exported model'
      )
      return

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

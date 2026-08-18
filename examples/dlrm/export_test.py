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
from examples.dlrm import export
from examples.dlrm import model as model_lib
from neptune_model_runner import model_runner


class ExportTest(absltest.TestCase):

  def test_export_model(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path)
    self.assertTrue(os.path.exists(export_path))
    self.assertTrue(os.path.isdir(os.path.join(export_path, 'checkpoint')))
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

    runner = model_runner.ModelRunner(model_path=export_path)

    batch = export.get_synthetic_batch(
        batch_size=1,
        valency=model_lib.VALENCY,
        vocab_sizes=model_lib.VOCAB_SIZES,
    )
    runner_inputs = {
        'dense_features': batch['dense_features'],
    }
    for k, v in batch['sparse_features'].items():
      runner_inputs[k] = v.values.numpy()
      runner_inputs[f'{k}_1'] = v.row_splits.numpy()

    outputs = runner.signature('serving_default').run_with_dict_return(
        **runner_inputs
    )
    self.assertIn('predictions', outputs)
    self.assertEqual(outputs['predictions'].shape, (1,))


if __name__ == '__main__':
  absltest.main()

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

"""Tests for XLA SavedModel export of NNX model."""

import os

from absl.testing import absltest
import tensorflow as tf

from examples.nnx_tf_ragged import export


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

    runner = _model_runner(export_path)

    ragged = tf.ragged.constant([[1, 2], [3, 4, 5]], dtype=tf.int32)
    sparse = tf.sparse.SparseTensor(
        indices=[[0, 1], [1, 0]], values=[10, 20], dense_shape=[2, 2]
    )

    # ModelRunner works with dense tensor inputs, so we flatten the ragged
    # and sparse tensors.
    inputs = {
        'ragged_input': ragged.values.numpy(),
        'ragged_input_1': ragged.row_splits.numpy(),
        'sparse_input': sparse.indices.numpy(),
        'sparse_input_1': sparse.values.numpy(),
        'sparse_input_2': sparse.dense_shape.numpy(),
    }
    outputs = runner.signature('serving_default').run_with_dict_return(**inputs)
    self.assertIn('pred', outputs)
    self.assertEqual(outputs['pred'].shape, (2, 1))
    self.assertSequenceAlmostEqual(
        outputs['pred'].flatten().tolist(), [72.0, 168.0], places=5
    )


if __name__ == '__main__':
  absltest.main()

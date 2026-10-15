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

"""Tests for the math and bitwise data ops preprocessor example."""

import os

from absl.testing import absltest
import numpy as np

from examples.math_preprocessor import export
from xla_saved_model_runner import model_runner


def _make_inputs(batch_size: int) -> dict[str, np.ndarray]:
  rng = np.random.default_rng(seed=0)
  shape = (batch_size, 1)
  return {
      'item_id': rng.integers(0, 1000, size=shape, dtype=np.int32),
      'price': rng.uniform(0.0, 100.0, size=shape).astype(np.float32),
      'quantity': rng.integers(1, 10, size=shape).astype(np.float32),
      'user_id': rng.integers(0, 1000, size=shape, dtype=np.int32),
  }


def _expected_features(
    inputs: dict[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
  log_revenue = np.log1p(inputs['price'] * inputs['quantity'])
  user_item_cross = np.bitwise_xor(inputs['user_id'], inputs['item_id'])
  return log_revenue, user_item_cross


class ExportTest(absltest.TestCase):

  def test_compute_features_eagerly(self):
    inputs = _make_inputs(batch_size=4)

    log_revenue, user_item_cross = export.compute_features(
        price=inputs['price'],
        quantity=inputs['quantity'],
        user_id=inputs['user_id'],
        item_id=inputs['item_id'],
    )

    expected_log_revenue, expected_cross = _expected_features(inputs)
    np.testing.assert_allclose(log_revenue, expected_log_revenue, rtol=1e-6)
    np.testing.assert_array_equal(user_item_cross, expected_cross)

  def test_export_model(self):
    export_path = self.create_tempdir().full_path
    export.export_model(export_path)
    self.assertTrue(
        os.path.isfile(os.path.join(export_path, 'orchestration.pb'))
    )

    runner = model_runner.ModelRunner(model_path=export_path)
    inputs = _make_inputs(batch_size=1)
    outputs = runner.signature('serving_default').run_with_dict_return(
        **inputs
    )

    log_revenue, user_item_cross = _expected_features(inputs)
    expected_pred = (
        export.WEIGHT * log_revenue
        + export.cross_weights()[user_item_cross % export.NUM_BUCKETS]
        + export.BIAS
    )
    self.assertIn('pred', outputs)
    np.testing.assert_allclose(outputs['pred'], expected_pred, rtol=1e-5)


if __name__ == '__main__':
  absltest.main()

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

"""Tests for block quantization utilities."""

import numpy as np
from .testing.pybase import googletest
from data.processor.python import quantize_utils


class QuantizeUtilsTest(googletest.TestCase):

  def test_asymmetric_block_quantize(self):
    # Dim = 6, BlockSize = 3 -> NumBlocks = 2
    weights = np.array(
        [[1.0, 2.0, 3.0, -1.0, 0.0, 1.0], [0.0, 10.0, 20.0, 5.0, 5.0, 5.0]],
        dtype=np.float32,
    )

    q_data, scales, offsets = quantize_utils.block_quantize(
        weights, block_size=3, symmetric=False
    )

    self.assertEqual(q_data.shape, (2, 6))
    self.assertEqual(scales.shape, (2, 2))
    self.assertEqual(offsets.shape, (2, 2))

    # Dequantize and check error
    for i in range(2):
      for j in range(6):
        b_idx = j // 3
        q_val = q_data[i, j]
        scale = scales[i, b_idx]
        offset = offsets[i, b_idx]
        dequant = q_val * scale + offset
        # Quantization error must be <= 0.5 * scale (half-LSB roundoff)
        self.assertLessEqual(abs(weights[i, j] - dequant), 0.51 * scale)

  def test_symmetric_block_quantize(self):
    weights = np.array(
        [[1.0, 2.0, 3.0, -1.0, 0.0, 1.0], [0.0, 10.0, 20.0, 5.0, 5.0, 5.0]],
        dtype=np.float32,
    )

    q_data, scales, offsets = quantize_utils.block_quantize(
        weights, block_size=3, symmetric=True
    )

    self.assertEqual(q_data.shape, (2, 6))
    self.assertEqual(scales.shape, (2, 2))
    self.assertEqual(offsets.shape, (0,))

    # Dequantize and check error
    for i in range(2):
      for j in range(6):
        b_idx = j // 3
        q_val = q_data[i, j]
        scale = scales[i, b_idx]
        dequant = q_val * scale
        self.assertLessEqual(abs(weights[i, j] - dequant), 0.51 * scale)

  def test_constant_weights(self):
    weights = np.zeros((3, 10), dtype=np.float32)
    q_data, scales, offsets = quantize_utils.block_quantize(
        weights, block_size=4, symmetric=False
    )
    self.assertEqual(q_data.shape, (3, 10))
    self.assertEqual(offsets.shape, (3, 3))
    np.testing.assert_allclose(scales, 1e-8)


if __name__ == "__main__":
  googletest.main()

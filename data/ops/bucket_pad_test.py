# Copyright 2026 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tests for BucketPadTransform and bucket_pad op."""

from absl.testing import absltest
import numpy as np
from data.ops import bucket_pad
from data.processor.python import composite_processor_pybind


class BucketPadOpTest(absltest.TestCase):

  def test_basic_bucket_pad(self):
    # Input shape: [2, 3] -> Buckets [4, 8, 16] -> Output: [2, 4]
    input_arr = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], dtype=np.float32)
    host_buffer = composite_processor_pybind.HostBuffer.from_dlpack(input_arr)

    output_buffer = bucket_pad.bucket_pad(
        host_buffer, target_dimensions=[4, 8, 16], axis=1
    )
    output_arr = np.from_dlpack(output_buffer)

    expected = np.array(
        [[1.0, 2.0, 3.0, 0.0], [4.0, 5.0, 6.0, 0.0]], dtype=np.float32
    )
    np.testing.assert_array_equal(output_arr, expected)

  def test_exact_bucket_match(self):
    # Input shape: [2, 4] matching bucket 4
    input_arr = np.arange(8, dtype=np.int32).reshape(2, 4)
    host_buffer = composite_processor_pybind.HostBuffer.from_dlpack(input_arr)

    output_buffer = bucket_pad.bucket_pad(
        host_buffer, target_dimensions=[4, 8, 16], axis=-1
    )
    output_arr = np.from_dlpack(output_buffer)

    np.testing.assert_array_equal(output_arr, input_arr)

  def test_custom_pad_value(self):
    input_arr = np.array([[7.0, 8.0]], dtype=np.float32)
    host_buffer = composite_processor_pybind.HostBuffer.from_dlpack(input_arr)

    output_buffer = bucket_pad.bucket_pad(
        host_buffer, target_dimensions=[5], axis=1, pad_value=-1.0
    )
    output_arr = np.from_dlpack(output_buffer)

    expected = np.array([[7.0, 8.0, -1.0, -1.0, -1.0]], dtype=np.float32)
    np.testing.assert_array_equal(output_arr, expected)

  def test_exceeds_max_bucket_raises(self):
    input_arr = np.zeros((1, 10), dtype=np.float32)
    host_buffer = composite_processor_pybind.HostBuffer.from_dlpack(input_arr)

    with self.assertRaises(Exception):
      bucket_pad.bucket_pad(host_buffer, target_dimensions=[4, 8], axis=1)


if __name__ == "__main__":
  absltest.main()

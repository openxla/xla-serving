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

"""Test numpy_utils through simple C++ methods wrapped with nanobind."""

import numpy as np
import tensorflow as tf

from .testing.pybase import googletest
from .testing.pybase import parameterized
from .third_party.neptune.runtime.python import numpy_utils_test_utils


class NumpyUtilsTest(parameterized.TestCase):

  def test_tensor_round_trip(self):
    """Tests the flow to convert numpy array to tensor, modify it, and convert it back."""
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    new_array = numpy_utils_test_utils.tensor_add_one(array)
    self.assertTrue(
        np.array_equal(new_array, np.array([[2, 3], [4, 5]], dtype=np.float32))
    )

  def test_host_buffer_round_trip(self):
    """Tests the flow to convert numpy array to host buffer, modify it, and convert it back."""
    array = np.array([[1, 2], [3, 4]], dtype=np.float32)
    new_array = numpy_utils_test_utils.host_buffer_times_two(array)
    self.assertTrue(
        np.array_equal(new_array, np.array([[2, 4], [6, 8]], dtype=np.float32))
    )

  def test_string_array_get_length(self):
    """Tests that numpy array with string type can be converted to tensor and fetched length."""
    array = np.array([[b'aa', b'b'], [b'ccc', b'd']], dtype=np.bytes_)
    new_array = numpy_utils_test_utils.string_array_get_length(array)
    self.assertTrue(np.array_equal(new_array, np.array([[2, 1], [3, 1]])))

  @parameterized.named_parameters(
      ('str', np.str_),
      ('StringDType', np.dtypes.StringDType()),  # type: ignore
  )
  def test_string_array(self, string_dtype: np.dtype):
    """Tests that numpy array with string type can be converted to tensor."""
    array = np.array([['aa', 'b'], ['ccc', 'd']], dtype=string_dtype)
    numpy_utils_test_utils.tensor_round_trip(array)

  @parameterized.named_parameters(
      ('bytes', np.bytes_),
      ('object', np.object_),
  )
  def test_bytes_array(self, bytes_dtype: np.dtype):
    """Tests that numpy array with bytes type can be converted to tensor."""
    array = np.array([[b'aa', b'b'], [b'ccc', b'd']], dtype=bytes_dtype)
    new_array = numpy_utils_test_utils.tensor_round_trip(array)
    self.assertTrue(np.array_equal(new_array, array))

  @parameterized.named_parameters(
      ('str', np.str_),
      ('object', np.object_),
      ('StringDType', np.dtypes.StringDType()),  # type: ignore
  )
  def test_unicode_string_array(self, string_dtype: np.dtype):
    """Tests that numpy array with unicode string type can be converted to tensor and back."""
    array = np.array(['\u2603', '\u03A9'], dtype=string_dtype)
    new_array = numpy_utils_test_utils.tensor_round_trip(array)
    for old, new in zip(array, new_array):
      # The returned string is in bytes.
      self.assertEqual(old, new.decode('utf-8'))

  def test_string_tensor_input(self):
    """Tests that a string TF tensor input is supported."""
    tensor = tf.constant(
        ['This is a string', 'This is another string'], dtype=tf.string
    )
    array = numpy_utils_test_utils.tensor_round_trip(tensor)
    self.assertEqual(array.dtype, np.object_)
    self.assertTrue(
        np.array_equal(
            array,
            np.array([b'This is a string', b'This is another string']),
        )
    )


if __name__ == '__main__':
  googletest.main()

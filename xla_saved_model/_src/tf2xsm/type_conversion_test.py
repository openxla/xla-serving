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

from absl.testing import absltest
import tensorflow as tf

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.tf2xsm import type_conversion


class TypeConversionTest(tf.test.TestCase):

  def test_tf_dtype_from_tf_dtype(self):
    self.assertEqual(type_conversion.tf_dtype_from(tf.int32), tf.int32)
    self.assertEqual(type_conversion.tf_dtype_from(tf.float32), tf.float32)

  def test_tf_dtype_from_shlo_dtype(self):
    self.assertEqual(
        type_conversion.tf_dtype_from(shlo_type.ShloDType.f32), tf.float32
    )
    self.assertEqual(
        type_conversion.tf_dtype_from(shlo_type.ShloDType.bf16), tf.bfloat16
    )
    self.assertEqual(
        type_conversion.tf_dtype_from(shlo_type.ShloDType.str), tf.string
    )
    self.assertEqual(
        type_conversion.tf_dtype_from(shlo_type.ShloDType.i32), tf.int32
    )

  def test_tf_type_spec_from_tf_type_spec(self):
    self.assertEqual(
        type_conversion.tf_type_spec_from(
            tf.TensorSpec(shape=(2, 3), dtype=tf.float32)
        ),
        tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
    )

    self.assertEqual(
        type_conversion.tf_type_spec_from(
            tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32)
        ),
        tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
    )

  def test_tf_type_spec_from_shlo_tensor_spec(self):
    self.assertEqual(
        type_conversion.tf_type_spec_from(
            shlo_type.ShloTensorSpec(
                shape=(2, 3), dtype=shlo_type.ShloDType.f32
            )
        ),
        tf.TensorSpec(shape=(2, 3), dtype=tf.float32),
    )

  def test_tf_type_spec_from_shlo_tensor_spec_with_name(self):
    self.assertEqual(
        type_conversion.tf_type_spec_from(
            shlo_type.ShloTensorSpec(
                shape=(1,), dtype=shlo_type.ShloDType.i32, name="foo"
            )
        ),
        tf.TensorSpec(shape=(1,), dtype=tf.int32, name="foo"),
    )

  def test_find_ragged_indices(self):
    input_signature = {
        "x": tf.TensorSpec(shape=(None, 10), dtype=tf.float32),
        "y": tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
        "z": tf.TensorSpec(shape=(None,), dtype=tf.float32),
        "sparse": tf.SparseTensorSpec(shape=(None, 5), dtype=tf.int32),
        "w": tf.TensorSpec(shape=(), dtype=tf.int32),
    }
    # tf.nest.flatten sorts the keys by default.
    # tf saved model also sorts the signature in the same order.
    # Sorted keys: sparse, w, x, y, z
    # sparse -> flat indices 0, 1, 2
    # w -> flat index 3
    # x -> flat index 4
    # y -> flat indices 5, 6
    # z -> flat index 7
    self.assertEqual(
        type_conversion.find_ragged_inputs_indices_if_any(input_signature),
        [(5, 6)],
    )


if __name__ == "__main__":
  absltest.main()

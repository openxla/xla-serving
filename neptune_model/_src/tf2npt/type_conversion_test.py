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

from neptune_model._src.core import shlo_type
from neptune_model._src.tf2npt import type_conversion


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

  def test_shlo_dtype_from_shlo_dtype(self):
    self.assertEqual(
        type_conversion.shlo_dtype_from(shlo_type.ShloDType.f32),
        shlo_type.ShloDType.f32,
    )

  def test_shlo_dtype_from_tf_dtype(self):
    self.assertEqual(
        type_conversion.shlo_dtype_from(tf.float32), shlo_type.ShloDType.f32
    )
    self.assertEqual(
        type_conversion.shlo_dtype_from(tf.bfloat16), shlo_type.ShloDType.bf16
    )
    self.assertEqual(
        type_conversion.shlo_dtype_from(tf.string), shlo_type.ShloDType.str
    )
    self.assertEqual(
        type_conversion.shlo_dtype_from(tf.int64), shlo_type.ShloDType.i64
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

  def test_shlo_tensor_spec_from_shlo_tensor_spec(self):
    self.assertEqual(
        type_conversion.shlo_tensor_spec_from(
            shlo_type.ShloTensorSpec(
                shape=(2,), dtype=shlo_type.ShloDType.f32, name="t"
            )
        ),
        shlo_type.ShloTensorSpec(
            shape=(2,), dtype=shlo_type.ShloDType.f32, name="t"
        ),
    )

  def test_shlo_tensor_spec_from_tf_tensor_spec(self):
    tf_spec = tf.TensorSpec(shape=(2, 3), dtype=tf.float32, name="t")
    self.assertEqual(
        type_conversion.shlo_tensor_spec_from(tf_spec),
        shlo_type.ShloTensorSpec(
            shape=(2, 3), dtype=shlo_type.ShloDType.f32, name="t"
        ),
    )

  def test_shlo_tensor_spec_from_tf_tensor_spec_unknown_rank(self):
    self.assertEqual(
        type_conversion.shlo_tensor_spec_from(
            tf.TensorSpec(shape=None, dtype=tf.int32)
        ),
        shlo_type.ShloTensorSpec(shape=None, dtype=shlo_type.ShloDType.i32),
    )

  def test_shlo_tensor_spec_from_tf_ragged_tensor_spec(self):
    self.assertEqual(
        type_conversion.shlo_tensor_spec_from(
            tf.RaggedTensorSpec(shape=(None, None), dtype=tf.int32),
        ),
        (
            shlo_type.ShloTensorSpec(
                shape=(None,), dtype=shlo_type.ShloDType.i32
            ),
            shlo_type.ShloTensorSpec(
                shape=(None,), dtype=shlo_type.ShloDType.i64
            ),
        ),
    )


if __name__ == "__main__":
  absltest.main()

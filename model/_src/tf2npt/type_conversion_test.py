"""Tests for type_conversion."""

from absl.testing import absltest
import tensorflow as tf

from neptune_model._src.core import shlo_type
from neptune_model._src.tf2npt import type_conversion


class TypeConversionTest(tf.test.TestCase):

  def test_to_tf_dtype(self):
    self.assertEqual(
        type_conversion.to_tf_dtype(shlo_type.ShloDType.f32), tf.float32
    )
    self.assertEqual(
        type_conversion.to_tf_dtype(shlo_type.ShloDType.bf16), tf.bfloat16
    )
    self.assertEqual(
        type_conversion.to_tf_dtype(shlo_type.ShloDType.str), tf.string
    )

  def test_to_tf_spec(self):
    spec = shlo_type.ShloTensorSpec(shape=(2, 3), dtype=shlo_type.ShloDType.f32)
    tf_spec = type_conversion.to_tf_spec(spec)
    self.assertEqual(tf_spec.shape, (2, 3))
    self.assertEqual(tf_spec.dtype, tf.float32)

  def test_to_npt_dtype(self):
    self.assertEqual(
        type_conversion.to_npt_dtype(tf.float32), shlo_type.ShloDType.f32
    )
    self.assertEqual(
        type_conversion.to_npt_dtype(tf.bfloat16), shlo_type.ShloDType.bf16
    )
    self.assertEqual(
        type_conversion.to_npt_dtype(tf.string), shlo_type.ShloDType.str
    )

  def test_to_npt_spec(self):
    tf_spec = tf.TensorSpec(shape=(2, 3), dtype=tf.float32)
    npt_spec = type_conversion.to_npt_spec(tf_spec)
    self.assertEqual(npt_spec.shape, (2, 3))
    self.assertEqual(npt_spec.dtype, shlo_type.ShloDType.f32)


if __name__ == "__main__":
  absltest.main()

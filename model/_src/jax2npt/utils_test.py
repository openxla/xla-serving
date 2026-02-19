"""Tests for utils."""

from absl.testing import absltest
from absl.testing import parameterized
import jax
import numpy as np
from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.jax2npt import utils
from google3.third_party.tensorflow.compiler.xla import xla_data_pb2


class UtilsTest(parameterized.TestCase):

  @parameterized.parameters(
      (jax.float0, shlo_type.ShloDType.bool),
      (jax.numpy.bfloat16, shlo_type.ShloDType.bf16),
      (jax.numpy.bool_, shlo_type.ShloDType.bool),
      (jax.numpy.int4, shlo_type.ShloDType.i4),
      (jax.numpy.int8, shlo_type.ShloDType.i8),
      (jax.numpy.int16, shlo_type.ShloDType.i16),
      (jax.numpy.int32, shlo_type.ShloDType.i32),
      (jax.numpy.int64, shlo_type.ShloDType.i64),
      (jax.numpy.uint4, shlo_type.ShloDType.ui4),
      (jax.numpy.uint8, shlo_type.ShloDType.ui8),
      (jax.numpy.uint16, shlo_type.ShloDType.ui16),
      (jax.numpy.uint32, shlo_type.ShloDType.ui32),
      (jax.numpy.uint64, shlo_type.ShloDType.ui64),
      (jax.numpy.float16, shlo_type.ShloDType.f16),
      (jax.numpy.float32, shlo_type.ShloDType.f32),
      (jax.numpy.float64, shlo_type.ShloDType.f64),
      (jax.numpy.complex64, shlo_type.ShloDType.c64),
      (jax.numpy.complex128, shlo_type.ShloDType.c128),
  )
  def test_get_physical_dtype_returns_expected_shlo_dtype(
      self, jax_dtype, expected_shlo_dtype
  ):
    self.assertEqual(utils.get_physical_dtype(jax_dtype), expected_shlo_dtype)

  def test_aval_dtype(self):
    aval = jax.core.ShapedArray((1, 2), jax.numpy.float32)
    self.assertEqual(utils.aval_dtype(aval), jax.numpy.float32)

  def test_hlo_sharding_to_op_sharding_with_none(self):
    self.assertIsNone(utils.hlo_sharding_to_op_sharding(None))

  def test_hlo_sharding_to_op_sharding(self):
    if jax.device_count() < 8:
      raise ValueError('Test requires 8 devices.')
    mesh = jax.sharding.Mesh(
        np.array(jax.devices()).reshape((2, 4)), ('x', 'y')
    )
    spec = jax.sharding.PartitionSpec('x', 'y')
    named_sharding = jax.sharding.NamedSharding(mesh, spec)
    hlo_sharding = named_sharding._to_xla_hlo_sharding(2)

    op_sharding = utils.hlo_sharding_to_op_sharding(hlo_sharding)

    self.assertIsInstance(op_sharding, xla_data_pb2.OpSharding)
    self.assertEqual(
        op_sharding.SerializeToString(),
        hlo_sharding.to_proto().SerializeToString(),
    )


if __name__ == '__main__':
  absltest.main()

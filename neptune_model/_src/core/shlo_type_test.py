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

"""Unit tests for shlo_type.py."""

from absl.testing import absltest
from absl.testing import parameterized
import ml_dtypes
import numpy as np

from neptune_model._src.core import shlo_type
from neptune_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


class ShloTypeTest(parameterized.TestCase):

  def test_shlo_tensor_spec_initialization_defaults(self):
    spec = shlo_type.ShloTensorSpec(shape=(2, 4), dtype=shlo_type.ShloDType.f32)
    self.assertEqual(spec.shape, (2, 4))
    self.assertEqual(spec.dtype, shlo_type.ShloDType.f32)
    self.assertIsNone(spec.sharding)
    self.assertIsNone(spec.layout)
    self.assertIsNone(spec.name)
    self.assertIsNone(spec.default_val)

  def test_shlo_tensor_spec_shape(self):
    spec_from_list = shlo_type.ShloTensorSpec(
        shape=[3, 5], dtype=shlo_type.ShloDType.i32
    )
    self.assertEqual(spec_from_list.shape, (3, 5))
    self.assertIsInstance(spec_from_list.shape, tuple)

    spec_none_shape = shlo_type.ShloTensorSpec(
        shape=None, dtype=shlo_type.ShloDType.i32
    )
    self.assertIsNone(spec_none_shape.shape)

  def test_shlo_tensor_spec_default_val_propagation(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(2,), dtype=shlo_type.ShloDType.f32, default_val=10.0
    )
    self.assertEqual(spec.default_val, 10.0)

  def test_shlo_tensor_spec_sharding_and_layout_propagation(self):
    sharding = xla_data_pb2.OpSharding(
        type=xla_data_pb2.OpSharding.REPLICATED
    )
    layout = xla_data_pb2.LayoutProto(minor_to_major=[1, 0])
    spec = shlo_type.ShloTensorSpec(
        shape=(2, 3),
        dtype=shlo_type.ShloDType.f32,
        sharding=sharding,
        layout=layout,
        name="test_spec",
    )
    self.assertEqual(spec.sharding, sharding)
    self.assertEqual(spec.layout, layout)
    self.assertEqual(spec.name, "test_spec")

  @parameterized.parameters(
      (np.bool_, shlo_type.ShloDType.bool),
      (bool, shlo_type.ShloDType.bool),
      (np.int8, shlo_type.ShloDType.i8),
      (np.int16, shlo_type.ShloDType.i16),
      (np.int32, shlo_type.ShloDType.i32),
      (np.int64, shlo_type.ShloDType.i64),
      (int, shlo_type.ShloDType.i64),
      (np.uint8, shlo_type.ShloDType.ui8),
      (np.uint16, shlo_type.ShloDType.ui16),
      (np.uint32, shlo_type.ShloDType.ui32),
      (np.uint64, shlo_type.ShloDType.ui64),
      (np.float16, shlo_type.ShloDType.f16),
      (np.float32, shlo_type.ShloDType.f32),
      (np.float64, shlo_type.ShloDType.f64),
      (float, shlo_type.ShloDType.f64),
      (np.complex64, shlo_type.ShloDType.c64),
      (np.complex128, shlo_type.ShloDType.c128),
      (complex, shlo_type.ShloDType.c128),
      (ml_dtypes.bfloat16, shlo_type.ShloDType.bf16),
      (ml_dtypes.int4, shlo_type.ShloDType.i4),
      (ml_dtypes.uint4, shlo_type.ShloDType.ui4),
      (ml_dtypes.float8_e4m3b11fnuz, shlo_type.ShloDType.f8_e4m3b11fnuz),
      (ml_dtypes.float8_e4m3fn, shlo_type.ShloDType.f8_e4m3fn),
      (ml_dtypes.float8_e4m3fnuz, shlo_type.ShloDType.f8_e4m3fnuz),
      (ml_dtypes.float8_e5m2, shlo_type.ShloDType.f8_e5m2),
      (ml_dtypes.float8_e5m2fnuz, shlo_type.ShloDType.f8_e5m2fnuz),
  )
  def test_shlo_tensor_spec_dtype_normalization(
      self, input_dtype, expected_shlo_dtype
  ):
    spec = shlo_type.ShloTensorSpec(shape=(2,), dtype=input_dtype)
    self.assertEqual(spec.dtype, expected_shlo_dtype)

  @parameterized.parameters(
      ("str", shlo_type.ShloDType.str),
      (str, shlo_type.ShloDType.str),
      (np.dtype("U"), shlo_type.ShloDType.str),
      (np.dtype("S"), shlo_type.ShloDType.str),
      (np.dtype(object), shlo_type.ShloDType.str),
  )
  def test_shlo_tensor_spec_string_kinds(
      self, input_dtype, expected_shlo_dtype
  ):
    spec = shlo_type.ShloTensorSpec(shape=(2,), dtype=input_dtype)
    self.assertEqual(spec.dtype, expected_shlo_dtype)

  def test_shlo_tensor_spec_as_numpy_dtype_type_property(self):
    class MockTfDTypeProperty:
      as_numpy_dtype = np.int32

    spec = shlo_type.ShloTensorSpec(shape=(2, 3), dtype=MockTfDTypeProperty())
    self.assertEqual(spec.dtype, shlo_type.ShloDType.i32)

  def test_shlo_tensor_spec_as_numpy_dtype_callable_method(self):
    class MockCallableDType:

      def as_numpy_dtype(self):
        return np.dtype("float64")

    spec = shlo_type.ShloTensorSpec(shape=(2, 3), dtype=MockCallableDType())
    self.assertEqual(spec.dtype, shlo_type.ShloDType.f64)

  def test_shlo_dtype_exhaustiveness(self):
    self.assertEqual(
        set(shlo_type.ShloDType),
        set(shlo_type._SHLO_DTYPE_TO_NP_DTYPE.keys()),
    )

  def test_shlo_dtype_bidirectional_roundtrip(self):
    for dtype in shlo_type.ShloDType:
      np_dt = shlo_type.shlo_dtype_to_np_dtype(dtype)
      self.assertEqual(shlo_type.np_dtype_to_shlo_dtype(np_dt), dtype)

  @parameterized.parameters(
      (np.dtype("U"), shlo_type.ShloDType.str),
      (np.dtype("S"), shlo_type.ShloDType.str),
      (np.dtype(object), shlo_type.ShloDType.str),
  )
  def test_np_dtype_to_shlo_dtype_string_and_object(
      self, np_dtype_val, expected_shlo_dtype
  ):
    self.assertEqual(
        shlo_type.np_dtype_to_shlo_dtype(np_dtype_val), expected_shlo_dtype
    )

  def test_shlo_tensor_spec_with_placement(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(2, 3),
        dtype=shlo_type.ShloDType.f32,
        placement=shlo_type.Placement.HOST,
    )
    self.assertEqual(spec.placement, shlo_type.Placement.HOST)


if __name__ == "__main__":
  absltest.main()

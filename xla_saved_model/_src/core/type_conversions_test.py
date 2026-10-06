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

"""Unit tests for to_shlo_tensor_spec in type_conversions."""

import dataclasses
import types
from typing import Any, NamedTuple
from absl.testing import absltest
from absl.testing import parameterized
import jax
import jax.numpy as jnp
import numpy as np

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import type_conversions
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


@dataclasses.dataclass
class DuckTypedSpec:
  shape: Any
  dtype: Any
  name: str | None = None
  sharding: Any = None
  layout: Any = None
  default_val: Any = None


class UnknownRankShape:
  """Mimics tf.TensorShape(None): not None, but raises when iterated."""

  rank = None

  def __iter__(self):
    raise ValueError("Cannot iterate over a shape with unknown rank.")


class RaisingShape:
  """A shape-like object that raises ValueError on iteration."""

  def __iter__(self):
    raise ValueError("boom")


class RaisingDType:
  """Mimics tf.resource: `.as_numpy_dtype` raises KeyError."""

  @property
  def as_numpy_dtype(self):
    raise KeyError("no numpy equivalent")


@jax.tree_util.register_pytree_node_class
@dataclasses.dataclass
class CustomPyTreeNode:
  a: Any
  b: Any

  def tree_flatten(self):
    return (self.a, self.b), None

  @classmethod
  def tree_unflatten(cls, aux, children):
    del aux
    return cls(*children)


class NamedTupleContainer(NamedTuple):
  weights: jax.ShapeDtypeStruct
  biases: jax.ShapeDtypeStruct


class TypeConversionsTest(parameterized.TestCase):

  # DType conversions.
  @parameterized.named_parameters(
      ("bool", np.dtype("bool"), shlo_type.ShloDType.bool),
      ("int8", np.dtype("int8"), shlo_type.ShloDType.i8),
      ("int16", np.dtype("int16"), shlo_type.ShloDType.i16),
      ("int32", np.dtype("int32"), shlo_type.ShloDType.i32),
      ("int64", np.dtype("int64"), shlo_type.ShloDType.i64),
      ("uint8", np.dtype("uint8"), shlo_type.ShloDType.ui8),
      ("uint16", np.dtype("uint16"), shlo_type.ShloDType.ui16),
      ("uint32", np.dtype("uint32"), shlo_type.ShloDType.ui32),
      ("uint64", np.dtype("uint64"), shlo_type.ShloDType.ui64),
      ("float16", np.dtype("float16"), shlo_type.ShloDType.f16),
      ("float32", np.dtype("float32"), shlo_type.ShloDType.f32),
      ("float64", np.dtype("float64"), shlo_type.ShloDType.f64),
      ("complex64", np.dtype("complex64"), shlo_type.ShloDType.c64),
      ("complex128", np.dtype("complex128"), shlo_type.ShloDType.c128),
      ("bfloat16", jnp.bfloat16, shlo_type.ShloDType.bf16),
      ("int4", jnp.int4, shlo_type.ShloDType.i4),
      ("uint4", jnp.uint4, shlo_type.ShloDType.ui4),
      ("float8_e4m3fn", jnp.float8_e4m3fn, shlo_type.ShloDType.f8_e4m3fn),
      ("float8_e5m2", jnp.float8_e5m2, shlo_type.ShloDType.f8_e5m2),
      ("float8_e4m3fnuz", jnp.float8_e4m3fnuz, shlo_type.ShloDType.f8_e4m3fnuz),
      ("float8_e5m2fnuz", jnp.float8_e5m2fnuz, shlo_type.ShloDType.f8_e5m2fnuz),
      (
          "float8_e4m3b11fnuz",
          jnp.float8_e4m3b11fnuz,
          shlo_type.ShloDType.f8_e4m3b11fnuz,
      ),
      ("jax_float0", jax.float0, shlo_type.ShloDType.bool),
      ("numpy_str", np.dtype("str"), shlo_type.ShloDType.str),
      ("numpy_object", np.dtype("object"), shlo_type.ShloDType.str),
      ("str_literal", "str", shlo_type.ShloDType.str),
      ("shlo_dtype_identity", shlo_type.ShloDType.f32, shlo_type.ShloDType.f32),
      (
          "py_int",
          int,
          shlo_type.ShloDType.i64
          if np.dtype(int) == np.int64
          else shlo_type.ShloDType.i32,
      ),
      ("py_float", float, shlo_type.ShloDType.f64),
      ("py_bool", bool, shlo_type.ShloDType.bool),
      (
          "duck_property",
          types.SimpleNamespace(as_numpy_dtype=np.dtype("int32")),
          shlo_type.ShloDType.i32,
      ),
      (
          "duck_callable",
          types.SimpleNamespace(as_numpy_dtype=lambda: np.dtype("float64")),
          shlo_type.ShloDType.f64,
      ),
  )
  def test_to_shlo_tensor_spec_dtypes(self, input_dtype, expected_shlo_dtype):
    spec = type_conversions.to_shlo_tensor_spec(
        DuckTypedSpec(shape=(2, 3), dtype=input_dtype)
    )
    self.assertEqual(spec.dtype, expected_shlo_dtype)

  # Leaf spec, shape, and array type conversions.
  @parameterized.named_parameters(
      (
          "shape_dtype_struct",
          jax.ShapeDtypeStruct((2, 4), jnp.float32),
          shlo_type.ShloTensorSpec((2, 4), shlo_type.ShloDType.f32),
          {},
      ),
      (
          "scalar_shape_dtype_struct",
          jax.ShapeDtypeStruct((), jnp.float32),
          shlo_type.ShloTensorSpec((), shlo_type.ShloDType.f32),
          {},
      ),
      (
          "scalar_numpy_array",
          np.array(42.5, dtype=np.float32),
          shlo_type.ShloTensorSpec((), shlo_type.ShloDType.f32),
          {},
      ),
      (
          "numpy_array_2d",
          np.ones((3, 5), dtype=np.float64),
          shlo_type.ShloTensorSpec((3, 5), shlo_type.ShloDType.f64),
          {},
      ),
      (
          "leaf_with_name_override",
          DuckTypedSpec(shape=(10,), dtype=np.int32),
          shlo_type.ShloTensorSpec(
              (10,), shlo_type.ShloDType.i32, name="test_input"
          ),
          {"name": "test_input"},
      ),
      (
          "unspecified_none_shape",
          DuckTypedSpec(shape=None, dtype=np.float32),
          shlo_type.ShloTensorSpec(shape=None, dtype=shlo_type.ShloDType.f32),
          {},
      ),
      (
          "unknown_rank_shape",
          DuckTypedSpec(shape=UnknownRankShape(), dtype=np.float32),
          shlo_type.ShloTensorSpec(shape=None, dtype=shlo_type.ShloDType.f32),
          {},
      ),
      (
          "symbolic_and_numpy_integers",
          DuckTypedSpec(
              shape=("batch", None, np.int64(128), np.int32(64), True),
              dtype=np.int32,
          ),
          shlo_type.ShloTensorSpec(
              (None, None, 128, 64, None), shlo_type.ShloDType.i32
          ),
          {},
      ),
      (
          "negative_dynamic_dimensions",
          DuckTypedSpec(
              shape=(-1, np.int32(-1), 128),
              dtype=np.float32,
          ),
          shlo_type.ShloTensorSpec((None, None, 128), shlo_type.ShloDType.f32),
          {},
      ),
      (
          "idempotent_shlo_spec",
          shlo_type.ShloTensorSpec(
              (1, 2), shlo_type.ShloDType.i32, name="orig"
          ),
          shlo_type.ShloTensorSpec(
              (1, 2), shlo_type.ShloDType.i32, name="orig"
          ),
          {},
      ),
      (
          "update_existing_shlo_spec",
          shlo_type.ShloTensorSpec(
              (1, 2), shlo_type.ShloDType.i32, name="orig"
          ),
          shlo_type.ShloTensorSpec(
              (1, 2),
              shlo_type.ShloDType.i32,
              sharding=xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.REPLICATED
              ),
              name="new_name",
          ),
          {
              "name": "new_name",
              "sharding": xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.REPLICATED
              ),
          },
      ),
      (
          "sharding_override",
          DuckTypedSpec(
              shape=(4,),
              dtype=np.float32,
              sharding=xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.MANUAL
              ),
          ),
          shlo_type.ShloTensorSpec(
              (4,),
              shlo_type.ShloDType.f32,
              sharding=xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.REPLICATED
              ),
          ),
          {
              "sharding": xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.REPLICATED
              )
          },
      ),
      (
          "root_name_override",
          DuckTypedSpec(shape=(2, 3), dtype=np.float32, name="orig_name"),
          shlo_type.ShloTensorSpec(
              (2, 3), shlo_type.ShloDType.f32, name="override_name"
          ),
          {"name": "override_name"},
      ),
      (
          "propagates_default_val",
          DuckTypedSpec((2, 3), np.int32, default_val=[1, 2, 3]),
          shlo_type.ShloTensorSpec(
              (2, 3), shlo_type.ShloDType.i32, default_val=[1, 2, 3]
          ),
          {},
      ),
  )
  def test_to_shlo_tensor_spec_leaf_conversions(
      self, input_obj, expected_spec, kwargs
  ):
    self.assertEqual(
        type_conversions.to_shlo_tensor_spec(input_obj, **kwargs),
        expected_spec,
    )

  def test_to_shlo_tensor_spec_from_shaped_array(self):
    arr = jax.core.ShapedArray((10,), jnp.int32)
    spec = type_conversions.to_shlo_tensor_spec(arr, name="test_input")
    self.assertEqual(spec.shape, (10,))
    self.assertEqual(spec.dtype, shlo_type.ShloDType.i32)
    self.assertEqual(spec.name, "test_input")
    self.assertIsInstance(spec.sharding, xla_data_pb2.OpSharding)

  def test_to_shlo_tensor_spec_from_jax_array(self):
    arr = jnp.zeros((3, 5), dtype=jnp.bfloat16)
    spec = type_conversions.to_shlo_tensor_spec(arr)
    self.assertEqual(spec.shape, (3, 5))
    self.assertEqual(spec.dtype, shlo_type.ShloDType.bf16)
    self.assertIsInstance(spec.sharding, xla_data_pb2.OpSharding)

  # Layout propagation.
  @parameterized.named_parameters(
      (
          "proto",
          xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
          xla_data_pb2.LayoutProto(minor_to_major=[1, 0]),
      ),
      (
          "to_proto",
          types.SimpleNamespace(
              to_proto=lambda: xla_data_pb2.LayoutProto(minor_to_major=[0, 1])
          ),
          xla_data_pb2.LayoutProto(minor_to_major=[0, 1]),
      ),
      (
          "jax_layout",
          types.SimpleNamespace(
              major_to_minor=(0, 1),
              tiling=((2, 2),),
              sub_byte_element_size_in_bits=4,
          ),
          xla_data_pb2.LayoutProto(
              minor_to_major=[1, 0],
              tiles=[xla_data_pb2.TileProto(dimensions=[2, 2])],
              element_size_in_bits=4,
          ),
      ),
      ("invalid_fallback", "unsupported_layout", None),
  )
  def test_to_shlo_tensor_spec_layout_propagation(
      self, layout, expected_layout
  ):
    spec = type_conversions.to_shlo_tensor_spec(
        DuckTypedSpec(shape=(4, 4), dtype=np.float32, layout=layout)
    )
    self.assertEqual(spec.layout, expected_layout)

  # Sharding propagation.
  @parameterized.named_parameters(
      (
          "op_sharding",
          xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED),
          xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED),
      ),
      (
          "to_proto",
          types.SimpleNamespace(
              to_proto=lambda: xla_data_pb2.OpSharding(
                  type=xla_data_pb2.OpSharding.REPLICATED
              )
          ),
          xla_data_pb2.OpSharding(type=xla_data_pb2.OpSharding.REPLICATED),
      ),
      (
          "broken_sharding_fallback",
          types.SimpleNamespace(
              to_proto=lambda: (_ for _ in ()).throw(RuntimeError("Corrupt"))
          ),
          None,
      ),
  )
  def test_to_shlo_tensor_spec_sharding_propagation(
      self, sharding, expected_sharding
  ):
    spec = type_conversions.to_shlo_tensor_spec(
        DuckTypedSpec(shape=(4,), dtype=np.float32, sharding=sharding)
    )
    self.assertEqual(spec.sharding, expected_sharding)

  def test_to_shlo_tensor_spec_jax_sharding_with_ndim(self):
    class MockJaxShardingWithNdim:

      def __init__(self):
        self.recorded_ndim = None

      def _to_xla_hlo_sharding(self, ndim: int):
        self.recorded_ndim = ndim
        return types.SimpleNamespace(
            to_proto=lambda: xla_data_pb2.OpSharding(
                type=xla_data_pb2.OpSharding.REPLICATED
            )
        )

    mock = MockJaxShardingWithNdim()
    spec = type_conversions.to_shlo_tensor_spec(
        DuckTypedSpec(shape=(4, 8), dtype=np.float32, sharding=mock)
    )
    self.assertEqual(mock.recorded_ndim, 2)
    self.assertEqual(spec.sharding.type, xla_data_pb2.OpSharding.REPLICATED)

  def test_to_shlo_tensor_spec_real_jax_sharding(self):
    sharding = jax.sharding.SingleDeviceSharding(jax.devices()[0])
    spec = type_conversions.to_shlo_tensor_spec(
        DuckTypedSpec(shape=(4, 8), dtype=np.float32, sharding=sharding)
    )
    self.assertIsInstance(spec.sharding, xla_data_pb2.OpSharding)

  # PyTree containers, naming precedence, and hierarchy.
  def test_to_shlo_tensor_spec_pytree_naming_and_containers(self):
    tree = {
        "params": {
            "w": jax.ShapeDtypeStruct((4, 2), jnp.float32),
            "b": jax.ShapeDtypeStruct((2,), jnp.float32),
        },
        "inputs": [jax.ShapeDtypeStruct(("batch", 4), jnp.float32)],
        "tuple_data": (
            jax.ShapeDtypeStruct((8,), jnp.int32),
            DuckTypedSpec((8,), np.int32, name="explicit_leaf_name"),
        ),
        "named_tuple": NamedTupleContainer(
            weights=jax.ShapeDtypeStruct((16, 8), jnp.float32),
            biases=jax.ShapeDtypeStruct((8,), jnp.float32),
        ),
        "custom_node": CustomPyTreeNode(
            a=jax.ShapeDtypeStruct((1,), jnp.float32),
            b=jax.ShapeDtypeStruct((2,), jnp.float32),
        ),
        "none_value": None,
    }
    result = type_conversions.to_shlo_tensor_spec(tree, name_leaves=True)

    self.assertEqual(result["params"]["w"].name, "params.w")
    self.assertEqual(result["params"]["b"].name, "params.b")
    self.assertEqual(result["inputs"][0].name, "inputs.0")
    self.assertEqual(result["tuple_data"][0].name, "tuple_data.0")
    self.assertEqual(result["tuple_data"][1].name, "explicit_leaf_name")
    self.assertEqual(result["named_tuple"].weights.name, "named_tuple.weights")
    self.assertEqual(result["named_tuple"].biases.name, "named_tuple.biases")
    self.assertEqual(result["custom_node"].a.name, "custom_node.0")
    self.assertEqual(result["custom_node"].b.name, "custom_node.1")
    self.assertIsNone(result["none_value"])

    unnamed_result = type_conversions.to_shlo_tensor_spec(tree)
    self.assertIsNone(unnamed_result["params"]["w"].name)
    self.assertEqual(
        unnamed_result["tuple_data"][1].name, "explicit_leaf_name"
    )

  @parameterized.parameters(({},), ([],), ((),), (None,))
  def test_to_shlo_tensor_spec_empty_and_none_containers(self, container):
    self.assertEqual(type_conversions.to_shlo_tensor_spec(container), container)

  def test_to_shlo_tensor_spec_leaf_callable_name_ignored(self):
    leaf = types.SimpleNamespace(
        shape=(2,), dtype=np.float32, name=lambda: "method_name"
    )
    spec = type_conversions.to_shlo_tensor_spec(
        {"weight": leaf}, name_leaves=True
    )
    self.assertEqual(spec["weight"].name, "weight")

  def test_to_shlo_tensor_spec_hierarchical_name_prefixing(self):
    tree = {
        "encoder": {
            "w": DuckTypedSpec(shape=(2, 3), dtype=np.float32),
            "b": DuckTypedSpec(shape=(3,), dtype=np.float32),
        }
    }
    spec = type_conversions.to_shlo_tensor_spec(tree, name="model")
    self.assertEqual(spec["encoder"]["w"].name, "model.encoder.w")
    self.assertEqual(spec["encoder"]["b"].name, "model.encoder.b")

  # Negative and boundary error handling.
  @parameterized.named_parameters(
      (
          "non_iterable_shape",
          DuckTypedSpec(128, np.float32),
          TypeError,
          "raw_shape must be an iterable or None",
      ),
      (
          "string_shape",
          DuckTypedSpec("batch", np.float32),
          TypeError,
          "raw_shape must be an iterable of dimension sizes or None",
      ),
      (
          "bytes_shape",
          DuckTypedSpec(b"batch", np.float32),
          TypeError,
          "raw_shape must be an iterable of dimension sizes or None",
      ),
      (
          "missing_shape",
          types.SimpleNamespace(dtype=np.float32),
          TypeError,
          r"Cannot convert object .* missing \.shape",
      ),
      (
          "missing_dtype",
          types.SimpleNamespace(shape=(2, 3)),
          TypeError,
          r"Cannot convert object .* missing \.dtype",
      ),
      (
          "invalid_dtype_string",
          DuckTypedSpec((2,), "not_a_valid_dtype_string"),
          TypeError,
          r"Cannot convert 'not_a_valid_dtype_string'.*to ShloDType",
      ),
      (
          "unsupported_numpy_dtype",
          DuckTypedSpec((2,), np.dtype("datetime64[ns]")),
          TypeError,
          r"NumPy dtype .* cannot be mapped to ShloDType",
      ),
      (
          "int_primitive",
          12345,
          TypeError,
          r"Cannot convert object .* missing \.shape",
      ),
      (
          "str_primitive",
          "raw_string",
          TypeError,
          r"Cannot convert object .* missing \.shape",
      ),
      (
          "shape_raises_on_iteration",
          DuckTypedSpec(RaisingShape(), np.float32),
          TypeError,
          "raw_shape must be an iterable or None",
      ),
      (
          "dtype_without_numpy_equivalent",
          DuckTypedSpec((2,), RaisingDType()),
          TypeError,
          r"NumPy dtype .* cannot be mapped to ShloDType",
      ),
  )
  def test_to_shlo_tensor_spec_invalid_inputs_raise(
      self, input_obj, exc_type, regex
  ):
    with self.assertRaisesRegex(exc_type, regex):
      type_conversions.to_shlo_tensor_spec(input_obj)


if __name__ == "__main__":
  absltest.main()

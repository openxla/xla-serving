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

import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format
import jax

from neptune_model._src.core import shlo_type
from neptune_model._src.core import test_utils
from neptune_model._src.core import type_proto_utils
from neptune_model.protos import type_pb2
from neptune_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


class CustomTree:

  def __init__(self, x, y):
    self.x = x
    self.y = y


jax.tree_util.register_pytree_node(
    CustomTree,
    lambda tree: ((tree.x, tree.y), ("custom", "keys")),
    lambda node_data, children: CustomTree(*children),
)


class MyCustomClass:

  def __init__(self, x):
    self.x = x


class TypeProtoUtilsTest(test_utils.NptTestCase, parameterized.TestCase):

  @parameterized.parameters(
      (shlo_type.ShloDType.bool, type_pb2.DType.i1),
      (shlo_type.ShloDType.i8, type_pb2.DType.si8),
      (shlo_type.ShloDType.i16, type_pb2.DType.si16),
      (shlo_type.ShloDType.i32, type_pb2.DType.si32),
      (shlo_type.ShloDType.i64, type_pb2.DType.si64),
      (shlo_type.ShloDType.ui8, type_pb2.DType.ui8),
      (shlo_type.ShloDType.ui16, type_pb2.DType.ui16),
      (shlo_type.ShloDType.ui32, type_pb2.DType.ui32),
      (shlo_type.ShloDType.ui64, type_pb2.DType.ui64),
      (shlo_type.ShloDType.f16, type_pb2.DType.f16),
      (shlo_type.ShloDType.f32, type_pb2.DType.f32),
      (shlo_type.ShloDType.f64, type_pb2.DType.f64),
      (shlo_type.ShloDType.bf16, type_pb2.DType.bf16),
      (shlo_type.ShloDType.str, type_pb2.DType.str),
  )
  def test_dtype_conversion(self, shlo_dt, npt_dt):
    self.assertEqual(
        type_proto_utils.npt_dtype_from_shlo_dtype(shlo_dt), npt_dt
    )
    self.assertEqual(
        type_proto_utils.shlo_dtype_from_npt_dtype(npt_dt), shlo_dt
    )

  @parameterized.parameters(
      (
          (),
          text_format.Parse("shape_with_known_rank {}", type_pb2.Shape()),
      ),
      (
          (1, 2),
          text_format.Parse(
              textwrap.dedent("""\
                  shape_with_known_rank {
                    dimension_sizes { size: 1 }
                    dimension_sizes { size: 2 }
                  }"""),
              type_pb2.Shape(),
          ),
      ),
      (
          (None, 4),
          text_format.Parse(
              textwrap.dedent("""\
                  shape_with_known_rank {
                    dimension_sizes {}
                    dimension_sizes { size: 4 }
                  }"""),
              type_pb2.Shape(),
          ),
      ),
      (None, type_pb2.Shape()),
  )
  def test_shape_conversion(self, shlo_shp, npt_shp):
    self.assertEqual(
        type_proto_utils.npt_shape_from_shlo_shape(shlo_shp), npt_shp
    )
    self.assertEqual(
        type_proto_utils.shlo_shape_from_npt_shape(npt_shp), shlo_shp
    )

  def test_tensor_spec_conversion(self):
    sharding_str = textwrap.dedent("""\
        type: OTHER
        tile_assignment_dimensions: [2, 2]
        iota_reshape_dims: [4]
        iota_transpose_perm: [0]
    """)
    sharding = text_format.Parse(sharding_str, xla_data_pb2.OpSharding())
    layout_str = textwrap.dedent("""\
        minor_to_major: [1, 0]
    """)
    layout = text_format.Parse(layout_str, xla_data_pb2.LayoutProto())
    spec = shlo_type.ShloTensorSpec(
        shape=(4, 2),
        dtype=shlo_type.ShloDType.f32,
        sharding=sharding,
        layout=layout,
    )

    expected_proto = text_format.Parse(
        textwrap.dedent("""\
            shape {
              shape_with_known_rank {
                dimension_sizes { size: 4 }
                dimension_sizes { size: 2 }
              }
            }
            dtype: f32
        """),
        type_pb2.TensorType(),
    )
    expected_proto.sharding.CopyFrom(sharding)
    expected_proto.layout.CopyFrom(layout)

    tensor_type_proto = type_proto_utils.npt_tensor_type_from_shlo_tensor_spec(
        spec
    )
    self.assertEqual(tensor_type_proto, expected_proto)

    r_spec = type_proto_utils.shlo_tensor_spec_from_npt_tensor_type(
        tensor_type_proto
    )
    self.assertEqual(r_spec, spec)

  def test_tensor_spec_conversion_without_sharding_and_layout(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(4, 2), dtype=shlo_type.ShloDType.f32, sharding=None, layout=None
    )
    expected_proto = text_format.Parse(
        textwrap.dedent("""\
            shape {
              shape_with_known_rank {
                dimension_sizes { size: 4 }
                dimension_sizes { size: 2 }
              }
            }
            dtype: f32
        """),
        type_pb2.TensorType(),
    )

    tensor_type_proto = type_proto_utils.npt_tensor_type_from_shlo_tensor_spec(
        spec
    )
    self.assertEqual(tensor_type_proto, expected_proto)

    r_spec = type_proto_utils.shlo_tensor_spec_from_npt_tensor_type(
        tensor_type_proto
    )
    self.assertEqual(r_spec, spec)

  def test_pytree_conversion_none(self):
    tree = None
    expected = type_pb2.Type()
    expected.none.CopyFrom(type_pb2.NoneType())

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(tree)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, tree)

  def test_pytree_conversion_leaf(self):
    spec = shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.i32)
    expected = type_pb2.Type()
    expected.leaf.tensor_type.CopyFrom(
        type_proto_utils.npt_tensor_type_from_shlo_tensor_spec(spec)
    )

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, spec)

  def test_pytree_conversion_list(self):
    spec1 = shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.i32)
    spec2 = shlo_type.ShloTensorSpec(shape=(2,), dtype=shlo_type.ShloDType.f32)
    tree = [spec1, spec2]

    expected = type_pb2.Type()
    expected.list.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec1)
    )
    expected.list.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec2)
    )

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(tree)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, tree)

  def test_pytree_conversion_tuple(self):
    spec1 = shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.i32)
    spec2 = shlo_type.ShloTensorSpec(shape=(2,), dtype=shlo_type.ShloDType.f32)
    tree = (spec1, spec2)

    expected = type_pb2.Type()
    expected.tuple.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec1)
    )
    expected.tuple.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec2)
    )

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(tree)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, tree)

  def test_pytree_conversion_dict(self):
    spec1 = shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.i32)
    spec2 = shlo_type.ShloTensorSpec(shape=(2,), dtype=shlo_type.ShloDType.f32)
    tree = {"a": spec1, "b": spec2}

    expected = type_pb2.Type()
    expected.dict.string_to_type["a"].CopyFrom(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec1)
    )
    expected.dict.string_to_type["b"].CopyFrom(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec2)
    )

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(tree)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, tree)

  def test_pytree_conversion_custom_pytree(self):
    spec1 = shlo_type.ShloTensorSpec(shape=(1,), dtype=shlo_type.ShloDType.i32)
    spec2 = shlo_type.ShloTensorSpec(shape=(2,), dtype=shlo_type.ShloDType.f32)
    tree = CustomTree(spec1, spec2)

    expected = type_pb2.Type()
    expected.list.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec1)
    )
    expected.list.elements.append(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(spec2)
    )

    proto = type_proto_utils.npt_type_from_shlo_tensor_spec_tree(tree)
    self.assertEqual(proto, expected)

    r_tree = type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)
    self.assertEqual(r_tree, [spec1, spec2])

  @parameterized.named_parameters(
      ("int", 1, "Unsupported tree type: <class 'int'>"),
      ("str", "hello", "Unsupported tree type: <class 'str'>"),
      (
          "custom_class",
          MyCustomClass(1),
          "Unsupported tree type: <class '.*MyCustomClass'>",
      ),
  )
  def test_unsupported_leaf_type_raises_error(self, value, regex):
    with self.assertRaisesRegex(ValueError, regex):
      type_proto_utils.npt_type_from_shlo_tensor_spec_tree(value)

  def test_shlo_tensor_spec_tree_from_npt_type_unsupported(self):
    proto = type_pb2.Type()
    with self.assertRaisesRegex(ValueError, "Unsupported manifest type"):
      type_proto_utils.shlo_tensor_spec_tree_from_npt_type(proto)

  def test_to_function_signature_proto(self):
    in_spec = shlo_type.ShloTensorSpec(
        shape=(1,), dtype=shlo_type.ShloDType.i32
    )
    out_spec = shlo_type.ShloTensorSpec(
        shape=(2,), dtype=shlo_type.ShloDType.f32
    )

    proto = type_proto_utils.npt_function_signature_from_shlo_signatures(
        in_spec, out_spec
    )

    expected = type_pb2.FunctionSignature()
    expected.input.CopyFrom(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(in_spec)
    )
    expected.output.CopyFrom(
        type_proto_utils.npt_type_from_shlo_tensor_spec_tree(out_spec)
    )

    self.assertEqual(proto, expected)


if __name__ == "__main__":
  absltest.main()

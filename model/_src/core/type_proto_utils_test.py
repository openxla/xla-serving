import textwrap

from absl.testing import absltest
from absl.testing import parameterized
from google.protobuf import text_format

from neptune_model._src.core import shlo_type
from neptune_model._src.core import test_utils
from neptune_model._src.core import type_proto_utils
from neptune_model.protos import type_pb2
from neptune_model.xla import xla_data_pb2


class ShloTensorSpecToTensorTypeProtoTest(
    test_utils.ObmTestCase, parameterized.TestCase
):

  def test_converts_spec(self):
    sharding_str = textwrap.dedent("""
      type: OTHER
      tile_assignment_dimensions: [2, 2]
      iota_reshape_dims: [4]
      iota_transpose_perm: [0]
    """)
    sharding = text_format.Parse(sharding_str, xla_data_pb2.OpSharding())
    layout_str = textwrap.dedent("""
      minor_to_major: [1, 0]
    """)
    layout = text_format.Parse(layout_str, xla_data_pb2.LayoutProto())
    spec = shlo_type.ShloTensorSpec(
        shape=(4, 2),
        dtype=shlo_type.ShloDType.f32,
        sharding=sharding,
        layout=layout,
    )

    tensor_type_proto = type_proto_utils.to_tensor_type_proto(spec)

    expected_proto = text_format.Parse(
        textwrap.dedent("""
        shape: {
          shape_with_known_rank: {
            dimension_sizes: { size: 4 }
            dimension_sizes: { size: 2 }
          }
        }
        dtype: f32
        """),
        type_pb2.TensorType(),
    )
    expected_proto.sharding.CopyFrom(sharding)
    expected_proto.layout.CopyFrom(layout)
    self.assertEqual(tensor_type_proto, expected_proto)

  def test_converts_spec_without_sharding_and_layout(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(4, 2), dtype=shlo_type.ShloDType.f32, sharding=None, layout=None
    )

    tensor_type_proto = type_proto_utils.to_tensor_type_proto(spec)

    expected_proto = text_format.Parse(
        textwrap.dedent("""
        shape: {
          shape_with_known_rank: {
            dimension_sizes: {
              size: 4
            }
            dimension_sizes: {
              size: 2
            }
          }
        }
        dtype: f32
        """),
        type_pb2.TensorType(),
    )
    self.assertEqual(tensor_type_proto, expected_proto)


class ShloTensorSpecPyTreeToTypeProtoTest(
    test_utils.ObmTestCase, parameterized.TestCase
):

  class MyCustomClass:

    def __init__(self, x):
      self.x = x

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
      type_proto_utils.to_type_proto(value)  # pytype: disable=wrong-arg-types


if __name__ == "__main__":
  absltest.main()

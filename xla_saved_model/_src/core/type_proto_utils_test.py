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

from xla_saved_model.protos import type_pb2
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import test_utils
from xla_saved_model._src.core import type_proto_utils
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


class ShloTensorSpecToTensorTypeProtoTest(
    test_utils.XsmTestCase, parameterized.TestCase
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

  def test_placement_enum_values(self):
    tensor_proto = type_pb2.TensorType()
    tensor_proto.placement = type_pb2.DEVICE
    self.assertEqual(tensor_proto.placement, type_pb2.DEVICE)
    tensor_proto.placement = type_pb2.HOST
    self.assertEqual(tensor_proto.placement, type_pb2.HOST)
    tensor_proto.placement = type_pb2.PLACEMENT_UNSPECIFIED
    self.assertEqual(tensor_proto.placement, type_pb2.PLACEMENT_UNSPECIFIED)

  def test_to_tensor_type_proto_with_placement(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(4, 2),
        dtype=shlo_type.ShloDType.f32,
        placement=type_pb2.HOST,
    )
    tensor_proto = type_proto_utils.to_tensor_type_proto(spec)
    self.assertEqual(tensor_proto.placement, type_pb2.HOST)

  def test_placement_bidirectional_roundtrip(self):
    for placement in (type_pb2.HOST, type_pb2.DEVICE):
      spec = shlo_type.ShloTensorSpec(
          shape=(4, 2),
          dtype=shlo_type.ShloDType.f32,
          placement=placement,
      )
      tensor_proto = type_proto_utils.to_tensor_type_proto(spec)
      restored_spec = type_proto_utils._from_tensor_type_proto(tensor_proto)
      self.assertEqual(restored_spec.placement, placement)

  def test_host_transpose_bidirectional_roundtrip(self):
    spec = shlo_type.ShloTensorSpec(
        shape=(1, 224, 224, 3),
        dtype=shlo_type.ShloDType.f32,
        host_transpose=[0, 3, 1, 2],
    )
    self.assertEqual(spec.host_transpose, (0, 3, 1, 2))
    tensor_proto = type_proto_utils.to_tensor_type_proto(spec)
    self.assertSequenceEqual(tensor_proto.host_transpose, (0, 3, 1, 2))
    restored_spec = type_proto_utils._from_tensor_type_proto(tensor_proto)
    self.assertEqual(restored_spec.host_transpose, (0, 3, 1, 2))


class ShloTensorSpecPyTreeToTypeProtoTest(
    test_utils.XsmTestCase, parameterized.TestCase
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


class RoundtripBetweenShloShapeAndManifestShapeTest(parameterized.TestCase):

  @parameterized.named_parameters(
      (f"_{idx}", shlo_shape, manifest_shape_proto_text)
      for idx, (shlo_shape, manifest_shape_proto_text) in enumerate([
          (
              [1, 2, 3],
              """
              shape_with_known_rank {
                dimension_sizes {
                  size: 1
                }
                dimension_sizes {
                  size: 2
                }
                dimension_sizes {
                  size: 3
                }
              }
              """,
          ),
          (
              [],
              """
              shape_with_known_rank {}
              """,
          ),
          (
              (),
              """
              shape_with_known_rank {}
              """,
          ),
          (
              None,
              "",
          ),
      ])
  )
  def test_roundtrip(self, shlo_shape, shape_proto_text):
    # Converts to proto
    shape_proto = type_proto_utils.to_shape_proto(shlo_shape)
    expected_shape_proto = text_format.Parse(shape_proto_text, type_pb2.Shape())
    self.assertEqual(shape_proto, expected_shape_proto)

    # Converts back to shlo shape.
    parsed_shlo_shape = type_proto_utils._from_shape_proto(expected_shape_proto)
    if shlo_shape is None:
      self.assertIsNone(parsed_shlo_shape)
    else:
      self.assertEqual(parsed_shlo_shape, tuple(shlo_shape))


class ManifestTypeToShloTensorSpecPyTreeTest(test_utils.XsmTestCase):

  def test_manifest_type_to_shlo_tensor_spec_tree(self):
    type_proto_str = textwrap.dedent("""
      tuple {
        elements {
          tuple {
            elements {
              dict {
                string_to_type {
                  key: "a"
                  value {
                    leaf {
                      tensor_type {
                        shape {
                          shape_with_known_rank {
                            dimension_sizes {
                              size: 4
                            }
                            dimension_sizes {
                              size: 4
                            }
                          }
                        }
                        dtype: si4
                        sharding {
                          type: OTHER
                          tile_assignment_dimensions: 4
                          tile_assignment_dimensions: 1
                          tile_assignment_dimensions: 2
                          replicate_on_last_tile_dim: true
                          iota_reshape_dims: 8
                          iota_transpose_perm: 0
                        }
                      }
                    }
                  }
                }
                string_to_type {
                  key: "b"
                  value {
                    leaf {
                      tensor_type {
                        shape {
                          shape_with_known_rank {
                            dimension_sizes {
                              size: 4
                            }
                            dimension_sizes {
                              size: 16
                            }
                          }
                        }
                        dtype: f32
                        sharding {
                          type: OTHER
                          tile_assignment_dimensions: 4
                          tile_assignment_dimensions: 2
                          iota_reshape_dims: 8
                          iota_transpose_perm: 0
                        }
                      }
                    }
                  }
                }
              }
            }
            elements {
              leaf {
                tensor_type {
                  shape {
                    shape_with_known_rank {
                      dimension_sizes {
                        size: 8
                      }
                      dimension_sizes {
                        size: 4
                      }
                    }
                  }
                  dtype: f32
                  sharding {
                    type: OTHER
                    tile_assignment_dimensions: 4
                    tile_assignment_dimensions: 1
                    tile_assignment_dimensions: 2
                    replicate_on_last_tile_dim: true
                    iota_reshape_dims: 8
                    iota_transpose_perm: 0
                  }
                }
              }
            }
          }
        }
        elements {
          dict {
          }
        }
      }
    """)

    type_proto = text_format.Parse(type_proto_str, type_pb2.Type())

    shlo_tensor_spec_tree = type_proto_utils.from_type_proto(type_proto)

    a_sharding = shlo_type.Sharding()
    a_sharding.type = shlo_type.Sharding.OTHER
    a_sharding.tile_assignment_dimensions.extend([4, 1, 2])
    a_sharding.replicate_on_last_tile_dim = True
    a_sharding.iota_reshape_dims.append(8)
    a_sharding.iota_transpose_perm.append(0)

    b_sharding = shlo_type.Sharding()
    b_sharding.type = shlo_type.Sharding.OTHER
    b_sharding.tile_assignment_dimensions.extend([4, 2])
    b_sharding.iota_reshape_dims.append(8)
    b_sharding.iota_transpose_perm.append(0)

    yet_another_sharding = shlo_type.Sharding()
    yet_another_sharding.type = shlo_type.Sharding.OTHER
    yet_another_sharding.tile_assignment_dimensions.extend([4, 1, 2])
    yet_another_sharding.replicate_on_last_tile_dim = True
    yet_another_sharding.iota_reshape_dims.append(8)
    yet_another_sharding.iota_transpose_perm.append(0)

    expected_shlo_tensor_spec_tree = (
        (
            {
                "a": shlo_type.ShloTensorSpec(
                    shape=(4, 4),
                    dtype=shlo_type.ShloDType.i4,
                    sharding=a_sharding,
                ),
                "b": shlo_type.ShloTensorSpec(
                    shape=(4, 16),
                    dtype=shlo_type.ShloDType.f32,
                    sharding=b_sharding,
                ),
            },
            shlo_type.ShloTensorSpec(
                shape=(8, 4),
                dtype=shlo_type.ShloDType.f32,
                sharding=yet_another_sharding,
            ),
        ),
        {},
    )

    self.assertTreesEqual(shlo_tensor_spec_tree, expected_shlo_tensor_spec_tree)


if __name__ == "__main__":
  absltest.main()

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

"""Type conversion between Neptune manifest types and SHLO types."""

import jax
import jaxtyping

from neptune_model._src.core import shlo_type
from neptune_model.protos import type_pb2

_SHLO_DTYPE_TO_MANIFEST_DTYPE: dict[shlo_type.ShloDType, type_pb2.DType] = {
    shlo_type.ShloDType.bool: type_pb2.DType.i1,
    shlo_type.ShloDType.i4: type_pb2.DType.si4,
    shlo_type.ShloDType.i8: type_pb2.DType.si8,
    shlo_type.ShloDType.i16: type_pb2.DType.si16,
    shlo_type.ShloDType.i32: type_pb2.DType.si32,
    shlo_type.ShloDType.i64: type_pb2.DType.si64,
    shlo_type.ShloDType.ui4: type_pb2.DType.ui4,
    shlo_type.ShloDType.ui8: type_pb2.DType.ui8,
    shlo_type.ShloDType.ui16: type_pb2.DType.ui16,
    shlo_type.ShloDType.ui32: type_pb2.DType.ui32,
    shlo_type.ShloDType.ui64: type_pb2.DType.ui64,
    shlo_type.ShloDType.f16: type_pb2.DType.f16,
    shlo_type.ShloDType.f32: type_pb2.DType.f32,
    shlo_type.ShloDType.f64: type_pb2.DType.f64,
    shlo_type.ShloDType.bf16: type_pb2.DType.bf16,
    shlo_type.ShloDType.str: type_pb2.DType.str,
}

_MANIFEST_DTYPE_TO_SHLO_DTYPE = {
    v: k for k, v in _SHLO_DTYPE_TO_MANIFEST_DTYPE.items()
}


def npt_dtype_from_shlo_dtype(
    shlo_dtype: shlo_type.ShloDType,
) -> type_pb2.DType:
  return _SHLO_DTYPE_TO_MANIFEST_DTYPE[shlo_dtype]


def shlo_dtype_from_npt_dtype(
    manifest_dtype: type_pb2.DType,
) -> shlo_type.ShloDType:
  return _MANIFEST_DTYPE_TO_SHLO_DTYPE[manifest_dtype]


def npt_shape_from_shlo_shape(
    shape: shlo_type.ShloShape,
) -> type_pb2.Shape:
  """Converts `ShloShape` to Neptune manifest `Shape`."""
  shape_proto = type_pb2.Shape()
  if shape is not None:
    shape_proto.shape_with_known_rank.CopyFrom(type_pb2.ShapeWithKnownRank())
    for dim in shape:
      dim: shlo_type.ShloDimSize
      ds_proto = type_pb2.DimensionSize()
      if dim is not None:
        ds_proto.size = dim
      shape_proto.shape_with_known_rank.dimension_sizes.append(ds_proto)
  return shape_proto


def shlo_shape_from_npt_shape(
    shape_proto: type_pb2.Shape,
) -> shlo_type.ShloShape:
  """Converts manifest `Shape` to `ShloShape`."""
  if shape_proto.HasField("shape_with_known_rank"):
    return tuple(
        dim.size if dim.HasField("size") else None
        for dim in shape_proto.shape_with_known_rank.dimension_sizes
    )
  return None


def shlo_tensor_spec_from_npt_tensor_type(
    manifest_tensor_type: type_pb2.TensorType,
) -> shlo_type.ShloTensorSpec:
  """Converts manifest `TensorType` to `ShloTensorSpec`."""
  sharding = (
      manifest_tensor_type.sharding
      if manifest_tensor_type.HasField("sharding")
      else None
  )
  layout = (
      manifest_tensor_type.layout
      if manifest_tensor_type.HasField("layout")
      else None
  )
  return shlo_type.ShloTensorSpec(
      shape=shlo_shape_from_npt_shape(manifest_tensor_type.shape),
      dtype=shlo_dtype_from_npt_dtype(manifest_tensor_type.dtype),
      sharding=sharding,
      layout=layout,
  )


def npt_tensor_type_from_shlo_tensor_spec(
    spec: shlo_type.ShloTensorSpec,
) -> type_pb2.TensorType:
  """Converts ShloTensorSpec to type_proto TensorType."""
  tensor_type_proto = type_pb2.TensorType()
  tensor_type_proto.shape.CopyFrom(npt_shape_from_shlo_shape(spec.shape))
  tensor_type_proto.dtype = npt_dtype_from_shlo_dtype(spec.dtype)
  if spec.sharding:
    tensor_type_proto.sharding.CopyFrom(spec.sharding)
  if spec.layout:
    tensor_type_proto.layout.CopyFrom(spec.layout)
  return tensor_type_proto


def npt_type_from_shlo_tensor_spec_tree(
    tree: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> type_pb2.Type:
  """Recursively translates a PyTree of ShloTensorSpecs to Neptune manifest Type."""
  result = type_pb2.Type()

  match tree:
    case None:
      result.none.CopyFrom(type_pb2.NoneType())

    case shlo_type.ShloTensorSpec():
      result.leaf.tensor_type.CopyFrom(
          npt_tensor_type_from_shlo_tensor_spec(tree)
      )

    case list():
      result.list.CopyFrom(type_pb2.List())
      for x in tree:
        result.list.elements.append(npt_type_from_shlo_tensor_spec_tree(x))

    case tuple():
      result.tuple.CopyFrom(type_pb2.Tuple())
      for x in tree:
        result.tuple.elements.append(npt_type_from_shlo_tensor_spec_tree(x))

    case dict():
      tree: dict[str, jaxtyping.PyTree[shlo_type.ShloTensorSpec]]
      result.dict.CopyFrom(type_pb2.Dict())
      for key, value in tree.items():
        result.dict.string_to_type[key].CopyFrom(
            npt_type_from_shlo_tensor_spec_tree(value)
        )

    case _:
      leaves, tree_def = jax.tree_util.tree_flatten(tree)
      # If tree_def.node_data() is empty, it means `tree` is a leaf node.
      # Since we have handled all supported leaf types, this indicates an
      # unsupported leaf type.
      if not tree_def.node_data():
        raise ValueError(f"Unsupported tree type: {type(tree)}")

      result.list.CopyFrom(type_pb2.List())
      for x in leaves:
        result.list.elements.append(npt_type_from_shlo_tensor_spec_tree(x))

  return result


def shlo_tensor_spec_tree_from_npt_type(
    manifest_type: type_pb2.Type,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Recursively translates manifest `Type` to a PyTree of `ShloTensorSpec`s."""
  if manifest_type.HasField("none"):
    return None

  if manifest_type.HasField("list"):
    return [
        shlo_tensor_spec_tree_from_npt_type(element)
        for element in manifest_type.list.elements
    ]

  if manifest_type.HasField("tuple"):
    return tuple(
        shlo_tensor_spec_tree_from_npt_type(element)
        for element in manifest_type.tuple.elements
    )

  if manifest_type.HasField("dict"):
    return {
        key: shlo_tensor_spec_tree_from_npt_type(value)
        for key, value in manifest_type.dict.string_to_type.items()
    }

  if manifest_type.HasField("leaf"):
    if manifest_type.leaf.HasField("tensor_type"):
      return shlo_tensor_spec_from_npt_tensor_type(
          manifest_type.leaf.tensor_type
      )

  raise ValueError(f"Unsupported manifest type: {manifest_type}")


def npt_function_signature_from_shlo_signatures(
    input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    output_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> type_pb2.FunctionSignature:
  signature = type_pb2.FunctionSignature()
  signature.input.CopyFrom(npt_type_from_shlo_tensor_spec_tree(input_signature))
  signature.output.CopyFrom(
      npt_type_from_shlo_tensor_spec_tree(output_signature)
  )
  return signature

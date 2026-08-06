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

"""Some utilities for type.proto ."""

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


def to_dtype_proto(shlo_dtype: shlo_type.ShloDType) -> type_pb2.DType:
  return _SHLO_DTYPE_TO_MANIFEST_DTYPE[shlo_dtype]


_MANIFEST_DTYPE_TO_SHLO_DTYPE = {
    v: k for k, v in _SHLO_DTYPE_TO_MANIFEST_DTYPE.items()
}


def to_shlo_dtype(
    manifest_dtype: type_pb2.DType,
) -> shlo_type.ShloDType:
  """Converts manifest DType to ShloDType."""
  return _MANIFEST_DTYPE_TO_SHLO_DTYPE[manifest_dtype]


def to_shape_proto(
    shape: shlo_type.ShloShape,
) -> type_pb2.Shape:
  """Converts `ShloShape` to type_proto `Shape`."""
  shape_proto = type_pb2.Shape()
  if shape is not None:
    shape_proto.shape_with_known_rank.SetInParent()
    for dim in shape:
      dim: shlo_type.ShloDimSize
      ds_proto = type_pb2.DimensionSize()
      if dim is not None:
        ds_proto.size = dim
      shape_proto.shape_with_known_rank.dimension_sizes.append(ds_proto)
  return shape_proto


def to_type_proto(
    tree: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> type_pb2.Type:
  """Recursively translates a PyTree of ShloTensorSpecs to manifest Type."""
  result = type_pb2.Type()
  if isinstance(tree, shlo_type.ShloTensorSpec):
    result.leaf.tensor_type.CopyFrom(to_tensor_type_proto(tree))
  elif tree is None:
    result.none.SetInParent()
  # list/tuple/dict can be empty, but we still want to record the type, so we
  # need to call SetInParent() for them.
  elif isinstance(tree, list):
    result.list.SetInParent()
    for x in tree:
      result.list.elements.append(to_type_proto(x))
  elif isinstance(tree, tuple):
    result.tuple.SetInParent()
    for x in tree:
      result.tuple.elements.append(to_type_proto(x))
  elif isinstance(tree, dict):
    tree: dict[str, jaxtyping.PyTree[shlo_type.ShloTensorSpec]]
    result.dict.SetInParent()
    for key, value in tree.items():
      result.dict.string_to_type[key].CopyFrom(to_type_proto(value))
  else:
    # TODO: Now we flattern the tree as list, but the custom
    # pytree node sometime can be respresented as other structures, like dict,
    # we should use the PytreeDef to decide how to represent it.
    leaves, tree_def = jax.tree_util.tree_flatten(tree)
    # If tree_def.node_data() is empty, it means `tree` is a leaf node.
    # Since we have handled all supported leaf types (ShloTensorSpec and None)
    # above, this indicates an unsupported leaf type.
    if not tree_def.node_data():
      raise ValueError(f"Unsupported tree type: {type(tree)}")
    result.list.SetInParent()
    for x in leaves:
      result.list.elements.append(to_type_proto(x))
  return result


def to_function_signature_proto(
    input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    output_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
) -> type_pb2.FunctionSignature:
  signature = type_pb2.FunctionSignature()
  signature.input.CopyFrom(to_type_proto(input_signature))
  signature.output.CopyFrom(to_type_proto(output_signature))
  return signature


_SHLO_PLACEMENT_TO_MANIFEST_PLACEMENT: dict[
    shlo_type.OutputPlacement, type_pb2.OutputPlacement
] = {
    shlo_type.OutputPlacement.UNSPECIFIED: (
        type_pb2.OutputPlacement.OUTPUT_PLACEMENT_UNSPECIFIED
    ),
    shlo_type.OutputPlacement.HOST: type_pb2.OutputPlacement.HOST,
    shlo_type.OutputPlacement.DEVICE: type_pb2.OutputPlacement.DEVICE,
}


def to_placement_proto(
    placement: shlo_type.OutputPlacement,
) -> type_pb2.OutputPlacement:
  """Converts Shlo Placement enum to manifest OutputPlacement proto enum."""
  return _SHLO_PLACEMENT_TO_MANIFEST_PLACEMENT[placement]


_MANIFEST_PLACEMENT_TO_SHLO_PLACEMENT = {
    v: k for k, v in _SHLO_PLACEMENT_TO_MANIFEST_PLACEMENT.items()
}


def to_shlo_placement(
    manifest_placement: type_pb2.OutputPlacement,
) -> shlo_type.OutputPlacement:
  """Converts manifest OutputPlacement proto enum to Shlo Placement enum."""
  return _MANIFEST_PLACEMENT_TO_SHLO_PLACEMENT[manifest_placement]


def to_tensor_type_proto(
    spec: shlo_type.ShloTensorSpec,
) -> type_pb2.TensorType:
  """Converts ShloTensorSpec to type_proto TensorType."""
  tensor_type_proto = type_pb2.TensorType()
  tensor_type_proto.shape.CopyFrom(to_shape_proto(spec.shape))
  tensor_type_proto.dtype = to_dtype_proto(spec.dtype)
  if spec.sharding:
    tensor_type_proto.sharding.CopyFrom(spec.sharding)
  if spec.layout:
    tensor_type_proto.layout.CopyFrom(spec.layout)
  if (
      spec.placement is not None
      and spec.placement != shlo_type.OutputPlacement.UNSPECIFIED
  ):
    tensor_type_proto.placement = to_placement_proto(spec.placement)
  return tensor_type_proto


def _from_shape_proto(
    shape: type_pb2.Shape,
) -> shlo_type.ShloShape:
  """Converts manifest `Shape` to `ShloShape`."""
  if shape.HasField("shape_with_known_rank"):
    return tuple(
        dim.size if dim.HasField("size") else None
        for dim in shape.shape_with_known_rank.dimension_sizes
    )
  return None


def _from_tensor_type_proto(
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
  placement = None
  if (
      manifest_tensor_type.HasField("placement")
      and manifest_tensor_type.placement
      != type_pb2.OutputPlacement.OUTPUT_PLACEMENT_UNSPECIFIED
  ):
    placement = to_shlo_placement(manifest_tensor_type.placement)
  return shlo_type.ShloTensorSpec(
      shape=_from_shape_proto(manifest_tensor_type.shape),
      dtype=to_shlo_dtype(manifest_tensor_type.dtype),
      sharding=sharding,
      layout=layout,
      placement=placement,
  )


def from_type_proto(
    manifest_type: type_pb2.Type,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Recursively translates manifest `Type` to a PyTree of `ShloTensorSpec`s."""
  if manifest_type.HasField("leaf"):
    if manifest_type.leaf.HasField("tensor_type"):
      return _from_tensor_type_proto(manifest_type.leaf.tensor_type)
    elif manifest_type.leaf.HasField("token_type"):
      raise NotImplementedError("TokenType is not supported yet")
  elif manifest_type.HasField("none"):
    return None
  elif manifest_type.HasField("list"):
    return [from_type_proto(element) for element in manifest_type.list.elements]
  elif manifest_type.HasField("tuple"):
    return tuple(
        from_type_proto(element) for element in manifest_type.tuple.elements
    )
  elif manifest_type.HasField("dict"):
    return {
        key: from_type_proto(value)
        for key, value in manifest_type.dict.string_to_type.items()
    }
  else:
    raise ValueError(f"Unsupported manifest type: {manifest_type}")

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

"""Class `JaxSpecificInfo` and its companions."""

# pylint: disable=g-importing-member
from collections.abc import Iterable, Sequence
import dataclasses
from typing import Any, TypeVar

from absl import logging
import jax

# Somehow JAX requires this import to make `jax.export` available.
from jax import export  # pylint: disable=unused-import
from jax.experimental import layout as jax_layout
import jax.extend as jex
import jax.numpy as jnp
import jaxtyping
import numpy as np

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import type_pb2
from xla_saved_model.protos.jax_supplemental_pb2 import DimensionSizeRefinement
from xla_saved_model.protos.jax_supplemental_pb2 import DTypeRefinement
from xla_saved_model.protos.jax_supplemental_pb2 import ShapeDTypeRefinement
from xla_saved_model.protos.jax_supplemental_pb2 import ShapeDTypeRefinements
from xla_saved_model.protos.jax_supplemental_pb2 import ShapeRefinement
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.jax2xsm import constants
from xla_saved_model._src.jax2xsm import pytree_proto
from xla_saved_model.protos.xla import xla_data_pb2

T1 = TypeVar("T1")
T2 = TypeVar("T2")

try:
  # JAX v0.10.0 or newer
  Effect = jex.core.Effect
except AttributeError:
  # JAX v0.9.2 or older
  Effect = getattr(jax.core, "Effect", Any)  # pyrefly: ignore


def unzip2(
    xys: Iterable[tuple[T1, T2]],
) -> tuple[tuple[T1, ...], tuple[T2, ...]]:
  """Unzip a sequence of tuples into two separate tuples."""
  xs: list[T1] = []
  ys: list[T2] = []
  for x, y in xys:
    xs.append(x)
    ys.append(y)
  return tuple(xs), tuple(ys)


# This function is adapted from the non-public
# `jax._src.layout.Layout._to_xla_layout`. Since there are no plans to make the
# original API public, the logic is replicated here.
def _to_xla_layout_proto(
    layout: jax_layout.Layout,
    dtype: jnp.dtype,
) -> xla_data_pb2.LayoutProto:
  """Converts a `jax.layout.Layout` to an `xla_data_pb2.LayoutProto`."""
  if layout.tiling is None:
    layout_proto = xla_data_pb2.LayoutProto()
    layout_proto.minor_to_major.extend(layout.major_to_minor[::-1])
  else:
    if layout.sub_byte_element_size_in_bits != 0:
      sub_byte_size = layout.sub_byte_element_size_in_bits
    elif jnp.issubdtype(dtype, np.integer):
      sub_byte_size = jnp.iinfo(dtype).bits if jnp.iinfo(dtype).bits < 8 else 0
    else:
      sub_byte_size = 0
    layout_proto = xla_data_pb2.LayoutProto()
    layout_proto.minor_to_major.extend(layout.major_to_minor[::-1])
    layout_proto.tiles.extend(
        xla_data_pb2.TileProto(dimensions=tile) for tile in layout.tiling
    )
    layout_proto.element_size_in_bits = sub_byte_size
  return layout_proto


def _name_leaf(
    path: jax.tree_util.KeyPath, leaf: shlo_type.ShloTensorSpec
) -> shlo_type.ShloTensorSpec:
  """Assigns a name to a leaf node in-place based on its PyTree path.

  The name is a dot-separated string representation of the path keys, extracted
  from dictionary keys, sequence indices, or attribute names. This function is
  designed to be used as the transform function in
  `jax.tree_util.tree_map_with_path`.

  The `leaf` argument's `name` attribute is modified in-place.

  Args:
    path: A tuple of path elements (e.g., DictKey, SequenceKey, GetAttrKey)
      provided by `tree_map_with_path`, representing the path to the leaf in a
      PyTree.
    leaf: The leaf node to be named. It must have a `name` attribute that can be
      set.

  Returns:
    The leaf node with its `name` attribute set to the dot-separated path
    string.

  Raises:
    TypeError: If an unknown key type is encountered in the path.
  """
  path_str_parts = []
  for key in path:
    if isinstance(key, jax.tree_util.DictKey):
      path_str_parts.append(str(key.key))
    elif isinstance(key, jax.tree_util.SequenceKey):
      path_str_parts.append(str(key.idx))
    elif isinstance(key, jax.tree_util.GetAttrKey):
      path_str_parts.append(str(key.name))
    elif isinstance(key, jax.tree_util.FlattenedIndexKey):
      path_str_parts.append(str(key.key))
    else:
      raise TypeError(f"Unknown key type: {type(key)}")
  leaf.name = ".".join(path_str_parts)
  return leaf


def _serialize_effect(eff: Effect) -> str:
  """Serializes a JAX Effect to a string.

  Args:
    eff: The JAX Effect to be serialized.

  Returns:
    A string representation of the Effect.
  """
  # Adapted from JAX's `_serialize_effect`.
  try:
    eff_replica = eff.__class__()
  except Exception as exc:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary constructor to be serializable"
    ) from exc
  try:
    hash_eff = hash(eff)
    hash_eff_replica = hash(eff_replica)
  except Exception as exc:
    raise NotImplementedError(
        f"Effect {eff} must be hashable to be serializable"
    ) from exc
  if eff != eff_replica or hash_eff != hash_eff_replica:
    raise NotImplementedError(
        f"Effect {eff} must have a nullary class constructor that produces an "
        "equal effect object."
    )
  return str(eff.__class__)


def _serialize_disabled_safety_check(
    check: jax.export.DisabledSafetyCheck,
) -> jax_supplemental_pb2.DisabledSafetyCheck:
  """Serializes a JAX DisabledSafetyCheck to a proto.

  Args:
    check: The JAX DisabledSafetyCheck to be serialized.

  Returns:
    A proto representation of the DisabledSafetyCheck.
  """
  # Adapted from JAX's `_serialize_disabled_safety_check`.
  proto = jax_supplemental_pb2.DisabledSafetyCheck()
  custom_call_target_str = check.is_custom_call()
  if custom_call_target_str is not None:
    proto.custom_call.target_name = custom_call_target_str
  elif check == jax.export.DisabledSafetyCheck.platform():
    proto.platform.SetInParent()
  else:
    raise ValueError(f"Unrecognized DisabledSafetyCheck: {check}")
  return proto


ShapeDTypeRefinementPair = tuple[ShapeRefinement | None, DTypeRefinement | None]


def _is_useful_refinement_pair(refinement: ShapeDTypeRefinementPair) -> bool:
  shape, dtype = refinement
  return shape is not None or dtype is not None


_MIN_RATIO_TO_USE_REFINEMENT_LIST = 0.5


def _to_shape_dtype_refinements_proto(
    refinements: Sequence[ShapeDTypeRefinementPair] | None,
) -> ShapeDTypeRefinements | None:
  """Converts `ShapeDTypeRefinementPair`s to a `ShapeDTypeRefinements`."""
  if refinements is None:
    return None
  n_useful_refinements = 0
  for refinement in refinements:
    if _is_useful_refinement_pair(refinement):
      n_useful_refinements += 1
  n_refinements = len(refinements)
  if (
      n_refinements == 0
      or n_useful_refinements / n_refinements
      >= _MIN_RATIO_TO_USE_REFINEMENT_LIST
  ):
    return ShapeDTypeRefinements(
        list=jax_supplemental_pb2.ShapeDTypeRefinementList(
            refinements=(
                ShapeDTypeRefinement(shape=shape, dtype=dtype)
                for shape, dtype in refinements
            )
        )
    )
  else:
    return ShapeDTypeRefinements(
        map=jax_supplemental_pb2.ShapeDTypeRefinementMap(
            idx_to_refinement={
                idx: ShapeDTypeRefinement(shape=shape, dtype=dtype)
                for idx, (shape, dtype) in enumerate(refinements)
                if _is_useful_refinement_pair((shape, dtype))
            }
        )
    )


CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE: str = (
    constants.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE
)
CURRENT_JAX_SUPPLEMENTAL_VERSION: str = (
    constants.CURRENT_JAX_SUPPLEMENTAL_VERSION
)


def _to_pytree_def_proto(
    treedef: jax.tree_util.PyTreeDef | None,
    what: str,
) -> jax_supplemental_pb2.PyTreeDef | None:
  """Serializes a `PyTreeDef`, or returns None if it need not be recorded.

  The manifest already describes tuples, lists, dicts and `None` exactly, so a
  structure built only from those is not recorded: doing so would add bytes to
  every artifact to restate what the manifest says. A structure that does
  contain a custom node is recorded if every such node is registered for
  serialization, and skipped with a warning otherwise, which leaves the artifact
  exactly as it would have been before this field existed.

  Args:
    treedef: The structure to serialize, or None if unknown.
    what: Human readable description of `treedef`, used in the warning.

  Returns:
    The proto representation, or None if the structure is either already
    implied by the manifest or not serializable.
  """
  if treedef is None or not pytree_proto.contains_custom_node(treedef):
    return None
  try:
    return pytree_proto.to_proto(treedef)
  except pytree_proto.UnsupportedPyTreeDefError as e:
    logging.warning(
        "Not recording the PyTree structure of %s (%s) in the JAX"
        " supplemental: %s. Callers that rebuild the calling convention from"
        " the saved model will see the custom nodes flattened into lists.",
        what,
        treedef,
        e,
    )
    return None


@dataclasses.dataclass
class JaxSpecificInfo(shlo_function.ShloFunctionSupplementalInfo):
  """JAX-specific information.

  Attributes:
    name: the name of the function.
    input_spec_refinements: refinements of input specs.
    output_spec_refinements: refinements of output specs.
    nr_devices: number of devices.
    ordered_effects: ordered effects.
    unordered_effects: unordered effects.
    disabled_safety_checks: disabled safety checks.
    uses_shape_polymorphism: bool indicating if shape polymorphism is used.
    in_tree: the PyTree structure of the function's inputs, recorded so that
      custom nodes, which the manifest cannot represent, survive a round trip.
      Only recorded when the manifest is not already exact; see
      `_to_pytree_def_proto`.
    out_tree: the PyTree structure of the function's outputs. See `in_tree`.
  """

  name: str
  input_spec_refinements: Sequence[ShapeDTypeRefinementPair] | None
  output_spec_refinements: Sequence[ShapeDTypeRefinementPair] | None
  nr_devices: int

  ordered_effects: Sequence[Effect]
  unordered_effects: Sequence[Effect]
  disabled_safety_checks: Sequence[jax.export.DisabledSafetyCheck]

  uses_shape_polymorphism: bool

  in_tree: jax.tree_util.PyTreeDef | None = None
  out_tree: jax.tree_util.PyTreeDef | None = None

  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    """Serializes to an `UnstructuredDataWithExtName`."""
    jax_proto = jax_supplemental_pb2.Function(
        name=self.name,
        input_spec_refinements=_to_shape_dtype_refinements_proto(
            self.input_spec_refinements
        ),
        output_spec_refinements=_to_shape_dtype_refinements_proto(
            self.output_spec_refinements
        ),
        nr_devices=self.nr_devices,
        ordered_effects=map(_serialize_effect, self.ordered_effects),
        unordered_effects=map(_serialize_effect, self.unordered_effects),
        disabled_checks=map(
            _serialize_disabled_safety_check, self.disabled_safety_checks
        ),
        uses_shape_polymorphism=self.uses_shape_polymorphism,
        input_tree=_to_pytree_def_proto(
            self.in_tree, f"the inputs of `{self.name}`"
        ),
        output_tree=_to_pytree_def_proto(
            self.out_tree, f"the outputs of `{self.name}`"
        ),
    )
    return unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(
            inlined_bytes=jax_proto.SerializeToString(),
            mime_type=CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE,
            version=CURRENT_JAX_SUPPLEMENTAL_VERSION,
        ),
        ext_name="pb",
    )


def _to_shlo_shape_and_refinement(
    jax_shape: Sequence[Any],
) -> tuple[shlo_type.ShloShape, ShapeRefinement | None]:
  """Gets a `ShloShape` and a `ShapeRefinement` from a JAX shape.

  Args:
    jax_shape: a JAX shape.

  Returns:
    A `ShloShape` and an optional `ShapeRefinement`. The `ShapeRefinement` will
    be `None` if no refinement is needed (i.e. all the dimensions are
    integers).
  """
  shlo_dim_sizes: list[shlo_type.ShloDimSize] = []
  dim_refinements: list[str | None] = []
  for dim in jax_shape:
    if dim is None:
      shlo_dim = None
      dim_refinement = None
    else:
      try:
        shlo_dim = int(dim)
        dim_refinement = None
      except Exception:  # pylint: disable=broad-exception-caught
        shlo_dim = None
        dim_refinement = str(dim)
    shlo_dim_sizes.append(shlo_dim)
    dim_refinements.append(dim_refinement)
  if all(r is None for r in dim_refinements):
    shape_refinement = None
  else:
    shape_refinement = ShapeRefinement(
        dimension_sizes=(
            DimensionSizeRefinement(size=r) for r in dim_refinements
        ),
    )
  return shlo_dim_sizes, shape_refinement


# The StableHLO type to represent float0 (the value will always be zero).
#
# We picked bool because this is what JAX uses when it lowers float0
# to StableHLO.
_shlo_dtype_for_float0 = shlo_type.ShloDType.bool


def _to_shlo_dtype_and_refinement(
    jax_dtype: np.dtype[Any],
) -> tuple[shlo_type.ShloDType, DTypeRefinement | None]:
  """Gets a `ShloDType` and a `DTypeRefinement` from a JAX dtype."""
  if jax_dtype == jax.numpy.int4:
    return shlo_type.ShloDType.i4, None
  if jax_dtype == jax.numpy.uint4:
    return shlo_type.ShloDType.ui4, None
  if jax_dtype == jax.float0:
    return _shlo_dtype_for_float0, DTypeRefinement.f0
  if jax_dtype == jax.numpy.bfloat16:
    return shlo_type.ShloDType.bf16, None
  if not isinstance(jax_dtype, np.dtype):
    raise TypeError(
        f"jax_dtype must be an instance of np.dtype, but got {jax_dtype} of"
        f" type {type(jax_dtype)}"
    )
  return shlo_type.np_dtype_to_shlo_dtype(jax_dtype), None


def _to_shlo_tensor_spec_and_refinement(
    aval: jax.core.AbstractValue,
    sharding: Any,
    layout: jax_layout.Layout | None = None,
    placement: type_pb2.Placement = type_pb2.PLACEMENT_UNSPECIFIED,
) -> tuple[shlo_type.ShloTensorSpec, ShapeDTypeRefinementPair]:
  """Gets a `ShloTensorSpec` and a `ShapeDTypeRefinement` from a `ShapedArray`.

  Args:
    aval: a JAX `ShapedArray`.
    sharding: the sharding of the `ShapedArray`.
    layout: the layout of the tensor.
    placement: the placement enum of the tensor.

  Returns:
    A `ShloTensorSpec` and a `ShapeDTypeRefinement`.
  """
  assert isinstance(aval, jax.core.ShapedArray)
  shlo_shape, shape_refinement = _to_shlo_shape_and_refinement(aval.shape)
  shlo_dtype, dtype_refinement = _to_shlo_dtype_and_refinement(aval.dtype)
  xla_layout_proto = (
      _to_xla_layout_proto(layout, aval.dtype) if layout is not None else None
  )
  spec = shlo_type.ShloTensorSpec(
      shape=shlo_shape,
      dtype=shlo_dtype,
      sharding=sharding,
      layout=xla_layout_proto,
      placement=(
          placement if placement != type_pb2.PLACEMENT_UNSPECIFIED else None
      ),
  )
  return spec, (shape_refinement, dtype_refinement)


def _to_flat_shlo_specs_and_refinements(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
    placements: Sequence[type_pb2.Placement] | None = None,
) -> tuple[
    tuple[shlo_type.ShloTensorSpec, ...],
    tuple[ShapeDTypeRefinementPair, ...] | None,
]:
  """Converts a sequence of avals to a tuple of ShloTensorSpecs."""
  effective_placements = (
      placements
      if placements
      else (type_pb2.PLACEMENT_UNSPECIFIED,) * len(avals)
  )
  specs_and_refinements = tuple(
      map(
          _to_shlo_tensor_spec_and_refinement,
          avals,
          shardings,
          layouts if layouts is not None else [None] * len(avals),
          effective_placements,
      )
  )
  specs, refinements = unzip2(specs_and_refinements)
  if all(shape is None and dtype is None for shape, dtype in refinements):
    refinements = None
  return specs, refinements


def _to_shlo_spec_tree_and_refinement_tuple(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
    tree_def: jax.tree_util.PyTreeDef | None = None,
    placements: Sequence[type_pb2.Placement] | None = None,
    name_leaves: bool = False,
) -> tuple[jaxtyping.PyTree, tuple[ShapeDTypeRefinementPair, ...] | None]:
  """Converts a sequence of avals to a tree of ShloTensorSpecs."""
  flat, refinements = _to_flat_shlo_specs_and_refinements(
      avals, shardings, layouts, placements
  )
  if tree_def is None:
    flat: jaxtyping.PyTree  # a tuple is also a tree
    jax_tree = flat
  else:
    jax_tree = jax.tree_util.tree_unflatten(tree_def, flat)

    leaves = jax.tree_util.tree_leaves(jax_tree)
    for x in leaves:
      if not isinstance(x, shlo_type.ShloTensorSpec):
        raise ValueError(
            f"Leaf needs to be a ShloTensorSpec, but its type is: {type(x)}"
        )
    jax_tree: jaxtyping.PyTree
  if name_leaves:
    jax_tree = jax.tree_util.tree_map_with_path(_name_leaf, jax_tree)
  return jax_tree, refinements

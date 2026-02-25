"""Common utils shared between jax_to_polymorphic_function.py and main_lib.py."""

from collections.abc import Iterable, Sequence
from typing import Any, TypeVar

import jax
from jax import export as jax_export
from jax.experimental import layout as jax_layout
from jax.extend import sharding as jex_sharding
import jax.numpy as jnp
import jaxtyping
import numpy as np

from google3.third_party.neptune.model._src.core import shlo_type
from google3.third_party.neptune.model._src.jax2npt import typing
from google3.third_party.neptune.protos import manifest_pb2
from google3.third_party.tensorflow.compiler.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import


def get_physical_dtype(dtype) -> shlo_type.ShloDType | None:
  try:
    if dtype == jax.numpy.int4:
      return shlo_type.ShloDType.i4
    if dtype == jax.numpy.uint4:
      return shlo_type.ShloDType.ui4
    if dtype == jax.numpy.bfloat16:
      return shlo_type.ShloDType.bf16
    if dtype == jax.float0:
      return shlo_type.ShloDType.bool
    return shlo_type.np_dtype_to_shlo_dtype(_jax_physical_dtype(dtype))
  except TypeError:
    return None


def assert_not_in_jax_transformation():
  """Raises an error if executed within a JAX transformation."""
  arr = jax.lax.iota("int32", 0)
  if isinstance(arr, jax.core.Tracer):
    raise ValueError("convert must be used outside all JAX transformations.")


def make_jax_exported_creator(
    fun_jax: typing.ApplyFn,
    platforms: Sequence[manifest_pb2.Platform] | None = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
):
  """Calls jax_export.export and returns a callable that returns Exported."""
  if platforms:
    native_serialization_platforms = tuple(
        manifest_pb2.Platform.Name(p).lower() for p in platforms
    )
  else:
    native_serialization_platforms = None

  return jax_export.export(
      fun_jax,
      platforms=native_serialization_platforms,
      disabled_checks=native_serialization_disabled_checks,
  )


JaxDType = np.dtype[Any]


def aval_dtype(a) -> JaxDType:
  assert isinstance(a, jax.core.ShapedArray)
  return a.dtype


def _jax_physical_aval(aval: jax.core.ShapedArray) -> jax.core.ShapedArray:
  """Converts JAX avals from logical to physical, if relevant.

  JAX might have avals whose logical vs physical shape/dtype may
  differ, and only the physical view is expected to possibly
  relate to EM. EM impl rules should operate on the physical form.

  A JAX logical aval might even correspond, in principle, to several
  physical avals, but we don't support those here. Instead we assert
  there is only one and return it.

  Args:
    aval: The JAX abstract value to be converted.

  Returns:
    The physical representation of the input aval. If the logical and physical
    representations are the same, the original `aval` is returned.
  """
  # TODO: jax.core doesn't publish physical_aval. Figure out how to
  # access it (or alike). physical_aval = jax.core.physical_aval(aval)
  physical_aval = aval
  assert (
      len(physical_aval.shape) >= len(aval.shape)
      and physical_aval.shape[: len(aval.shape)] == aval.shape
  ), (physical_aval, aval)
  return physical_aval


def _jax_physical_dtype(dtype) -> JaxDType:
  # assuming () is a fine stand-in shape
  return _jax_physical_aval(jax.core.ShapedArray((), dtype)).dtype


def hlo_sharding_to_op_sharding(
    hlo_sharding: Any | None,
) -> xla_data_pb2.OpSharding | None:
  """Converts `HloSharding` to proto `OpSharding`.

  Args:
    hlo_sharding: an `HloSharding`.

  Returns:
    An xla `OpSharding` proto.
  """
  if hlo_sharding is None:
    return None
  output = xla_data_pb2.OpSharding()
  # Note: `to_proto(self) -> xla_extension.OpSharding` so we need to serialize
  # and then parse.
  # TODO: See if go/proto-generated-python#sharing-messages can
  # help us share messages between Python and C++.
  output.ParseFromString(
      jex_sharding.get_serialized_proto_from_hlo_sharding(hlo_sharding)
  )
  return output


T1 = TypeVar("T1")
T2 = TypeVar("T2")


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


def _to_shlo_shape(
    jax_shape: Sequence[Any],
) -> shlo_type.ShloShape:
  """Gets a `ShloShape` from a JAX shape."""
  shlo_dim_sizes: list[shlo_type.ShloDimSize] = []
  for dim in jax_shape:
    if dim is None:
      shlo_dim = None
    else:
      try:
        shlo_dim = int(dim)
      except Exception:
        shlo_dim = None
    shlo_dim_sizes.append(shlo_dim)
  return shlo_dim_sizes


# The StableHLO type to represent float0 (the value will always be zero).
#
# We picked bool because this is what JAX uses when it lowers float0
# to StableHLO.
_shlo_dtype_for_float0 = shlo_type.ShloDType.bool


def _to_shlo_dtype(
    jax_dtype: np.dtype[Any],
) -> shlo_type.ShloDType:
  if jax_dtype == jax.numpy.int4:
    return shlo_type.ShloDType.i4
  if jax_dtype == jax.numpy.uint4:
    return shlo_type.ShloDType.ui4
  if jax_dtype == jax.float0:
    return _shlo_dtype_for_float0
  if jax_dtype == jax.numpy.bfloat16:
    return shlo_type.ShloDType.bf16
  if not isinstance(jax_dtype, np.dtype):
    raise TypeError(
        f"jax_dtype must be an instance of np.dtype, but got {jax_dtype} of"
        f" type {type(jax_dtype)}"
    )
  return shlo_type.np_dtype_to_shlo_dtype(jax_dtype)


def _to_shlo_tensor_spec(
    aval: jax.core.AbstractValue,
    sharding: Any,
    layout: jax_layout.Layout | None = None,
) -> shlo_type.ShloTensorSpec:
  """Gets a `ShloTensorSpec` from a `ShapedArray`.

  Args:
    aval: a JAX `ShapedArray`.
    sharding: the sharding of the `ShapedArray`.
    layout: the layout of the tensor.

  Returns:
    A `ShloTensorSpec`.
  """
  assert isinstance(aval, jax.core.ShapedArray)
  shlo_shape = _to_shlo_shape(aval.shape)
  shlo_dtype = _to_shlo_dtype(aval.dtype)
  xla_layout_proto = (
      _to_xla_layout_proto(layout, aval.dtype) if layout is not None else None
  )
  spec = shlo_type.ShloTensorSpec(
      shape=shlo_shape,
      dtype=shlo_dtype,
      sharding=sharding,
      layout=xla_layout_proto,
  )
  return spec


def _to_flat_shlo_specs(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
) -> tuple[shlo_type.ShloTensorSpec, ...]:
  """Converts a sequence of avals to a tuple of ShloTensorSpecs."""
  specs = tuple(
      map(
          _to_shlo_tensor_spec,
          avals,
          shardings,
          layouts if layouts is not None else [None] * len(avals),
      )
  )
  return specs


def to_shlo_spec_tree(
    avals: Sequence[jax.core.AbstractValue],
    shardings: Sequence[Any],
    layouts: Sequence[jax_layout.Layout] | None = None,
    tree_def: jax.tree_util.PyTreeDef | None = None,
    name_leaves: bool = False,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Converts a sequence of avals to a tree of ShloTensorSpecs."""
  flat = _to_flat_shlo_specs(avals, shardings, layouts)
  if tree_def is None:
    flat: jaxtyping.PyTree[shlo_type.ShloTensorSpec]  # a tuple is also a tree
    jax_tree = flat
  else:
    jax_tree = jax.tree_util.tree_unflatten(tree_def, flat)

  if name_leaves:
    jax_tree = jax.tree_util.tree_map_with_path(_name_leaf, jax_tree)
  return jax_tree

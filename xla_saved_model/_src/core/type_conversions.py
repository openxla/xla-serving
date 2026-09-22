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

"""Type conversion utilities to Neptune ShloTensorSpec."""

from collections.abc import Sequence
import dataclasses
import functools
from typing import Any, TypeVar

import jax
import jaxtyping
import numpy as np

from xla_saved_model._src.core import shlo_type
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import

_ProtoT = TypeVar("_ProtoT")


def _extract_proto(obj: Any, proto_cls: type[_ProtoT]) -> _ProtoT | None:
  """Extracts or deserializes a Protobuf message from an object or .to_proto()."""
  if isinstance(obj, proto_cls):
    return obj
  if hasattr(obj, "to_proto"):
    try:
      val = obj.to_proto()
      if isinstance(val, proto_cls):
        return val
      if hasattr(val, "SerializeToString") and hasattr(proto_cls, "FromString"):
        return getattr(proto_cls, "FromString")(val.SerializeToString())
    except Exception:  # pylint: disable=broad-exception-caught
      pass
  return None


def _to_shlo_dtype(raw_dt: Any) -> shlo_type.ShloDType:
  """Converts a JAX, NumPy, TF, or Python dtype to ShloDType."""
  if raw_dt == jax.float0:
    return shlo_type.ShloDType.bool
  if isinstance(raw_dt, shlo_type.ShloDType):
    return raw_dt

  if hasattr(raw_dt, "as_numpy_dtype"):
    attr = getattr(raw_dt, "as_numpy_dtype")
    raw_dt = (
        attr()
        if callable(attr) and not isinstance(attr, (type, np.dtype))
        else attr
    )

  try:
    np_dt = np.dtype(raw_dt)
    return shlo_type.np_dtype_to_shlo_dtype(np_dt)
  except KeyError as e:
    raise TypeError(
        f"NumPy dtype {raw_dt!r} cannot be mapped to ShloDType."
    ) from e
  except (TypeError, ValueError) as e:
    raise TypeError(
        f"Cannot convert {raw_dt!r} (type {type(raw_dt)}) to ShloDType."
    ) from e


def _to_dim(d: Any) -> int | None:
  if isinstance(d, (int, np.integer)) and not isinstance(d, bool):
    if d < 0:
      return None
    return int(d)
  return None


def _to_shlo_shape(raw_shape: Sequence[Any] | None) -> shlo_type.ShloShape:
  """Converts a shape sequence to a tuple of (int | None)."""
  if raw_shape is None:
    return None
  if isinstance(raw_shape, (str, bytes)):
    raise TypeError("raw_shape must be an iterable of dimension sizes or None.")
  try:
    return tuple(_to_dim(d) for d in raw_shape)
  except TypeError as e:
    raise TypeError("raw_shape must be an iterable or None.") from e


def _to_op_sharding(
    hlo_sharding: Any | None,
    ndim: int | None = None,
) -> xla_data_pb2.OpSharding | None:
  """Converts an HloSharding or OpSharding to an xla OpSharding proto."""
  if hlo_sharding is None:
    return None

  if hasattr(hlo_sharding, "_to_xla_hlo_sharding") and ndim is not None:
    try:
      # pylint: disable=protected-access
      hlo_sharding = hlo_sharding._to_xla_hlo_sharding(ndim)
      # pylint: enable=protected-access
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  proto = _extract_proto(hlo_sharding, xla_data_pb2.OpSharding)
  if proto is not None:
    return proto

  try:
    return xla_data_pb2.OpSharding.FromString(
        jax.extend.sharding.get_serialized_proto_from_hlo_sharding(hlo_sharding)
    )
  except Exception:  # pylint: disable=broad-exception-caught
    return None


def _to_layout_proto(
    raw_layout: Any | None,
) -> xla_data_pb2.LayoutProto | None:
  """Converts a JAX Layout, LayoutProto, or duck-typed layout to an xla LayoutProto."""
  if raw_layout is None:
    return None

  proto = _extract_proto(raw_layout, xla_data_pb2.LayoutProto)
  if proto is not None:
    return proto

  if hasattr(raw_layout, "major_to_minor"):
    try:
      tiles = [
          xla_data_pb2.TileProto(dimensions=t)
          for t in getattr(raw_layout, "tiling", ()) or ()
      ]
      sub_byte = getattr(raw_layout, "sub_byte_element_size_in_bits", 0) or 0
      return xla_data_pb2.LayoutProto(
          minor_to_major=raw_layout.major_to_minor[::-1],
          tiles=tiles,
          element_size_in_bits=sub_byte,
      )
    except Exception:  # pylint: disable=broad-exception-caught
      pass

  return None


def _resolve_leaf_name(
    path: jax.tree_util.KeyPath,
    x: Any,
    name_override: str | None = None,
    name_leaves: bool = False,
) -> str | None:
  """Resolves the leaf name using explicit override, attribute, or KeyPath."""
  path_str = (
      jax.tree_util.keystr(path, simple=True, separator=".") if path else None
  )
  if name_override is not None:
    return f"{name_override}.{path_str}" if path_str else name_override
  if isinstance(getattr(x, "name", None), str):
    return x.name
  if name_leaves:
    return path_str
  return None


def _convert_leaf(
    path: jax.tree_util.KeyPath,
    x: Any,
    sharding_override: Any | None = None,
    name_override: str | None = None,
    name_leaves: bool = False,
) -> shlo_type.ShloTensorSpec | None:
  """Converts a single leaf object to ShloTensorSpec."""
  if x is None:
    return None

  if isinstance(x, shlo_type.ShloTensorSpec):
    name = name_override if name_override is not None else x.name
    if sharding_override is not None:
      ndim = len(x.shape) if x.shape is not None else None
      sharding = _to_op_sharding(sharding_override, ndim=ndim)
    else:
      sharding = x.sharding
    if name != x.name or sharding != x.sharding:
      return dataclasses.replace(x, name=name, sharding=sharding)
    return x

  if not hasattr(x, "shape") or not hasattr(x, "dtype"):
    missing = ".shape" if not hasattr(x, "shape") else ".dtype"
    raise TypeError(
        f"Cannot convert object of type {type(x)} to ShloTensorSpec: missing"
        f" {missing}"
    )

  shape = _to_shlo_shape(x.shape)
  ndim = len(shape) if shape is not None else getattr(x, "ndim", None)
  raw_sharding = (
      sharding_override
      if sharding_override is not None
      else getattr(x, "sharding", None)
  )

  return shlo_type.ShloTensorSpec(
      shape=shape,
      dtype=_to_shlo_dtype(x.dtype),
      sharding=_to_op_sharding(raw_sharding, ndim=ndim),
      layout=_to_layout_proto(getattr(x, "layout", None)),
      name=_resolve_leaf_name(path, x, name_override, name_leaves=name_leaves),
      default_val=getattr(x, "default_val", None),
  )


def to_shlo_tensor_spec(
    tree: Any,
    *,
    sharding: Any | None = None,
    name: str | None = None,
    name_leaves: bool = False,
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Converts any JAX spec, array, or PyTree thereof to ShloTensorSpec.

  Accepts:
    - jax.ShapeDtypeStruct
    - jax.core.ShapedArray / jax.core.AbstractValue
    - jax.Array / np.ndarray
    - Objects with .shape and .dtype (TensorSpec, TensorSpecWithDefault, etc.)
    - ShloTensorSpec (idempotent)
    - Any nested PyTree (dict, list, tuple) containing the above.

  Args:
    tree: A spec, array, or PyTree with tensor-like leaves.
    sharding: Optional sharding override.
    name: Optional name override.
    name_leaves: Whether to assign names to leaves based on their PyTree path
      when they don't have an explicit name.

  Returns:
    A PyTree with the same structure, with leaves converted to ShloTensorSpec.
  """
  convert_fn = functools.partial(
      _convert_leaf,
      sharding_override=sharding,
      name_override=name,
      name_leaves=name_leaves,
  )
  return jax.tree_util.tree_map_with_path(
      convert_fn,
      tree,
      is_leaf=lambda x: isinstance(x, shlo_type.ShloTensorSpec),
  )

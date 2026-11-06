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

"""The shape and dtype of a tensor in a neptune signature.

Unlike `jax.ShapeDtypeStruct`, a dimension may be unknown, which is None. A
negative size is never read as unknown: it is an error.
"""

from collections.abc import Iterable, Iterator
import dataclasses
from typing import Any

import jax
import numpy as np
import numpy.typing as npt

from data.processor.proto import transform_config_pb2


def _is_size(d: Any) -> bool:
  """Returns True iff `d` is a Python or NumPy int, and not a bool."""
  return isinstance(d, (int, np.integer)) and not isinstance(
      d, (bool, np.bool_)
  )


def to_partial_shape(shape: Iterable[Any]) -> tuple[int | None, ...]:
  """Returns the shape of an input signature leaf, with None for unknown dims.

  Accepts the shape of any signature leaf, such as a `jax.ShapeDtypeStruct`,
  whose dims may be symbolic.

  Args:
    shape: The dims. None, and any dim that isn't an int (e.g. a JAX symbolic
      dim), is unknown. A non-negative int is a static size.

  Returns:
    The dims, with None for each unknown one.

  Raises:
    ValueError: If a dim is a negative int. -1 never means unknown.
  """
  shape = tuple(shape)
  dims = []
  for i, d in enumerate(shape):
    if not _is_size(d):
      dims.append(None)
    elif d < 0:
      raise ValueError(
          f'Dimension {i} is {d} in shape {shape!r}. Use None for an'
          ' unknown dimension; -1 is not accepted.'
      )
    else:
      dims.append(int(d))
  return tuple(dims)


@dataclasses.dataclass(frozen=True, init=False)
class ShapeDtypeStruct:
  """The shape and dtype of a tensor, with None for an unknown dimension.

  Attributes:
    shape: A tuple with a non-negative int, or None for an unknown dimension,
      per dimension. Any sequence is accepted and stored as a tuple of Python
      ints and Nones.
    dtype: A `np.dtype`. Anything `np.dtype` accepts is converted, as
      `jax.ShapeDtypeStruct` does.
  """

  shape: tuple[int | None, ...]
  dtype: np.dtype

  def __init__(
      self,
      shape: Iterable[int | np.integer | None],
      dtype: npt.DTypeLike,
  ):
    if not isinstance(shape, Iterable) or isinstance(shape, (str, bytes)):
      raise TypeError(
          f'ShapeDtypeStruct shape must be a sequence, got {shape!r}'
      )
    original_shape = tuple(shape)
    dims = []
    for i, d in enumerate(original_shape):
      if d is None:
        dims.append(None)
      elif not _is_size(d):
        raise TypeError(
            f'ShapeDtypeStruct dimension {i} must be an int or None, got'
            f' {d!r} in shape {original_shape!r}'
        )
      elif d < 0:
        raise ValueError(
            f'ShapeDtypeStruct dimension {i} is {d} in shape'
            f' {original_shape!r}. Use None for an unknown dimension; -1 is not'
            ' accepted.'
        )
      else:
        dims.append(int(d))
    object.__setattr__(self, 'shape', tuple(dims))
    object.__setattr__(self, 'dtype', np.dtype(dtype))

  def to_jax(self) -> jax.ShapeDtypeStruct:
    """Returns the `jax.ShapeDtypeStruct`, which can't hold an unknown dim.

    Raises:
      ValueError: If a dimension is unknown.
    """
    if any(d is None for d in self.shape):
      raise ValueError(
          f'Cannot convert shape {self.shape} with an unknown dimension to a'
          ' jax.ShapeDtypeStruct.'
      )
    return jax.ShapeDtypeStruct(self.shape, self.dtype)


def _set_leaf_shape(
    leaf_value: transform_config_pb2.ShapeDtypeStructProto, shape: Any
) -> None:
  """Writes a signature leaf's shape, with an unset size for unknown dims."""
  leaf_value.shape.SetInParent()  # A scalar has no dims, but a shape.
  for d in to_partial_shape(shape):
    dim = leaf_value.shape.dims.add()
    if d is not None:
      dim.size = d


def _get_leaf_shape(
    leaf_value: transform_config_pb2.ShapeDtypeStructProto,
) -> tuple[int | None, ...]:
  """Reads a signature leaf's shape, with None for unknown dims."""
  return tuple(
      dim.size if dim.HasField('size') else None
      for dim in leaf_value.shape.dims
  )


def tree_to_proto(tree: Any) -> transform_config_pb2.PyTreeProto:
  """Serializes a PyTree of tensor specs into a `PyTreeProto`."""
  proto = transform_config_pb2.PyTreeProto()
  if tree is None:
    proto.type = transform_config_pb2.PyTreeProto.NodeType.NONE
  elif isinstance(tree, dict):
    proto.type = transform_config_pb2.PyTreeProto.NodeType.DICT
    for k, v in sorted(tree.items()):
      proto.dict_keys.append(k)
      proto.children.append(tree_to_proto(v))
  elif isinstance(tree, list):
    proto.type = transform_config_pb2.PyTreeProto.NodeType.LIST
    for v in tree:
      proto.children.append(tree_to_proto(v))
  elif isinstance(tree, tuple):
    proto.type = transform_config_pb2.PyTreeProto.NodeType.TUPLE
    for v in tree:
      proto.children.append(tree_to_proto(v))
  else:
    proto.type = transform_config_pb2.PyTreeProto.NodeType.LEAF
    if hasattr(tree, 'shape') and tree.shape is not None:
      _set_leaf_shape(proto.leaf_value, tree.shape)
    dtype = getattr(tree, 'dtype', None)
    if dtype is not None:
      if hasattr(dtype, 'name'):
        proto.leaf_value.dtype = str(dtype.name)
      else:
        proto.leaf_value.dtype = str(dtype)
  return proto


def proto_to_tree(
    proto: transform_config_pb2.PyTreeProto,
    leaf_iter: Iterator[Any] | None = None,
) -> Any:
  """Reconstructs a PyTree from `proto`, optionally consuming `leaf_iter`."""
  if proto.type == transform_config_pb2.PyTreeProto.NodeType.NONE:
    return None
  elif proto.type == transform_config_pb2.PyTreeProto.NodeType.DICT:
    return {
        k: proto_to_tree(c, leaf_iter)
        for k, c in zip(proto.dict_keys, proto.children)
    }
  elif proto.type == transform_config_pb2.PyTreeProto.NodeType.LIST:
    return [proto_to_tree(c, leaf_iter) for c in proto.children]
  elif proto.type == transform_config_pb2.PyTreeProto.NodeType.TUPLE:
    return tuple(proto_to_tree(c, leaf_iter) for c in proto.children)
  elif proto.type == transform_config_pb2.PyTreeProto.NodeType.LEAF:
    if leaf_iter is not None:
      return next(leaf_iter)
    return ShapeDtypeStruct(
        shape=_get_leaf_shape(proto.leaf_value), dtype=proto.leaf_value.dtype
    )
  else:
    raise ValueError(f'Unknown node type: {proto.type}')

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

"""Conversion between `jax.tree_util.PyTreeDef` and its proto representation.

The manifest describes a function's calling convention with a
framework-neutral `Type`, which cannot represent a custom PyTree node such as a
`flax.struct.dataclass`. Such a node is written to the manifest as a plain list
of its leaves, so a caller that rebuilds the calling convention from the
manifest alone produces a `list` where the function expects the custom node,
and the call is rejected even though every leaf matches.

This module serializes the structure into the JAX supplemental so it can be
restored exactly.

A custom node is identified in the proto by a registered name rather than by a
Python import path, for two reasons: loading a model must never be able to
cause an arbitrary module to be imported, and a name stays stable when code
moves. Types opt in through `register_pytree_node_serialization` below, which
mirrors `jax.export.register_pytree_node_serialization`. We keep our own
registry because JAX exposes its registration function but not the registries
it populates, and a serializer has to read them.
"""

from collections.abc import Sequence
import types
from typing import Any, NamedTuple, Protocol, TypeVar

import jax

from xla_saved_model.protos import jax_supplemental_pb2

_T = TypeVar("_T")

PyTreeDefProto = jax_supplemental_pb2.PyTreeDef
_Kind = jax_supplemental_pb2.PyTreeDef.Kind

# The node types the proto spells out directly. Everything else needs a
# registration to be serializable.
_KIND_BY_NODE_TYPE = {
    types.NoneType: _Kind.NONE,
    tuple: _Kind.TUPLE,
    list: _Kind.LIST,
    dict: _Kind.DICT,
}
_NODE_TYPE_BY_KIND = {kind: t for t, kind in _KIND_BY_NODE_TYPE.items()}

# `PyTreeDef.from_node_data_and_children` is bound as a static method, but the
# nanobind signature it is published with (and therefore the checked-in type
# stub) declares a leading `self` parameter that the callable does not actually
# take. Calling it directly makes a type checker bind `registry` to `self`,
# `node_data` to `registry`, and so on. Going through an untyped alias keeps
# this file checkable; the arguments are verified by the tests instead.
_FROM_NODE_DATA_AND_CHILDREN: Any = (
    jax.tree_util.PyTreeDef.from_node_data_and_children
)


class SerializeAuxData(Protocol):

  def __call__(self, aux_data: Any, /) -> bytes:
    """Serializes the auxiliary data of a PyTree node.

    The auxiliary data is the second element of the pair returned by the
    `flatten_func` registered with `jax.tree_util.register_pytree_node`.
    """


class DeserializeAuxData(Protocol):

  def __call__(self, serialized_aux_data: bytes, /) -> Any:
    """Inverse of `SerializeAuxData`."""


class _Registration(NamedTuple):
  serialized_name: str
  nodetype: type[Any]
  serialize_auxdata: SerializeAuxData
  deserialize_auxdata: DeserializeAuxData


_registry_by_node_type: dict[type[Any], _Registration] = {}
_registry_by_serialized_name: dict[str, _Registration] = {}


def register_pytree_node_serialization(
    nodetype: type[_T],
    *,
    serialized_name: str,
    serialize_auxdata: SerializeAuxData,
    deserialize_auxdata: DeserializeAuxData,
) -> type[_T]:
  """Registers a custom PyTree node type for serialization.

  Call this for any type that appears as a node in the inputs or outputs of a
  function being exported and that is not a tuple, list, dict or `None`.
  Without a registration the structure is simply not recorded, and the
  structure a reader derives from the manifest flattens the node into a list.

  The type must already be a PyTree node, that is
  `jax.tree_util.register_pytree_node` (or one of its wrappers, such as
  `jax.tree_util.register_dataclass` or `flax.struct.dataclass`) must have been
  called for it. Only the auxiliary data needs callbacks here; the node itself
  is rebuilt through JAX's own registry.

  Both the reading and the writing process must have performed the
  registration, with the same `serialized_name`.

  Args:
    nodetype: The type whose PyTree nodes are to be serialized. It is an error
      to register a type twice.
    serialized_name: The name recorded in the proto and used to look the
      registration up when reading. It is an error to register a name twice.
      Prefer a name that identifies the owning project, for example
      `"qwix.QArray"`.
    serialize_auxdata: Converts the node's auxiliary data to bytes.
    deserialize_auxdata: Inverse of `serialize_auxdata`.

  Returns:
    `nodetype`, so that this function can be used as a class decorator.

  Raises:
    ValueError: If `nodetype` or `serialized_name` is already registered.
  """
  if nodetype in _registry_by_node_type:
    raise ValueError(
        f"Duplicate serialization registration for type `{nodetype}`. The"
        " previous registration used the serialized_name"
        f" `{_registry_by_node_type[nodetype].serialized_name}`."
    )
  if serialized_name in _registry_by_serialized_name:
    raise ValueError(
        "Duplicate serialization registration for the serialized_name"
        f" `{serialized_name}`. The previous registration was for the type"
        f" `{_registry_by_serialized_name[serialized_name].nodetype}`."
    )
  if nodetype in _KIND_BY_NODE_TYPE:
    raise ValueError(
        f"`{nodetype}` is represented by the proto directly and must not be"
        " registered."
    )

  registration = _Registration(
      serialized_name=serialized_name,
      nodetype=nodetype,
      serialize_auxdata=serialize_auxdata,
      deserialize_auxdata=deserialize_auxdata,
  )
  _registry_by_node_type[nodetype] = registration
  _registry_by_serialized_name[serialized_name] = registration
  return nodetype


class UnsupportedPyTreeDefError(Exception):
  """Raised when a `PyTreeDef` cannot be represented in the proto.

  The expected cause is a custom node whose type was never registered with
  `register_pytree_node_serialization`. Writers are expected to treat this as
  "record nothing" rather than as a failure: the manifest still describes every
  leaf, so the artifact stays exactly as it would have been before this feature
  existed.
  """


def contains_custom_node(treedef: jax.tree_util.PyTreeDef) -> bool:
  """Returns whether `treedef` holds a node the manifest cannot represent.

  The manifest expresses tuples, lists, dicts and `None` faithfully, so for the
  overwhelming majority of signatures the structure derived from it is already
  exact and recording a second copy would only add bytes. Writers use this to
  decide whether the supplemental needs to carry the structure at all, which
  also keeps existing artifacts byte for byte unchanged.

  Args:
    treedef: The structure to inspect.

  Returns:
    True if any node in `treedef` is of a type other than tuple, list, dict or
    `None`.
  """
  node_data = treedef.node_data()
  if node_data is not None and node_data[0] not in _KIND_BY_NODE_TYPE:
    return True
  return any(contains_custom_node(child) for child in treedef.children())


def to_proto(treedef: jax.tree_util.PyTreeDef) -> PyTreeDefProto:
  """Serializes a `PyTreeDef`.

  Args:
    treedef: The structure to serialize.

  Returns:
    The proto representation of `treedef`.

  Raises:
    UnsupportedPyTreeDefError: If `treedef` contains a custom node that is not
      registered for serialization, a dictionary with a non-string key, or a
      registered node whose `serialize_auxdata` does not return bytes.
  """
  proto = PyTreeDefProto()
  node_data = treedef.node_data()

  if node_data is None:
    proto.kind = _Kind.LEAF
    return proto

  node_type, aux_data = node_data
  kind = _KIND_BY_NODE_TYPE.get(node_type)

  if kind is None:
    registration = _registry_by_node_type.get(node_type)
    if registration is None:
      raise UnsupportedPyTreeDefError(
          f"PyTree node type `{node_type}` is not registered for"
          " serialization. Call"
          " `pytree_proto.register_pytree_node_serialization` for it."
      )
    serialized_auxdata = registration.serialize_auxdata(aux_data)
    if not isinstance(serialized_auxdata, (bytes, bytearray)):
      raise UnsupportedPyTreeDefError(
          f"The `serialize_auxdata` registered for `{node_type}` must return"
          f" bytes, but it returned a {type(serialized_auxdata)}."
      )
    proto.kind = _Kind.CUSTOM
    proto.custom_name = registration.serialized_name
    proto.custom_auxdata = bytes(serialized_auxdata)
  else:
    proto.kind = kind
    if kind == _Kind.DICT:
      for key in aux_data:
        if not isinstance(key, str):
          raise UnsupportedPyTreeDefError(
              "Only dictionaries with string keys can be serialized. Found the"
              f" key {key!r} of type {type(key)}."
          )
        proto.children_names.append(key)

  for child in treedef.children():
    proto.children.append(to_proto(child))

  if proto.kind == _Kind.DICT and len(proto.children_names) != len(
      proto.children
  ):
    raise UnsupportedPyTreeDefError(
        f"Dictionary has {len(proto.children_names)} keys but"
        f" {len(proto.children)} children."
    )

  return proto


def from_proto(proto: PyTreeDefProto) -> jax.tree_util.PyTreeDef:
  """Deserializes a `PyTreeDef`.

  Args:
    proto: The structure to deserialize.

  Returns:
    The deserialized structure.

  Raises:
    ValueError: If the proto is malformed, or refers to a node type that is not
      registered for deserialization in this process, or refers to a type that
      is registered here but is not a PyTree node in this process.
  """
  children = [from_proto(child) for child in proto.children]
  kind = proto.kind

  if kind == _Kind.LEAF:
    if children:
      raise ValueError(f"A LEAF cannot have {len(children)} children.")
    node_data = None
  elif kind == _Kind.CUSTOM:
    registration = _registry_by_serialized_name.get(proto.custom_name)
    if registration is None:
      raise ValueError(
          f"The PyTree node type `{proto.custom_name}` is not registered for"
          " deserialization in this process. Call"
          " `pytree_proto.register_pytree_node_serialization` for it before"
          " loading this model."
      )
    node_data = (
        registration.nodetype,
        registration.deserialize_auxdata(proto.custom_auxdata),
    )
  elif kind == _Kind.DICT:
    if len(proto.children_names) != len(children):
      raise ValueError(
          f"Dictionary has {len(proto.children_names)} keys but"
          f" {len(children)} children."
      )
    node_data = (dict, list(proto.children_names))
  elif kind in _NODE_TYPE_BY_KIND:
    if kind == _Kind.NONE and children:
      raise ValueError(f"A NONE cannot have {len(children)} children.")
    node_data = (_NODE_TYPE_BY_KIND[kind], None)
  else:
    raise ValueError(f"Unknown PyTreeDef kind: {kind}.")

  return _from_node_data_and_children(node_data, children)


def _from_node_data_and_children(
    node_data: tuple[type[Any], Any] | None,
    children: Sequence[jax.tree_util.PyTreeDef],
) -> jax.tree_util.PyTreeDef:
  """Builds a single `PyTreeDef` node, translating the failure mode."""
  try:
    return _FROM_NODE_DATA_AND_CHILDREN(
        jax.tree_util.default_registry, node_data, children
    )
  except Exception as e:
    # `from_node_data_and_children` resolves the node type through JAX's own
    # PyTree registry, which raises if the type was never registered as a
    # PyTree node in this process.
    raise ValueError(
        "Could not rebuild a PyTree node for"
        f" `{None if node_data is None else node_data[0]}`. The type must be"
        " registered with `jax.tree_util.register_pytree_node` before the"
        " model is loaded."
    ) from e

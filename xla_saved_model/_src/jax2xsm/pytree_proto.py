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
and the call is rejected even though every leaf matches. This module records
the structure in the JAX supplemental so it can be restored exactly.

A custom node is recorded under the module and qualified name of its type, a
name both sides derive without having to coordinate one, and its auxiliary
data through `auxdata`. Nothing has to be registered anywhere.

The name resembles an import path and is deliberately never used as one:
reading a model must not be able to cause a module to be imported. A reader
resolves the name against the modules it has already imported, so a reader that
wants its custom nodes rebuilt imports the packages defining them, the same way
it would to construct one. A name that resolves to nothing raises
`UnresolvedPyTreeNodeError`.
"""

import sys
import types
from typing import Any

import jax

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model._src.jax2xsm import auxdata

PyTreeDefProto = jax_supplemental_pb2.PyTreeDef
_Kind = jax_supplemental_pb2.PyTreeDef.Kind

# The node types the proto spells out directly. Everything else is written as
# a CUSTOM node, under a derived name.
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


def default_serialized_name(nodetype: type[Any]) -> str:
  """Returns the name `nodetype` is recorded under.

  Deriving the name lets a writer and a reader agree on it without
  coordinating a string. `__qualname__` rather than `__name__` keeps any
  enclosing classes, so two nested `_Inner` types do not collide.

  Args:
    nodetype: The type to name.

  Returns:
    The type's module and qualified name joined by a dot, for example
    `qwix._src.core.qarray.QArray`.
  """
  return f"{nodetype.__module__}.{nodetype.__qualname__}"


class UnsupportedPyTreeDefError(Exception):
  """Raised when a `PyTreeDef` cannot be represented in the proto.

  The expected cause is a custom node whose auxiliary data `auxdata.serialize`
  cannot express. Writers are expected to treat this as "record nothing" rather
  as a failure: the manifest still describes every leaf, so the artifact stays
  exactly as it would have been before this feature existed.
  """


class UnresolvedPyTreeNodeError(ValueError):
  """Raised when a recorded custom node names a type this process lacks.

  A process that cannot resolve the type cannot build the node, and so could
  not have called the function with one either. Failing here names the missing
  type at the point the model is read, rather than letting the caller discover
  it later as a structure mismatch that mentions no type at all. The remedy is
  to import the package defining the type before loading the model.

  It subclasses `ValueError` so that callers predating the distinct type still
  catch it.
  """


def _check_pytree_node_imported(serialized_name: str) -> type[Any]:
  """Returns the PyTree node type `serialized_name` names, if it is loaded.

  Only modules this process has already imported are consulted, so that
  reading a model can never cause an import: an artifact names the type it
  wants, and a process that has no reason to have that type simply does not
  find it. A module name and a qualified name can each contain dots, so every
  way of splitting the name is tried, deepest module first.

  Args:
    serialized_name: The name the writer recorded.

  Returns:
    The type.

  Raises:
    UnresolvedPyTreeNodeError: If no imported module defines a PyTree node of
      that name.
  """
  parts = serialized_name.split(".")
  for split in range(len(parts) - 1, 0, -1):
    module = sys.modules.get(".".join(parts[:split]))
    if module is None:
      continue
    found: Any = module
    for attribute in parts[split:]:
      found = getattr(found, attribute, None)
      if found is None:
        break
    # A name can be shared by something that is not the node type it was
    # recorded for, so take it only if it is in fact a PyTree node.
    if isinstance(found, type) and jax.tree_util.default_registry.is_node(
        found
    ):
      return found
  raise UnresolvedPyTreeNodeError(
      f"The PyTree node type recorded as `{serialized_name}` is not defined by"
      " any module this process has imported. Import the module that defines"
      " the type before loading the model."
  )


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
    UnsupportedPyTreeDefError: If `treedef` contains a dictionary with a
      non-string key, or a custom node whose auxiliary data `auxdata.serialize`
      cannot express.
  """
  proto = PyTreeDefProto()
  node_data = treedef.node_data()

  if node_data is None:
    proto.kind = _Kind.LEAF
    return proto

  node_type, aux_data = node_data
  kind = _KIND_BY_NODE_TYPE.get(node_type, _Kind.CUSTOM)
  proto.kind = kind

  if kind == _Kind.CUSTOM:
    proto.custom_name = default_serialized_name(node_type)
    try:
      proto.custom_auxdata = auxdata.serialize(aux_data)
    except Exception as e:
      # A node whose auxiliary data cannot be written down simply goes
      # unrecorded, and the artifact is the one this writer used to produce.
      raise UnsupportedPyTreeDefError(
          f"The auxiliary data {aux_data!r} of `{node_type}` cannot be"
          f" serialized: {e}"
      ) from e
  elif kind == _Kind.DICT:
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

  Any error JAX raises while rebuilding a node, for example because a custom
  node's unflatten function rejects the recorded auxiliary data, propagates
  unchanged.

  Args:
    proto: The structure to deserialize.

  Returns:
    The deserialized structure.

  Raises:
    ValueError: If the proto is malformed, or the recorded auxiliary data
      cannot be read back.
    UnresolvedPyTreeNodeError: If the proto refers to a custom node whose type
      no already imported module defines.
  """
  children = [from_proto(child) for child in proto.children]
  kind = proto.kind

  if kind == _Kind.LEAF:
    if children:
      raise ValueError(f"A LEAF cannot have {len(children)} children.")
    node_data = None
  elif kind == _Kind.CUSTOM:
    nodetype = _check_pytree_node_imported(proto.custom_name)
    try:
      aux_data = auxdata.deserialize(proto.custom_auxdata)
    except Exception as e:
      raise ValueError(
          "Could not read the auxiliary data recorded for"
          f" `{proto.custom_name}`: {e}"
      ) from e
    node_data = (nodetype, aux_data)
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

  return _FROM_NODE_DATA_AND_CHILDREN(
      jax.tree_util.default_registry, node_data, children
  )

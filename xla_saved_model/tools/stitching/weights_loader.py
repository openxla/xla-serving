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

"""Tools for loading the weights of an XLA SavedModel function back to JAX."""

from collections.abc import Sequence
import os
import pathlib
from typing import Any

import jax
import jaxtyping
import orbax.checkpoint as ocp

from xla_saved_model.protos import manifest_pb2
from xla_saved_model.protos import orchestration_pb2
from xla_saved_model._src.core import constants as core_constants
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.export import constants as export_constants

# Orbax export writes the same `Pipelines` message under its own type name.
_PIPELINES_MIME_TYPES = frozenset((
    core_constants.ORCHESTRATION_MULTI_PIPELINE_MIME_TYPE,
    "application/protobuf; type=orbax_export_orchestration.Pipelines",
))

# How many offending leaves an error message lists.
_MAX_REPORTED = 5


def _read_pipelines(
    model_dir: str, manifest: manifest_pb2.Manifest
) -> list[orchestration_pb2.Pipelines]:
  """Returns every `Pipelines` supplemental of the manifest."""
  return [
      orchestration_pb2.Pipelines.FromString(
          manifest_utils.read_unstructured_data_bytes(data, model_dir, manifest)
      )
      for data in manifest.supplemental_info.values()
      if data.mime_type in _PIPELINES_MIME_TYPES
  ]


def _manifest_weights_name(
    manifest: manifest_pb2.Manifest, function_name: str
) -> str | None:
  """Returns the weights name bound to `function_name` in `manifest`, if any."""
  obj = manifest.objects.get(function_name)
  if obj is None:
    return None
  match obj.WhichOneof("case"):
    case "function":
      if obj.function.HasField("weights_name") and obj.function.weights_name:
        return obj.function.weights_name
      return None
    case "poly_fn":
      if not obj.poly_fn.concrete_functions:
        return None
      names = [fn.weights_name for fn in obj.poly_fn.concrete_functions]
      if len(set(names)) > 1:
        raise ValueError(
            f"Concrete functions of '{function_name}' bind inconsistent"
            f" weights: {names}."
        )
      return names[0] or None
    case _:
      return None


# TODO: Delete once all models in use carry `Function.weights_name`
# in the manifest and the deprecated orchestration bindings are removed.
def _bound_weights_name(
    pipelines: Sequence[orchestration_pb2.Pipelines], function_name: str
) -> str | None:
  """Returns the weights name legacy pipelines bind to `function_name`, if any."""
  for multi_pipeline in pipelines:
    for pipeline in multi_pipeline.name_to_pipeline.values():
      for component in pipeline.components:
        if (
            component.function_name == function_name
            and component.HasField("weights_name")
            and component.weights_name
        ):
          return component.weights_name
      # Pipelines written before `components` existed.
      for model_function in pipeline.model_functions:
        if (
            model_function.model_function_name == function_name
            and model_function.HasField("weights_name")
            and model_function.weights_name
        ):
          return model_function.weights_name
  return None


def find_checkpoint_path(
    model_dir: pathlib.Path | str,
    function_name: str,
) -> str:
  """Returns the path of the Orbax checkpoint holding a function's weights.

  The weights are resolved from `Function.weights_name` in the manifest first,
  falling back to legacy orchestration pipeline bindings for older models.

  Args:
    model_dir: Directory where the XLA SavedModel was saved.
    function_name: Name of the function in the model manifest.

  Returns:
    The checkpoint path.

  Raises:
    ValueError: If no weights are bound to the function, or the weights do not
      name an Orbax checkpoint on the file system.
  """
  model_dir = str(model_dir)
  manifest = persistence_lib.load(model_dir)

  weights_name = _manifest_weights_name(manifest, function_name)
  if weights_name is None:
    pipelines = _read_pipelines(model_dir, manifest)
    weights_name = _bound_weights_name(pipelines, function_name)
  if weights_name is None:
    raise ValueError(f"No weights are bound to '{function_name}'.")

  obj = manifest.objects.get(weights_name)
  if obj is None or obj.WhichOneof("case") != "value":
    raise ValueError(f"The weights '{weights_name}' are not in the manifest.")
  data = obj.value.external.data
  if data.mime_type != export_constants.ORBAX_CHECKPOINT_MIME_TYPE:
    raise ValueError(
        f"The weights '{weights_name}' are not an Orbax checkpoint (MIME type"
        f" '{data.mime_type}')."
    )
  if data.WhichOneof("data") != "file_system_location":
    raise ValueError(
        f"The weights '{weights_name}' do not name a file system location."
    )
  path = manifest_utils.resolve_relative_path(
      data.file_system_location, manifest
  )
  if ".." in pathlib.PurePath(path).parts:
    raise ValueError(f"The checkpoint path cannot contain '..': {path}")
  # A model may reference a checkpoint outside of its own directory.
  return path if os.path.isabs(path) else os.path.join(model_dir, path)


def _key_name(entry: Any) -> str:
  """Returns the name of one step of a key path, whatever the node type."""
  match entry:
    case jax.tree_util.DictKey(key=key):
      return str(key)
    case jax.tree_util.GetAttrKey(name=name):
      return name
    case jax.tree_util.SequenceKey(idx=idx):
      return str(idx)
    case jax.tree_util.FlattenedIndexKey(key=key):
      return str(key)
    case _:
      return str(entry)


def _leaf_names(path: Sequence[Any]) -> tuple[str, ...]:
  return tuple(_key_name(entry) for entry in path)


def _shapes_agree(shape: Sequence[Any], expected: Sequence[Any]) -> bool:
  """Returns whether a concrete shape fits one that may be symbolic."""
  if len(shape) != len(expected):
    return False
  return all(
      not isinstance(want, int) or have == want
      for have, want in zip(shape, expected)
  )


def _fits(leaf: Any, want: Any) -> bool:
  return leaf.dtype == want.dtype and _shapes_agree(leaf.shape, want.shape)


def _unrecorded_node_prefix(path: Sequence[Any]) -> tuple[str, ...] | None:
  """Returns the path of the unrecorded custom node a leaf may belong to.

  A function whose recorded calling convention lost a custom node, e.g. because
  the node's auxiliary data could not be serialized, sees that node as nested
  lists. Its leaves then end in a run of sequence indices, which name nothing in
  a checkpoint saved from the real node. This returns the path without that
  run, or None if the path does not end in one or has nothing before it.
  """
  end = len(path)
  while end > 0 and isinstance(path[end - 1], jax.tree_util.SequenceKey):
    end -= 1
  if end == len(path) or end == 0:
    return None
  return _leaf_names(path[:end])


def _match_unrecorded_node(
    expected: Sequence[tuple[int, Any]],
    candidates: Sequence[tuple[tuple[str, ...], Any]],
) -> dict[int, tuple[str, ...]] | None:
  """Matches the leaves of a node seen as lists to the leaves saved for it.

  Names cannot be used, and neither can positions: the order a custom node
  flattens its fields in need not be the order of the names the checkpoint
  saved them under. Each expected leaf is matched to the only saved leaf of its
  shape and dtype instead, so an array can never be bound to the wrong field.

  Args:
    expected: `(index, abstract value)` of each expected leaf of the node.
    candidates: `(name, array)` of each unmatched saved leaf under the node.

  Returns:
    The name of the saved leaf for each expected index, or None if the leaves
    are not in one-to-one correspondence by shape and dtype.
  """
  if len(expected) != len(candidates):
    return None
  matches = {}
  for index, want in expected:
    fitting = [name for name, leaf in candidates if _fits(leaf, want)]
    if len(fitting) != 1:
      return None
    matches[index] = fitting[0]
  if len(set(matches.values())) != len(matches):
    return None
  return matches


def restructure_params(
    restored: jaxtyping.PyTree, expected: jaxtyping.PyTree
) -> jaxtyping.PyTree:
  """Rebuilds restored parameters with the structure a function expects.

  A checkpoint restored without a target comes back as nested containers keyed
  by name, even where the function was traced with custom nodes such as
  `qwix.QArray`. Leaves are matched by the names along their key paths, not by
  position, so a difference in flattening order between the two cannot bind an
  array to the wrong parameter.

  A function loaded from a model that could not record a custom node, e.g. a
  `qwix.WithAux` whose `HowToQuantize` cannot be serialized, expects that node
  as nested lists, whose indices name nothing in the checkpoint. The leaves of
  such a node are matched to the leaves saved under the node's path by shape
  and dtype instead, provided each expected leaf fits exactly one of them.

  Args:
    restored: The restored parameters.
    expected: The parameters the function expects, as abstract values.

  Returns:
    The restored arrays, arranged in the expected structure.

  Raises:
    ValueError: If a leaf is missing, has the wrong shape or dtype, or the
      checkpoint holds leaves the function does not take.
  """
  by_name = {
      _leaf_names(path): leaf
      for path, leaf in jax.tree.leaves_with_path(restored)
  }
  expected_with_path, treedef = jax.tree.flatten_with_path(expected)

  leaves: list[Any] = [None] * len(expected_with_path)
  unmatched = []
  mismatched = []
  for index, (path, want) in enumerate(expected_with_path):
    name = _leaf_names(path)
    if name not in by_name:
      unmatched.append(index)
      continue
    leaf = by_name.pop(name)
    if not _fits(leaf, want):
      mismatched.append(
          f"{'/'.join(name)}: {leaf.dtype}{list(leaf.shape)}, expected"
          f" {want.dtype}{list(want.shape)}"
      )
    leaves[index] = leaf

  # Leaves of custom nodes the function sees as lists, grouped by node.
  unrecorded_nodes: dict[tuple[str, ...], list[tuple[int, Any]]] = {}
  missing = []
  for index in unmatched:
    path, want = expected_with_path[index]
    prefix = _unrecorded_node_prefix(path)
    if prefix is None:
      missing.append(index)
    else:
      unrecorded_nodes.setdefault(prefix, []).append((index, want))
  for prefix, node_leaves in unrecorded_nodes.items():
    candidates = [
        (name, leaf)
        for name, leaf in by_name.items()
        if len(name) > len(prefix) and name[: len(prefix)] == prefix
    ]
    matches = _match_unrecorded_node(node_leaves, candidates)
    if matches is None:
      missing.extend(index for index, _ in node_leaves)
      continue
    for index, name in matches.items():
      leaves[index] = by_name.pop(name)
  missing = [
      "/".join(_leaf_names(expected_with_path[index][0]))
      for index in sorted(missing)
  ]

  problems = []
  if missing:
    problems.append(f"missing {missing[:_MAX_REPORTED]}")
  if mismatched:
    problems.append(f"mismatched {mismatched[:_MAX_REPORTED]}")
  if by_name:
    unexpected = ["/".join(name) for name in sorted(by_name)]
    problems.append(f"unexpected {unexpected[:_MAX_REPORTED]}")
  if problems:
    raise ValueError(
        "The checkpoint does not match the function's parameters: "
        + "; ".join(problems)
        + "."
    )
  return jax.tree.unflatten(treedef, leaves)


def load_params(
    model_dir: pathlib.Path | str,
    function_name: str,
    params_structure: jaxtyping.PyTree,
) -> jaxtyping.PyTree:
  """Loads the weights of an XLA SavedModel function.

  Args:
    model_dir: Directory where the XLA SavedModel was saved.
    function_name: Name of the function in the model manifest.
    params_structure: The parameters the function expects, with any custom nodes
      such as `qwix.QArray` in place. Its leaves only need `shape` and `dtype`,
      so `jax.ShapeDtypeStruct`s (e.g. from `jax.eval_shape`) or concrete arrays
      both work.

  Returns:
    The restored weights, arranged in the structure of `params_structure`.

  Raises:
    ValueError: If no weights are bound to the function, or the checkpoint does
      not match `params_structure`.
  """
  checkpoint_path = find_checkpoint_path(model_dir, function_name)
  with ocp.StandardCheckpointer() as checkpointer:
    restored = checkpointer.restore(checkpoint_path)
  return restructure_params(restored, params_structure)


__all__ = [
    "find_checkpoint_path",
    "load_params",
    "restructure_params",
]

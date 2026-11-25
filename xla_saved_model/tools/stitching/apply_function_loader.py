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

"""Tools for loading XLA SavedModel functions back to JAX."""

from collections.abc import Sequence
import pathlib
from typing import Any

import jax
import jaxtyping

from xla_saved_model.protos import jax_supplemental_pb2
from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src import xsm2jax
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import persistence_lib
from xla_saved_model._src.jax2xsm import constants as jax2xsm_constants

# How many input leaves an error message lists per signature.
_MAX_REPORTED = 5


def _get_single_params_structure(
    apply_fn: xsm2jax.ExportedApplyFn,
) -> jaxtyping.PyTree:
  """Returns the `params` structure for a single `ExportedApplyFn`."""
  exported = apply_fn.exported
  structs = [
      jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in exported.in_avals
  ]
  (params, _), _ = jax.tree.unflatten(exported.in_tree, structs)
  return params


def _normalize_shape(shape: Sequence[Any]) -> tuple[Any, ...]:
  """Normalizes concrete and symbolic dims for cross-scope comparison."""
  return tuple(
      str(dim) if jax.export.is_symbolic_dim(dim) else int(dim) for dim in shape
  )


def _params_structures_match(
    first: jaxtyping.PyTree, second: jaxtyping.PyTree
) -> bool:
  """Returns whether two `params` ShapeDtypeStruct pytrees are identical."""
  first_leaves, first_tree = jax.tree.flatten(first)
  second_leaves, second_tree = jax.tree.flatten(second)
  if first_tree != second_tree:
    return False
  return all(
      leaf_a.dtype == leaf_b.dtype
      and _normalize_shape(leaf_a.shape) == _normalize_shape(leaf_b.shape)
      for leaf_a, leaf_b in zip(first_leaves, second_leaves, strict=True)
  )


def _leaf_shape_and_dtype(leaf: Any) -> tuple[tuple[Any, ...], Any]:
  """Returns `(shape, dtype)` for a JAX array, tracer, scalar, or spec."""
  if isinstance(leaf, jax.ShapeDtypeStruct):
    return tuple(leaf.shape), leaf.dtype
  aval = jax.typeof(leaf)
  if isinstance(aval, jax.core.ShapedArray):
    return tuple(aval.shape), aval.dtype
  raise TypeError(
      "Expected a JAX array, tracer, or ShapeDtypeStruct leaf, got"
      f" {type(leaf).__name__}."
  )


def _dim_matches(
    actual_dim: Any,
    expected_dim: Any,
    sym_bindings: dict[str, Any],
) -> tuple[bool, bool]:
  """Checks whether `actual_dim` fits `expected_dim`.

  Tracks symbolic dimension bindings in `sym_bindings` so repeated symbolic
  dimensions within the same signature (e.g. `f32[b, b]`) must bind to the same
  actual dimension value.

  Args:
    actual_dim: Dimension size from the caller's argument (int or symbolic).
    expected_dim: Dimension size from an exported `in_aval` (int or symbolic).
    sym_bindings: Mutable map from symbolic dim expression to its bound value
      for the candidate function currently being checked.

  Returns:
    A tuple `(matches, is_concrete_match)` where `is_concrete_match` is True
    only when `expected_dim` is concrete and equals `actual_dim`.
  """
  if jax.export.is_symbolic_dim(expected_dim):
    sym_key = str(expected_dim)
    actual_val = (
        str(actual_dim)
        if jax.export.is_symbolic_dim(actual_dim)
        else int(actual_dim)
    )
    if sym_key in sym_bindings:
      return sym_bindings[sym_key] == actual_val, False
    sym_bindings[sym_key] = actual_val
    return True, False
  if jax.export.is_symbolic_dim(actual_dim):
    return False, False
  return int(actual_dim) == int(expected_dim), True


def _matches_exported(
    actual_tree: jax.tree_util.PyTreeDef,
    actual_specs: Sequence[tuple[tuple[Any, ...], Any]],
    exported: jax.export.Exported,
) -> tuple[bool, tuple[bool, ...]]:
  """Returns `(matches, concrete_mask)` for one concrete `Exported`.

  `concrete_mask` contains one boolean per dimension across all leaves in
  `exported.in_avals` (`True` when the expected dimension is concrete, `False`
  when symbolic), enabling partial-order specificity tie-breaking when both
  partially concretized and symbolic specializations match.

  Args:
    actual_tree: Flattened PyTreeDef of `((params, inputs), {})`.
    actual_specs: Sequence of `(shape, dtype)` pairs for the flattened leaves.
    exported: Candidate `jax.export.Exported` to match against.

  Returns:
    A tuple `(matches, concrete_mask)`.
  """
  if actual_tree != exported.in_tree:
    return False, ()
  sym_bindings: dict[str, Any] = {}
  concrete_mask: list[bool] = []
  for (actual_shape, actual_dtype), exp_aval in zip(
      actual_specs, exported.in_avals, strict=True
  ):
    if actual_dtype != exp_aval.dtype or len(actual_shape) != len(
        exp_aval.shape
    ):
      return False, ()
    for actual_dim, exp_dim in zip(actual_shape, exp_aval.shape, strict=True):
      matched, is_concrete = _dim_matches(actual_dim, exp_dim, sym_bindings)
      if not matched:
        return False, ()
      concrete_mask.append(is_concrete)
  return True, tuple(concrete_mask)


def _is_strictly_more_specific(
    mask_a: tuple[bool, ...], mask_b: tuple[bool, ...]
) -> bool:
  """Returns True if `mask_a` is concrete everywhere `mask_b` is, plus more."""
  return (
      len(mask_a) == len(mask_b)
      and mask_a != mask_b
      and all(a or not b for a, b in zip(mask_a, mask_b, strict=True))
  )


def _select_most_specific(
    matches: Sequence[tuple[xsm2jax.ExportedApplyFn, tuple[bool, ...]]],
) -> xsm2jax.ExportedApplyFn | None:
  """Returns the unique match strictly more specific than all others, or None."""
  for i, (fn_i, mask_i) in enumerate(matches):
    if all(
        _is_strictly_more_specific(mask_i, mask_j)
        for j, (_, mask_j) in enumerate(matches)
        if j != i
    ):
      return fn_i
  return None


def _format_leaves_summary(leaves_with_path: Sequence[tuple[Any, Any]]) -> str:
  """Formats up to `_MAX_REPORTED` `(path, shape_dtype)` leaves concisely."""
  parts = []
  for path, leaf in leaves_with_path[:_MAX_REPORTED]:
    key = jax.tree_util.keystr(path) or "<input>"
    shape, dtype = _leaf_shape_and_dtype(leaf)
    parts.append(f"{key}: {dtype}{list(shape)}")
  if len(leaves_with_path) > _MAX_REPORTED:
    parts.append(f"... ({len(leaves_with_path) - _MAX_REPORTED} more)")
  return "{" + ", ".join(parts) + "}"


def _format_inputs_signature(exported: jax.export.Exported) -> str:
  """Returns a concise summary of a concrete function's `inputs` argument."""
  structs = [
      jax.ShapeDtypeStruct(aval.shape, aval.dtype) for aval in exported.in_avals
  ]
  (_, inputs), _ = jax.tree.unflatten(exported.in_tree, structs)
  return _format_leaves_summary(jax.tree.leaves_with_path(inputs))


class PolymorphicExportedApplyFn:
  """Callable `ApplyFn` dispatching across concrete `ExportedApplyFn`s.

  Wraps the concrete functions loaded from a manifest `PolymorphicFunction`
  (`poly_fn`) and routes each `fn(params, inputs)` call to the concrete
  function matching the provided pytree structure, shapes, and dtypes. Works
  during eager execution as well as inside `jax.jit`, `jax.eval_shape`, and
  `jax.export.export`.

  Attributes:
    concrete_functions: Loaded concrete `ExportedApplyFn` instances.
    function_name: Manifest function name, or empty string if unspecified.
  """

  def __init__(
      self,
      concrete_functions: Sequence[xsm2jax.ExportedApplyFn],
      function_name: str = "",
  ):
    """Initializes the polymorphic apply function.

    Args:
      concrete_functions: Non-empty sequence of concrete `ExportedApplyFn`
        instances sharing the same `params` structure.
      function_name: Optional manifest function name for error messages.

    Raises:
      ValueError: If `concrete_functions` is empty or its concrete functions
        have inconsistent `params` structures.
    """
    if not concrete_functions:
      if function_name:
        raise ValueError(
            f"Polymorphic function '{function_name}' has no concrete functions."
        )
      raise ValueError(
          "PolymorphicExportedApplyFn requires at least one concrete function."
      )
    functions = tuple(concrete_functions)
    base_params = _get_single_params_structure(functions[0])
    for idx, fn in enumerate(functions[1:], start=1):
      if not _params_structures_match(
          base_params, _get_single_params_structure(fn)
      ):
        label = f" '{function_name}'" if function_name else ""
        raise ValueError(
            f"Concrete function {idx} of polymorphic function{label} has a"
            " different `params` structure than concrete function 0."
        )
    self._concrete_functions = functions
    self._function_name = function_name

  @property
  def concrete_functions(self) -> tuple[xsm2jax.ExportedApplyFn, ...]:
    """The concrete `ExportedApplyFn` instances."""
    return self._concrete_functions

  @property
  def function_name(self) -> str:
    """The manifest function name, or empty string if unspecified."""
    return self._function_name

  @property
  def exported(self) -> jax.export.Exported:
    """The sole underlying `Exported`, when only one concrete function exists.

    Raises:
      ValueError: If this polymorphic function has multiple concrete functions.
    """
    if len(self._concrete_functions) != 1:
      label = f" '{self._function_name}'" if self._function_name else ""
      raise ValueError(
          f"Polymorphic function{label} has"
          f" {len(self._concrete_functions)} concrete functions; inspect"
          " `concrete_functions` instead of `exported`."
      )
    return self._concrete_functions[0].exported

  def __call__(
      self,
      params: jaxtyping.PyTree,
      inputs: jaxtyping.PyTree,
  ) -> jaxtyping.PyTree:
    actual_leaves, actual_tree = jax.tree.flatten(((params, inputs), {}))
    actual_specs = [_leaf_shape_and_dtype(leaf) for leaf in actual_leaves]

    matches: list[tuple[xsm2jax.ExportedApplyFn, tuple[bool, ...]]] = []
    tree_matched = False
    for fn in self._concrete_functions:
      if actual_tree == fn.exported.in_tree:
        tree_matched = True
      matched, concrete_mask = _matches_exported(
          actual_tree, actual_specs, fn.exported
      )
      if matched:
        matches.append((fn, concrete_mask))

    if len(matches) == 1:
      return matches[0][0](params, inputs)

    label = f" '{self._function_name}'" if self._function_name else ""
    if len(matches) > 1:
      # Prefer a unique specialization that is strictly more specific than
      # every other matching candidate (e.g. `f32[1, l]` beats `f32[b, l]` on
      # `f32[1, 5]`, while incomparable shapes like `f32[1, l]` vs `f32[b, 5]`
      # still raise an ambiguity error).
      most_specific = _select_most_specific(matches)
      if most_specific is not None:
        return most_specific(params, inputs)
      raise ValueError(
          f"Ambiguous call to polymorphic function{label}:"
          f" {len(matches)} concrete functions match the provided arguments."
      )

    available = [
        _format_inputs_signature(fn.exported) for fn in self._concrete_functions
    ]
    params_leaves, params_tree = jax.tree.flatten(params)
    actual_params_specs = jax.tree.unflatten(
        params_tree,
        [
            jax.ShapeDtypeStruct(*_leaf_shape_and_dtype(leaf))
            for leaf in params_leaves
        ],
    )
    expected_params = _get_single_params_structure(self._concrete_functions[0])
    params_note = (
        ""
        if _params_structures_match(actual_params_specs, expected_params)
        else " (params mismatch)"
    )
    structure_note = "" if tree_matched else " (pytree structure mismatch)"
    actual_inputs_summary = _format_leaves_summary(
        jax.tree.leaves_with_path(inputs)
    )
    raise ValueError(
        f"No concrete function in polymorphic function{label} matches the"
        f" provided arguments{structure_note}{params_note} (inputs:"
        f" {actual_inputs_summary}). Available input signatures: {available}."
    )

  def __repr__(self) -> str:
    return (
        f"PolymorphicExportedApplyFn(function_name={self._function_name!r},"
        f" num_concrete_functions={len(self._concrete_functions)})"
    )


def _load_manifest_function(
    manifest_function: manifest_pb2.Function,
    model_dir: pathlib.Path,
    manifest: manifest_pb2.Manifest,
    function_name: str,
) -> xsm2jax.ExportedApplyFn:
  """Converts one manifest `Function` proto into an `ExportedApplyFn`."""
  if not manifest_function.body.HasField("stable_hlo_body"):
    raise ValueError(
        f"Function '{function_name}' does not have a StableHLO body."
    )
  stable_hlo_body = manifest_function.body.stable_hlo_body

  jax_supplemental_data = None
  for data in stable_hlo_body.supplemental_info.values():
    if data.mime_type == jax2xsm_constants.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE:
      jax_supplemental_data = data
      break

  if jax_supplemental_data is None:
    raise ValueError(
        f"Function '{function_name}' does not contain JAX supplemental info"
        " with standard MIME type"
        f" '{jax2xsm_constants.CURRENT_JAX_SUPPLEMENTAL_MIME_TYPE}'."
    )

  raw_bytes = manifest_utils.read_unstructured_data_bytes(
      jax_supplemental_data, str(model_dir), manifest
  )
  jax_supplemental_proto = jax_supplemental_pb2.Function.FromString(raw_bytes)

  return xsm2jax.xsm_fn_to_apply_fn(manifest_function, jax_supplemental_proto)


def load_apply_fn(
    model_dir: pathlib.Path | str,
    function_name: str,
) -> xsm2jax.ExportedApplyFn | PolymorphicExportedApplyFn:
  """Loads an XLA SavedModel function from a model directory as a JAX ApplyFn.

  Supports both single manifest `Function` objects (returning an
  `ExportedApplyFn`) and `PolymorphicFunction` (`poly_fn`) objects containing
  multiple concrete specializations (returning a `PolymorphicExportedApplyFn`
  that dispatches calls by input shape and dtype).

  Args:
    model_dir: Directory where the XLA SavedModel was saved. Can be a
      `pathlib.Path` or string.
    function_name: Name of the function in the model manifest to load.

  Returns:
    A JAX `ApplyFn` (`ExportedApplyFn` for a single function, or
    `PolymorphicExportedApplyFn` for a `poly_fn`) that can be called directly
    as `fn(params, inputs)`.

  Raises:
    FileNotFoundError: If the model directory or manifest cannot be found.
    KeyError: If the specified function name does not exist in the manifest.
    ValueError: If the object is not a function or polymorphic function, has no
      concrete functions, has inconsistent parameter structures across concrete
      functions, lacks a StableHLO body, or lacks JAX supplemental information
      with standard MIME type.
  """
  model_dir = pathlib.Path(model_dir)

  manifest = persistence_lib.load(str(model_dir))

  if function_name not in manifest.objects:
    available_fns = [
        k
        for k, v in manifest.objects.items()
        if v.WhichOneof("case") in ("function", "poly_fn")
    ]
    raise KeyError(
        f"Function '{function_name}' not found in manifest. "
        f"Available functions: {available_fns}"
    )

  obj = manifest.objects[function_name]
  match obj.WhichOneof("case"):
    case "function":
      return _load_manifest_function(
          obj.function, model_dir, manifest, function_name
      )
    case "poly_fn":
      concrete_fns = [
          _load_manifest_function(
              concrete_fn, model_dir, manifest, function_name
          )
          for concrete_fn in obj.poly_fn.concrete_functions
      ]
      return PolymorphicExportedApplyFn(
          concrete_fns, function_name=function_name
      )
    case other_case:
      raise ValueError(
          f"Object '{function_name}' is not a function (got '{other_case}')."
      )


def get_params_structure(
    apply_fn: xsm2jax.ExportedApplyFn | PolymorphicExportedApplyFn,
) -> jaxtyping.PyTree:
  """Returns the structure of a loaded function's `params` argument.

  A loaded function is called as `apply_fn(params, inputs)`; this returns the
  pytree structure of its first positional argument, `params`, with each leaf
  as a `jax.ShapeDtypeStruct`. `inputs` is not included.

  The structure comes from the function's recorded input signature, so custom
  nodes such as `qwix.QArray` are rebuilt only if the model recorded them and
  the module defining each one is imported before `load_apply_fn` is called.
  Otherwise they appear as plain lists.

  Args:
    apply_fn: The function as returned by `load_apply_fn`.

  Returns:
    The `params` pytree, with each leaf a `jax.ShapeDtypeStruct`. Shapes may
    contain symbolic dimensions.
  """
  if isinstance(apply_fn, PolymorphicExportedApplyFn):
    return _get_single_params_structure(apply_fn.concrete_functions[0])
  return _get_single_params_structure(apply_fn)


__all__ = [
    "PolymorphicExportedApplyFn",
    "get_params_structure",
    "load_apply_fn",
]

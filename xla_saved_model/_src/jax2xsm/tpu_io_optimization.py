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

"""Export-time TPU I/O optimizations (R1 flattening and NCHW transpose)."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
from jax import export as jax_export
from jax.experimental import layout as jax_layout
import jax.numpy as jnp
import jaxtyping

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.jax2xsm import utils
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import

NHWC_TO_NCHW_PERM: tuple[int, int, int, int] = (0, 3, 1, 2)
NCHW_TO_NHWC_PERM: tuple[int, int, int, int] = (0, 2, 3, 1)


@dataclasses.dataclass(frozen=True)
class _InputLeafPlan:
  """Optimization plan for a single flattened input leaf.

  Attributes:
    transpose_nchw: Whether the caller transposes `[N, H, W, C]` to `[N, C, H,
      W]` on host and the StableHLO entry reshapes/transposes back to NHWC.
    flatten_r1: Whether the StableHLO entry parameter is flattened to 1-D.
    post_nchw_shape: Shape after host NCHW transpose (or original shape if not
      transposed), used to reshape a 1-D physical input back before the
      on-device NCHW->NHWC transpose.
    physical_shape: Physical tensor shape expected by `@main`.
  """

  transpose_nchw: bool
  flatten_r1: bool
  post_nchw_shape: tuple[int, ...]
  physical_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TpuIoOptimizationPlan:
  """Resolved TPU I/O optimization wrapper and updated specs.

  Attributes:
    wrapped_fn: Jitted wrapper function taking physical inputs and returning
      physical outputs.
    args_spec: Positional argument specs updated with physical input shapes.
    kwargs_spec: Keyword argument specs updated with physical input shapes.
    has_input_optimization: True if at least one input leaf was optimized.
    has_output_optimization: True if at least one output leaf was optimized.
    transposed_input_leaf_indices: Indices of flattened input leaves that use
      host NHWC -> NCHW transposition.
  """

  wrapped_fn: Callable[..., Any]
  args_spec: Sequence[Any]
  kwargs_spec: dict[str, Any]
  has_input_optimization: bool
  has_output_optimization: bool
  transposed_input_leaf_indices: frozenset[int]


def _has_static_shape(shape: Sequence[Any]) -> bool:
  """Returns True if every dimension in `shape` is a concrete Python int."""
  return all(
      isinstance(d, int) and not jax_export.is_symbolic_dim(d) for d in shape
  )


def _has_unsupported_explicit_sharding(sharding: Any) -> bool:
  """Returns True if `sharding` is a rank-bound non-NamedSharding Sharding."""
  if isinstance(sharding, jax_layout.Format):
    sharding = sharding.sharding
  if not isinstance(sharding, jax.sharding.Sharding):
    return False
  if getattr(sharding, "_device_backend", False):
    return False
  return not isinstance(
      sharding,
      (jax.sharding.NamedSharding, jax.sharding.SingleDeviceSharding),
  )


def _has_special_partition_spec(sharding: Any) -> bool:
  """Returns True if a NamedSharding uses unconstrained or (un)reduced axes."""
  if isinstance(sharding, jax_layout.Format):
    sharding = sharding.sharding
  if not isinstance(sharding, jax.sharding.NamedSharding):
    return False
  spec = sharding.spec
  return bool(
      getattr(spec, "unreduced", ())
      or getattr(spec, "reduced", ())
      or any(p is jax.sharding.PartitionSpec.UNCONSTRAINED for p in spec)
  )


def _get_op_sharding(
    sharding: Any,
    aval: jax.core.ShapedArray,
) -> xla_data_pb2.OpSharding | None:
  """Converts a JAX sharding to an XLA `OpSharding` proto, or None."""
  if not isinstance(sharding, jax.sharding.Sharding):
    return None
  if isinstance(sharding, jax.sharding.NamedSharding) and sharding.mesh.empty:
    return None
  if isinstance(sharding, jax.sharding.SingleDeviceSharding):
    return None
  hlo_sharding = sharding._to_xla_hlo_sharding(aval.ndim)  # pylint: disable=protected-access
  return utils.hlo_sharding_to_op_sharding(hlo_sharding)


def _should_skip_for_sharding(
    op_sharding: xla_data_pb2.OpSharding | None,
    ndim: int,
) -> bool:
  """Returns True if sharded along any non-batch tensor dimension (dim >= 1)."""
  if (
      op_sharding is None
      or op_sharding.type != xla_data_pb2.OpSharding.OTHER
      or not op_sharding.tile_assignment_dimensions
  ):
    return False
  num_tensor_dims = len(op_sharding.tile_assignment_dimensions)
  if op_sharding.replicate_on_last_tile_dim:
    num_tensor_dims -= 1
  elif op_sharding.last_tile_dims:
    num_tensor_dims -= len(op_sharding.last_tile_dims)
  num_tensor_dims = min(num_tensor_dims, ndim)
  return any(
      op_sharding.tile_assignment_dimensions[j] > 1
      for j in range(1, num_tensor_dims)
  )


def _should_transpose_nhwc_input(
    aval: jax.core.ShapedArray,
    max_transpose_channels: int,
) -> bool:
  """Checks if a 4-D [N, H, W, C] input qualifies for NHWC -> NCHW transpose."""
  if max_transpose_channels <= 0 or not _has_static_shape(aval.shape):
    return False
  if (
      len(aval.shape) != 4
      or aval.shape[3] <= 0
      or aval.shape[3] > max_transpose_channels
      or aval.shape[3] >= aval.shape[2]
  ):
    return False
  if aval.dtype == jax.dtypes.float0 or aval.dtype == jnp.bool_:
    return False
  if not (
      jnp.issubdtype(aval.dtype, jnp.integer)
      or jnp.issubdtype(aval.dtype, jnp.floating)
  ):
    return False
  return jax.dtypes.itemsize_bits(aval.dtype) in (8, 16, 32, 64)


def _resolve_input_sharding(
    spec: Any,
    raw_sharding: Any,
    exported_nsd: jax.sharding.NamedSharding | None,
) -> jax.sharding.Sharding | None:
  """Resolves effective input leaf sharding, skipping empty meshes."""
  if isinstance(raw_sharding, jax.sharding.Sharding):
    if getattr(raw_sharding, "_device_backend", False):
      return None
    if (
        isinstance(raw_sharding, jax.sharding.NamedSharding)
        and raw_sharding.mesh.empty
    ):
      return None
    return raw_sharding

  spec_sharding = getattr(spec, "sharding", None)
  if isinstance(spec_sharding, jax_layout.Format):
    spec_sharding = spec_sharding.sharding
  if isinstance(spec_sharding, jax.sharding.Sharding):
    if (
        isinstance(spec_sharding, jax.sharding.NamedSharding)
        and spec_sharding.mesh.empty
    ):
      return None
    if isinstance(spec_sharding, jax.sharding.SingleDeviceSharding):
      # Concrete single-device `jax.Array` specs carry a host CPU/GPU/TPU
      # `SingleDeviceSharding` by default; only preserve it when it carries an
      # explicit `pinned_host` memory kind so cross-platform TPU export does
      # not pin a local CPU device into `in_shardings`.
      return (
          spec_sharding if spec_sharding.memory_kind == "pinned_host" else None
      )
    return spec_sharding

  if exported_nsd is not None and not exported_nsd.mesh.empty:
    return exported_nsd
  return None


def _resolve_output_sharding(
    raw_sharding: Any,
    exported_nsd: jax.sharding.NamedSharding | None,
) -> jax.sharding.Sharding | None:
  """Resolves effective output leaf sharding, skipping empty meshes."""
  if isinstance(raw_sharding, jax.sharding.Sharding):
    if getattr(raw_sharding, "_device_backend", False):
      return None
    if (
        isinstance(raw_sharding, jax.sharding.NamedSharding)
        and raw_sharding.mesh.empty
    ):
      return None
    return raw_sharding

  if exported_nsd is not None and not exported_nsd.mesh.empty:
    return exported_nsd
  return None


def _collapse_batch_sharding(
    sharding: jax.sharding.Sharding | None,
) -> jax.sharding.Sharding | None:
  """Keeps only the batch dimension (dim 0) spec on a NamedSharding."""
  if sharding is None:
    return None
  if isinstance(sharding, jax.sharding.NamedSharding):
    if sharding.mesh.empty:
      return None
    batch_spec = sharding.spec[:1] if len(sharding.spec) > 0 else ()
    return sharding.update(spec=jax.sharding.PartitionSpec(*batch_spec))
  return sharding


def _transform_input_sharding(
    sharding: jax.sharding.Sharding | None,
    plan: _InputLeafPlan,
) -> jax.sharding.Sharding | None:
  """Adapts an eligible input leaf's sharding to its physical shape."""
  if sharding is None:
    return None
  if plan.flatten_r1:
    return _collapse_batch_sharding(sharding)
  if plan.transpose_nchw and isinstance(sharding, jax.sharding.NamedSharding):
    if sharding.mesh.empty:
      return None
    padded = tuple(
        sharding.spec[i] if i < len(sharding.spec) else None for i in range(4)
    )
    permuted = tuple(padded[i] for i in NHWC_TO_NCHW_PERM)
    return sharding.update(spec=jax.sharding.PartitionSpec(*permuted))
  return sharding


def _wrap_with_layout(
    layout: Any,
    sharding: jax.sharding.Sharding | None,
) -> Any:
  """Wraps `(layout, sharding)` in `jax_layout.Format` when a layout is set."""
  if layout is not None:
    return jax_layout.Format(layout, sharding)
  return sharding


def _has_pinned_input_layout(
    idx: int,
    spec: Any,
    raw_in_layout: Any,
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None,
) -> bool:
  """Returns True if input leaf `idx` has an explicit custom device layout."""
  if (
      model_inputs_layout is not None
      and idx < len(model_inputs_layout)
      and model_inputs_layout[idx] is not None
  ):
    return True
  if raw_in_layout is not None:
    return True
  spec_sharding = getattr(spec, "sharding", None)
  if (
      isinstance(spec_sharding, jax_layout.Format)
      and spec_sharding.layout is not None
  ):
    return True
  spec_format = getattr(spec, "format", None)
  if (
      isinstance(spec_format, jax_layout.Format)
      and spec_format.layout is not None
  ):
    # Concrete `jax.Array` instances expose `.format.layout` with the default
    # layout; only treat `ShapeDtypeStruct` (not `jax.Array`) `.format` as
    # user-pinned when `raw_in_layout` resolved to `None`.
    return not isinstance(spec, jax.Array)
  return False


def apply_host_transpose_to_full_input_sig(
    full_input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    transposed_input_leaf_indices: frozenset[int],
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Sets `host_transpose` on transposed leaves in a full function signature.

  Args:
    full_input_signature: Full `ShloTensorSpec` PyTree of the exported function.
    transposed_input_leaf_indices: Flat leaf indices whose caller inputs are
      transposed from NHWC to NCHW on the host.

  Returns:
    Updated `ShloTensorSpec` PyTree with `host_transpose=(0, 3, 1, 2)` set on
    each leaf in `transposed_input_leaf_indices`.
  """
  if not transposed_input_leaf_indices:
    return full_input_signature
  flat_leaves, tree_def = jax.tree.flatten(
      full_input_signature,
      is_leaf=lambda x: isinstance(x, shlo_type.ShloTensorSpec),
  )
  updated_leaves = [
      dataclasses.replace(leaf, host_transpose=NHWC_TO_NCHW_PERM)
      if idx in transposed_input_leaf_indices
      else leaf
      for idx, leaf in enumerate(flat_leaves)
  ]
  return jax.tree.unflatten(tree_def, updated_leaves)


def build_tpu_io_optimization_plan(
    fun_jax: Callable[..., Any],
    args_spec: Sequence[Any],
    kwargs_spec: dict[str, Any],
    unoptimized_exported: jax_export.Exported,
    *,
    num_weight_leaves: int = 0,
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None = None,
    num_input_aliased_output: int = 0,
    enable_input_r1_optimization: bool = False,
    enable_output_r1_optimization: bool = False,
    max_transpose_channels: int = 0,
) -> TpuIoOptimizationPlan | None:
  """Builds a wrapped JAX function and specs for TPU I/O shape optimizations.

  Args:
    fun_jax: Jitted JAX function being exported.
    args_spec: Positional argument specs passed to `jax.export.export`.
    kwargs_spec: Keyword argument specs passed to `jax.export.export`.
    unoptimized_exported: Unoptimized `jax.export.Exported` artifact used to
      inspect lowered shardings, kept variable indices, and abstract values.
    num_weight_leaves: Number of leading flattened input leaves corresponding to
      model weights/parameters (excluded from I/O optimizations).
    model_inputs_layout: Optional per-input-leaf custom XLA layouts.
    num_input_aliased_output: Number of trailing donated output buffer leaves on
      the final positional argument when `use_input_aliasing=True`.
    enable_input_r1_optimization: Whether to flatten eligible static rank >= 2
      user inputs to 1-D.
    enable_output_r1_optimization: Whether to flatten eligible static rank >= 2
      outputs to 1-D.
    max_transpose_channels: Maximum channel count `C` for NHWC -> NCHW input
      transposition (`0` disables NCHW transposition).

  Returns:
    A `TpuIoOptimizationPlan` if at least one input or output leaf is optimized,
    or `None` when no optimizations apply or the function uses features
    incompatible with export-time rewriting (e.g., unconcretized symbolic shapes
    or `static_argnums`).

  Raises:
    ValueError: If both `enable_output_r1_optimization` and output donation
      (`num_input_aliased_output > 0`) are enabled.
  """
  if not (
      enable_input_r1_optimization
      or enable_output_r1_optimization
      or max_transpose_channels > 0
  ):
    return None

  if enable_output_r1_optimization and num_input_aliased_output > 0:
    raise ValueError(
        "Output R1 optimization is incompatible with output donation"
        " (use_input_aliasing=True)."
    )

  # Unconcretized shape-polymorphic functions must keep logical `@main`
  # signatures so C++ `RefineShapesIfNeeded` can validate caller shapes and
  # `StablehloTpuOptimizer` can optimize all inputs/outputs post-refinement.
  if unoptimized_exported.uses_global_constants:
    return None

  in_avals = unoptimized_exported.in_avals
  num_user_outputs = unoptimized_exported.out_tree.num_leaves
  out_avals = unoptimized_exported.out_avals[:num_user_outputs]
  if any(
      not isinstance(a, jax.core.ShapedArray) or not _has_static_shape(a.shape)
      for a in (*in_avals, *out_avals)
  ):
    return None

  jit_info: Any = getattr(  # pylint: disable=protected-access
      fun_jax, "_jit_info", None
  )
  if jit_info is not None and (
      jit_info.static_argnums or jit_info.static_argnames
  ):
    return None

  flat_specs, in_tree = jax.tree.flatten((tuple(args_spec), kwargs_spec))
  if in_tree != unoptimized_exported.in_tree or len(flat_specs) != len(
      in_avals
  ):
    return None

  if not hasattr(fun_jax, "trace"):
    return None
  traced = fun_jax.trace(*args_spec, **kwargs_spec)
  num_consts: int = getattr(traced, "_num_consts", 0)  # pylint: disable=protected-access
  traced_params: dict[str, Any] | None = getattr(  # pylint: disable=protected-access
      traced, "_params", None
  )
  if traced_params is None or num_consts != 0:
    return None

  raw_in_shardings = traced_params.get("in_shardings", (None,) * len(in_avals))
  raw_in_layouts = traced_params.get("in_layouts", (None,) * len(in_avals))
  raw_out_shardings = traced_params.get(
      "out_shardings", (None,) * len(unoptimized_exported.out_avals)
  )[:num_user_outputs]
  raw_out_layouts = traced_params.get(
      "out_layouts", (None,) * len(unoptimized_exported.out_avals)
  )[:num_user_outputs]
  donated_invars = traced_params.get(
      "donated_invars", (False,) * len(in_avals)
  )
  keep_unused = bool(traced_params.get("keep_unused", False))
  compiler_options_kvs = traced_params.get("compiler_options_kvs", ())

  in_named_shardings = getattr(  # pylint: disable=protected-access
      unoptimized_exported, "_in_named_shardings", (None,) * len(in_avals)
  )
  out_named_shardings = getattr(  # pylint: disable=protected-access
      unoptimized_exported,
      "_out_named_shardings",
      (None,) * len(unoptimized_exported.out_avals),
  )[:num_user_outputs]

  num_pos_leaves = len(jax.tree.leaves(tuple(args_spec)))
  donated_leaf_indices = (
      set(range(num_pos_leaves - num_input_aliased_output, num_pos_leaves))
      if num_input_aliased_output > 0
      else set()
  )
  kept_var_indices = set(unoptimized_exported.module_kept_var_idx)

  input_plans: list[_InputLeafPlan | None] = [None] * len(in_avals)
  transposed_indices: set[int] = set()
  has_input_optimization = False

  for idx in range(num_weight_leaves, len(in_avals)):
    if idx not in kept_var_indices:
      continue
    if idx in donated_leaf_indices or donated_invars[idx]:
      continue
    if _has_pinned_input_layout(
        idx, flat_specs[idx], raw_in_layouts[idx], model_inputs_layout
    ):
      continue
    aval = in_avals[idx]
    if aval.dtype == jax.dtypes.float0:
      continue
    if (
        _has_unsupported_explicit_sharding(raw_in_shardings[idx])
        or _has_unsupported_explicit_sharding(
            getattr(flat_specs[idx], "sharding", None)
        )
        or _has_special_partition_spec(raw_in_shardings[idx])
        or _has_special_partition_spec(in_named_shardings[idx])
    ):
      continue
    eff_in_sharding = _resolve_input_sharding(
        flat_specs[idx], raw_in_shardings[idx], in_named_shardings[idx]
    )
    if _has_unsupported_explicit_sharding(eff_in_sharding):
      continue
    op_sharding = _get_op_sharding(
        eff_in_sharding or in_named_shardings[idx], aval
    )
    if _should_skip_for_sharding(op_sharding, aval.ndim):
      continue

    orig_shape = tuple(int(d) for d in aval.shape)
    transpose_nchw = _should_transpose_nhwc_input(aval, max_transpose_channels)
    if transpose_nchw:
      post_nchw_shape = (
          orig_shape[0],
          orig_shape[3],
          orig_shape[1],
          orig_shape[2],
      )
    else:
      post_nchw_shape = orig_shape

    flatten_r1 = enable_input_r1_optimization and len(post_nchw_shape) >= 2
    if flatten_r1:
      physical_shape = (math.prod(post_nchw_shape),)
    else:
      physical_shape = post_nchw_shape

    if transpose_nchw or flatten_r1:
      input_plans[idx] = _InputLeafPlan(
          transpose_nchw=transpose_nchw,
          flatten_r1=flatten_r1,
          post_nchw_shape=post_nchw_shape,
          physical_shape=physical_shape,
      )
      has_input_optimization = True
      if transpose_nchw:
        transposed_indices.add(idx)

  output_flatten_flags: list[bool] = [False] * num_user_outputs
  has_output_optimization = False
  if enable_output_r1_optimization:
    for out_idx, aval in enumerate(out_avals):
      if aval.dtype == jax.dtypes.float0 or len(aval.shape) <= 1:
        continue
      if raw_out_layouts[out_idx] is not None:
        continue
      if (
          _has_unsupported_explicit_sharding(raw_out_shardings[out_idx])
          or _has_special_partition_spec(raw_out_shardings[out_idx])
          or _has_special_partition_spec(out_named_shardings[out_idx])
      ):
        continue
      eff_out_sharding = _resolve_output_sharding(
          raw_out_shardings[out_idx], out_named_shardings[out_idx]
      )
      if _has_unsupported_explicit_sharding(eff_out_sharding):
        continue
      op_sharding = _get_op_sharding(
          eff_out_sharding or out_named_shardings[out_idx], aval
      )
      if _should_skip_for_sharding(op_sharding, aval.ndim):
        continue
      output_flatten_flags[out_idx] = True
      has_output_optimization = True

  if not has_input_optimization and not has_output_optimization:
    return None

  has_kwargs = bool(kwargs_spec)
  updated_in_entries: list[Any] = []
  updated_flat_specs: list[Any] = []
  for idx, spec in enumerate(flat_specs):
    plan = input_plans[idx]
    orig_sharding = _resolve_input_sharding(
        spec, raw_in_shardings[idx], in_named_shardings[idx]
    )
    if plan is not None:
      new_sharding = _transform_input_sharding(orig_sharding, plan)
      updated_in_entries.append(new_sharding)
      updated_flat_specs.append(
          jax.ShapeDtypeStruct(
              plan.physical_shape,
              spec.dtype,
              sharding=new_sharding,
              weak_type=getattr(spec, "weak_type", False),
          )
      )
    else:
      entry = _wrap_with_layout(raw_in_layouts[idx], orig_sharding)
      updated_in_entries.append(entry)
      if has_kwargs and entry is not None:
        # `jax.jit` forbids passing `in_shardings` when keyword arguments are
        # present; carry per-leaf sharding/layout on `ShapeDtypeStruct` instead.
        aval = in_avals[idx]
        updated_flat_specs.append(
            jax.ShapeDtypeStruct(
                aval.shape if aval is not None else spec.shape,
                spec.dtype,
                sharding=entry,
                weak_type=getattr(spec, "weak_type", False),
            )
        )
      else:
        updated_flat_specs.append(spec)

  opt_args_spec, opt_kwargs_spec = jax.tree.unflatten(
      in_tree, updated_flat_specs
  )

  updated_out_entries: list[Any] = []
  for out_idx, flatten in enumerate(output_flatten_flags):
    orig_out_sharding = _resolve_output_sharding(
        raw_out_shardings[out_idx], out_named_shardings[out_idx]
    )
    if flatten:
      updated_out_entries.append(_collapse_batch_sharding(orig_out_sharding))
    else:
      updated_out_entries.append(
          _wrap_with_layout(raw_out_layouts[out_idx], orig_out_sharding)
      )

  @functools.wraps(fun_jax)
  def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
    if has_input_optimization:
      flat_args = in_tree.flatten_up_to((args, kwargs))
      restored_flat_args = []
      for idx, arg in enumerate(flat_args):
        plan = input_plans[idx]
        if plan is not None:
          x = arg
          if plan.flatten_r1:
            x = jnp.reshape(x, plan.post_nchw_shape)
          if plan.transpose_nchw:
            x = jnp.transpose(x, NCHW_TO_NHWC_PERM)
          restored_flat_args.append(x)
        else:
          restored_flat_args.append(arg)
      restored_args, restored_kwargs = jax.tree.unflatten(
          in_tree, restored_flat_args
      )
    else:
      restored_args, restored_kwargs = args, kwargs

    out = traced(*restored_args, **restored_kwargs)

    if has_output_optimization:
      flat_outs = unoptimized_exported.out_tree.flatten_up_to(out)
      flattened_outs = [
          jnp.reshape(y, (-1,)) if flatten else y
          for y, flatten in zip(flat_outs, output_flatten_flags, strict=True)
      ]
      out = unoptimized_exported.out_tree.unflatten(flattened_outs)
    return out

  device_or_backend_set = jit_info is not None and (
      jit_info.device is not None or jit_info.backend is not None
  )
  jit_kwargs: dict[str, Any] = {}
  if jit_info is not None:
    if jit_info.device is not None:
      jit_kwargs["device"] = jit_info.device
    if jit_info.backend is not None:
      jit_kwargs["backend"] = jit_info.backend
    if jit_info.donate_argnames:
      jit_kwargs["donate_argnames"] = jit_info.donate_argnames

  if not device_or_backend_set:
    if (
        any(entry is not None for entry in updated_in_entries)
        and not opt_kwargs_spec
    ):
      args_in_shardings, _ = jax.tree.unflatten(in_tree, updated_in_entries)
      jit_kwargs["in_shardings"] = args_in_shardings
    if any(entry is not None for entry in updated_out_entries):
      jit_kwargs["out_shardings"] = jax.tree.unflatten(
          unoptimized_exported.out_tree, updated_out_entries
      )

  donate_argnums = (
      set(jit_info.donate_argnums) if jit_info is not None else set()
  )
  if num_input_aliased_output > 0:
    donate_argnums.add(len(opt_args_spec) - 1)
  if donate_argnums:
    jit_kwargs["donate_argnums"] = tuple(sorted(donate_argnums))
  if keep_unused or num_input_aliased_output > 0:
    jit_kwargs["keep_unused"] = True
  if compiler_options_kvs:
    jit_kwargs["compiler_options"] = dict(compiler_options_kvs)

  return TpuIoOptimizationPlan(
      wrapped_fn=jax.jit(wrapped_fn, **jit_kwargs),
      args_spec=opt_args_spec,
      kwargs_spec=opt_kwargs_spec,
      has_input_optimization=has_input_optimization,
      has_output_optimization=has_output_optimization,
      transposed_input_leaf_indices=frozenset(transposed_indices),
  )

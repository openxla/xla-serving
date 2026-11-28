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

"""Export-time TPU I/O optimizations (R1 input and output flattening)."""

from collections.abc import Callable, Sequence
import dataclasses
import functools
import math
from typing import Any

import jax
from jax import export as jax_export
from jax.experimental import layout as jax_layout
from jax.interpreters import partial_eval
import jax.numpy as jnp


@dataclasses.dataclass(frozen=True)
class TpuIoOptimizationPlan:
  """Export-time TPU I/O optimization plan for `jax2xsm.convert`."""

  wrapped_fn: jax.stages.Wrapped
  args_spec: Sequence[Any]
  kwargs_spec: dict[str, Any]
  has_input_optimization: bool
  has_output_optimization: bool


def _is_byte_addressable_dtype(dtype: Any) -> bool:
  """Returns True if `dtype` is byte-addressable (excludes float0 and int4)."""
  return (
      dtype != jax.dtypes.float0 and jax.dtypes.itemsize_bits(dtype) % 8 == 0
  )


def _has_symbolic_dim(avals: Sequence[jax.core.ShapedArray]) -> bool:
  """Returns True if any aval in `avals` contains a symbolic dimension."""
  return any(
      any(jax_export.is_symbolic_dim(dim) for dim in aval.shape)
      for aval in avals
  )


def _is_unsharded_device_io(*sources: Sequence[Any]) -> bool:
  """Returns True if no source carries multi-device, host, or Format layouts."""
  is_single_dev = getattr(
      jax.sharding,
      "is_single_device_sharding",
      lambda sharding: isinstance(sharding, jax.sharding.SingleDeviceSharding),
  )
  for src in sources:
    for candidate in src:
      if isinstance(candidate, jax_layout.Format):
        return False
      if isinstance(candidate, jax.sharding.Sharding):
        is_pinned = getattr(candidate, "memory_kind", None) == "pinned_host"
        if is_pinned or not (
            getattr(candidate, "_device_backend", False)
            or is_single_dev(candidate)
        ):
          return False
  return True


def _plan_input_leaf(
    spec: Any,
    aval: jax.core.ShapedArray,
    *,
    enable_input_r1: bool,
) -> tuple[tuple[int, ...] | None, Any]:
  """Computes `(restored_shape, updated_spec)` for an eligible leaf."""
  if (
      not enable_input_r1
      or len(aval.shape) < 2
      or not _is_byte_addressable_dtype(aval.dtype)
  ):
    return None, spec
  orig_shape = tuple(int(dim) for dim in aval.shape)
  new_spec = jax.ShapeDtypeStruct(
      (math.prod(orig_shape),),
      spec.dtype,
      weak_type=getattr(spec, "weak_type", False),
  )
  return orig_shape, new_spec


@dataclasses.dataclass(frozen=True)
class _TraceInfo:
  """Validated trace and PyTree metadata for planning."""

  jit_info: Any
  flat_specs: list[Any]
  in_avals: list[jax.core.ShapedArray]
  out_avals: list[jax.core.ShapedArray]
  in_tree: jax.tree_util.PyTreeDef
  out_tree: jax.tree_util.PyTreeDef
  kept_vars: frozenset[int]
  keep_unused: bool


def _make_jit_kwargs(trace_info: _TraceInfo) -> dict[str, Any]:
  """Builds `jax.jit` keyword arguments preserving device/backend placement."""
  jit_kwargs: dict[str, Any] = {}
  jit_info = trace_info.jit_info
  if jit_info is not None:
    if jit_info.device is not None:
      jit_kwargs["device"] = jit_info.device
    if jit_info.backend is not None:
      jit_kwargs["backend"] = jit_info.backend
  if trace_info.keep_unused:
    jit_kwargs["keep_unused"] = True
  return jit_kwargs


def _is_pre_trace_eligible(
    fun_jax: Callable[..., Any],
    kwargs_spec: dict[str, Any],
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None,
) -> bool:
  """Returns True if `fun_jax` and layout args pass pre-trace safety guards."""
  if model_inputs_layout and any(
      layout is not None for layout in model_inputs_layout
  ):
    return False
  jit_info: Any = getattr(fun_jax, "_jit_info", None)
  return jit_info is None or not (
      jit_info.static_argnums
      or jit_info.static_argnames
      or jit_info.compiler_options_kvs
      or (kwargs_spec and jit_info.donate_argnames)
  )


def _compute_kept_vars(
    traced: jax.stages.Traced, num_consts: int, keep_unused: bool
) -> frozenset[int]:
  """Returns flat explicit input indices retained after dead-code pruning."""
  jaxpr: Any = getattr(traced, "jaxpr")
  if keep_unused:
    return frozenset(range(len(jaxpr.in_avals) - num_consts))
  _, _, used_inputs = partial_eval.dce_jaxpr_consts(
      jaxpr, [True] * len(jaxpr.outvars)
  )
  return frozenset(
      i for i, used in enumerate(used_inputs[num_consts:]) if used
  )


def _prepare_trace_info(
    fun_jax: Callable[..., Any],
    args_spec: Sequence[Any],
    kwargs_spec: dict[str, Any],
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None,
) -> _TraceInfo | None:
  """Validates `fun_jax` and traces it for single-device R1 planning."""
  if not _is_pre_trace_eligible(fun_jax, kwargs_spec, model_inputs_layout):
    return None
  flat_specs, in_tree = jax.tree.flatten((tuple(args_spec), kwargs_spec))
  jitted_fn = (
      fun_jax if isinstance(fun_jax, jax.stages.Wrapped) else jax.jit(fun_jax)
  )
  traced = jitted_fn.trace(*args_spec, **kwargs_spec)
  in_avals: list[jax.core.ShapedArray] = jax.tree.leaves(traced.in_avals)
  out_avals: list[jax.core.ShapedArray] = jax.tree.leaves(traced.out_avals)
  if (
      len(flat_specs) != len(in_avals)
      or _has_symbolic_dim(in_avals)
      or any(
          not isinstance(aval, jax.core.ShapedArray)
          for aval in (*in_avals, *out_avals)
      )
  ):
    return None
  num_consts: int = getattr(traced, "_num_consts", 0)
  traced_params: dict[str, Any] = getattr(traced, "_params")
  num_out = traced.out_tree.num_leaves
  in_layouts = traced_params["in_layouts"][num_consts:]
  out_layouts = traced_params["out_layouts"][:num_out]
  donated = traced_params["donated_invars"][num_consts:]
  if any(layout is not None for layout in (*in_layouts, *out_layouts)) or any(
      donated
  ):
    return None
  if not _is_unsharded_device_io(
      traced_params["in_shardings"][num_consts:],
      [getattr(spec, "sharding", None) for spec in flat_specs],
      traced_params["out_shardings"][:num_out],
  ):
    return None
  keep_unused = bool(traced_params["keep_unused"])
  return _TraceInfo(
      jit_info=getattr(fun_jax, "_jit_info", None),
      flat_specs=flat_specs,
      in_avals=in_avals,
      out_avals=out_avals[:num_out],
      in_tree=in_tree,
      out_tree=traced.out_tree,
      kept_vars=_compute_kept_vars(traced, num_consts, keep_unused),
      keep_unused=keep_unused,
  )


def _plan_all_inputs(
    trace_info: _TraceInfo,
    *,
    num_weight_leaves: int,
    enable_input_r1: bool,
) -> tuple[list[tuple[int, ...] | None], list[Any]]:
  """Plans input R1 flattening across all flat input leaves."""
  restored_shapes: list[tuple[int, ...] | None] = []
  updated_specs: list[Any] = []
  for i, (spec, aval) in enumerate(
      zip(trace_info.flat_specs, trace_info.in_avals, strict=True)
  ):
    if i >= num_weight_leaves and i in trace_info.kept_vars:
      shape, spec = _plan_input_leaf(
          spec, aval, enable_input_r1=enable_input_r1
      )
    else:
      shape = None
    restored_shapes.append(shape)
    updated_specs.append(spec)
  return restored_shapes, updated_specs


def _plan_all_outputs(
    trace_info: _TraceInfo,
    *,
    enable_output_r1: bool,
) -> list[bool]:
  """Plans output R1 flattening across output leaves."""
  return [
      enable_output_r1
      and _is_byte_addressable_dtype(aval.dtype)
      and len(aval.shape) >= 2
      for aval in trace_info.out_avals
  ]


def _make_wrapped_fn(
    fun_jax: Callable[..., Any],
    trace_info: _TraceInfo,
    restored_input_shapes: Sequence[tuple[int, ...] | None],
    output_flatten_flags: Sequence[bool],
) -> Callable[..., Any]:
  """Wraps `fun_jax` with TPU input restoration and output R1 reshape."""

  @functools.wraps(fun_jax)
  def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
    restored = [
        jnp.reshape(arg, shape) if shape is not None else arg
        for arg, shape in zip(
            trace_info.in_tree.flatten_up_to((args, kwargs)),
            restored_input_shapes,
        )
    ]
    args, kwargs = jax.tree.unflatten(trace_info.in_tree, restored)
    out = fun_jax(*args, **kwargs)
    return trace_info.out_tree.unflatten([
        jnp.reshape(leaf, (-1,)) if flatten else leaf
        for leaf, flatten in zip(
            trace_info.out_tree.flatten_up_to(out),
            output_flatten_flags,
        )
    ])

  return wrapped_fn


def build_tpu_io_optimization_plan(
    fun_jax: Callable[..., Any],
    args_spec: Sequence[Any],
    kwargs_spec: dict[str, Any],
    *,
    num_weight_leaves: int = 0,
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None = None,
    enable_input_r1_optimization: bool = False,
    enable_output_r1_optimization: bool = False,
) -> TpuIoOptimizationPlan | None:
  """Builds a wrapped JAX function and specs for TPU I/O R1 optimizations.

  Args:
    fun_jax: JAX function (raw or `jax.jit`-wrapped) to optimize.
    args_spec: Positional input specs used to trace `fun_jax`.
    kwargs_spec: Keyword input specs used to trace `fun_jax`.
    num_weight_leaves: Leading flat weight leaf count (never optimized).
    model_inputs_layout: Optional explicit layouts; skips optimization if set.
    enable_input_r1_optimization: Whether to flatten rank >= 2 user inputs.
    enable_output_r1_optimization: Whether to flatten rank >= 2 outputs.

  Returns:
    A `TpuIoOptimizationPlan` if at least one leaf is optimized, else `None`.
  """
  if not (enable_input_r1_optimization or enable_output_r1_optimization):
    return None
  trace_info = _prepare_trace_info(
      fun_jax, args_spec, kwargs_spec, model_inputs_layout
  )
  if trace_info is None:
    return None
  restored_shapes, updated_specs = _plan_all_inputs(
      trace_info,
      num_weight_leaves=num_weight_leaves,
      enable_input_r1=enable_input_r1_optimization,
  )
  out_flags = _plan_all_outputs(
      trace_info,
      enable_output_r1=enable_output_r1_optimization,
  )
  has_in_opt = any(shape is not None for shape in restored_shapes)
  has_out_opt = any(out_flags)
  if not has_in_opt and not has_out_opt:
    return None
  opt_args, opt_kwargs = jax.tree.unflatten(trace_info.in_tree, updated_specs)
  jit_kwargs = _make_jit_kwargs(trace_info)
  wrapped_fn = _make_wrapped_fn(fun_jax, trace_info, restored_shapes, out_flags)
  return TpuIoOptimizationPlan(
      wrapped_fn=jax.jit(wrapped_fn, **jit_kwargs),
      args_spec=opt_args,
      kwargs_spec=opt_kwargs,
      has_input_optimization=has_in_opt,
      has_output_optimization=has_out_opt,
  )

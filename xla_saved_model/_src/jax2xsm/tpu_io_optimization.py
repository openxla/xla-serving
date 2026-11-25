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


def _has_multi_device_sharding(candidates: Sequence[Any]) -> bool:
  """Returns True if any candidate sharding spans multiple devices."""
  return any(
      isinstance(s, jax.sharding.Sharding) and s.num_devices > 1
      for s in candidates
  )


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


def _plan_all_inputs(
    flat_specs: Sequence[Any],
    in_avals: Sequence[jax.core.ShapedArray],
    *,
    num_weight_leaves: int,
    enable_input_r1: bool,
) -> tuple[list[tuple[int, ...] | None], list[Any]]:
  """Plans input R1 flattening across all flat input leaves."""
  restored_shapes: list[tuple[int, ...] | None] = []
  updated_specs: list[Any] = []
  for i, (spec, aval) in enumerate(zip(flat_specs, in_avals)):
    if i >= num_weight_leaves:
      shape, spec = _plan_input_leaf(
          spec, aval, enable_input_r1=enable_input_r1
      )
    else:
      shape = None
    restored_shapes.append(shape)
    updated_specs.append(spec)
  return restored_shapes, updated_specs


def _plan_all_outputs(
    out_avals: Sequence[jax.core.ShapedArray],
    *,
    enable_output_r1: bool,
) -> list[bool]:
  """Plans output R1 flattening across output leaves."""
  return [
      enable_output_r1
      and _is_byte_addressable_dtype(aval.dtype)
      and len(aval.shape) >= 2
      for aval in out_avals
  ]


def _make_wrapped_fn(
    fun_jax: Callable[..., Any],
    in_tree: jax.tree_util.PyTreeDef,
    out_tree: jax.tree_util.PyTreeDef,
    restored_input_shapes: Sequence[tuple[int, ...] | None],
    output_flatten_flags: Sequence[bool],
) -> Callable[..., Any]:
  """Wraps `fun_jax` with TPU input restoration and output R1 reshape."""

  @functools.wraps(fun_jax)
  def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
    restored = [
        jnp.reshape(arg, shape) if shape is not None else arg
        for arg, shape in zip(
            in_tree.flatten_up_to((args, kwargs)),
            restored_input_shapes,
        )
    ]
    args, kwargs = jax.tree.unflatten(in_tree, restored)
    out = fun_jax(*args, **kwargs)
    return out_tree.unflatten([
        jnp.reshape(leaf, (-1,)) if flatten else leaf
        for leaf, flatten in zip(
            out_tree.flatten_up_to(out),
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
    enable_input_r1_optimization: bool = False,
    enable_output_r1_optimization: bool = False,
) -> TpuIoOptimizationPlan | None:
  """Builds a wrapped JAX function and specs for TPU I/O R1 optimizations.

  Args:
    fun_jax: JAX function (raw or `jax.jit`-wrapped) to optimize.
    args_spec: Positional input specs used to trace `fun_jax`.
    kwargs_spec: Keyword input specs used to trace `fun_jax`.
    num_weight_leaves: Leading flat weight leaf count (never optimized).
    enable_input_r1_optimization: Whether to flatten rank >= 2 user inputs.
    enable_output_r1_optimization: Whether to flatten rank >= 2 outputs.

  Returns:
    A `TpuIoOptimizationPlan` if at least one leaf is optimized, else `None`.
  """
  if not (enable_input_r1_optimization or enable_output_r1_optimization):
    return None
  flat_specs, in_tree = jax.tree.flatten((tuple(args_spec), kwargs_spec))
  jitted_fn = (
      fun_jax if isinstance(fun_jax, jax.stages.Wrapped) else jax.jit(fun_jax)
  )
  traced = jitted_fn.trace(*args_spec, **kwargs_spec)
  in_avals: list[jax.core.ShapedArray] = jax.tree.leaves(traced.in_avals)
  out_avals: list[jax.core.ShapedArray] = jax.tree.leaves(traced.out_avals)
  traced_params: dict[str, Any] = getattr(traced, "_params")
  shardings = (
      *(getattr(spec, "sharding", None) for spec in flat_specs),
      *traced_params["in_shardings"],
      *traced_params["out_shardings"],
  )
  if _has_symbolic_dim(in_avals) or _has_multi_device_sharding(shardings):
    return None
  restored_shapes, updated_specs = _plan_all_inputs(
      flat_specs,
      in_avals,
      num_weight_leaves=num_weight_leaves,
      enable_input_r1=enable_input_r1_optimization,
  )
  out_flags = _plan_all_outputs(
      out_avals,
      enable_output_r1=enable_output_r1_optimization,
  )
  has_in_opt = any(shape is not None for shape in restored_shapes)
  has_out_opt = any(out_flags)
  if not has_in_opt and not has_out_opt:
    return None
  opt_args, opt_kwargs = jax.tree.unflatten(in_tree, updated_specs)
  wrapped_fn = _make_wrapped_fn(
      fun_jax, in_tree, traced.out_tree, restored_shapes, out_flags
  )
  return TpuIoOptimizationPlan(
      wrapped_fn=jax.jit(wrapped_fn),
      args_spec=opt_args,
      kwargs_spec=opt_kwargs,
      has_input_optimization=has_in_opt,
      has_output_optimization=has_out_opt,
  )

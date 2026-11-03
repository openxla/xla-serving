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

"""Export-time TPU I/O shape optimizations (R1 flattening and NCHW transpose)."""

# pylint: disable=protected-access
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
import numpy as np

from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.jax2xsm import utils
from xla_saved_model.protos.xla import xla_data_pb2  # pylint: disable=g-direct-tensorflow-import

NHWC_TO_NCHW_PERM: tuple[int, int, int, int] = (0, 3, 1, 2)
NCHW_TO_NHWC_PERM: tuple[int, int, int, int] = (0, 2, 3, 1)


@dataclasses.dataclass(frozen=True)
class _InputLeafPlan:
  """Optimization plan for a single flattened input leaf."""

  transpose_nchw: bool
  flatten_r1: bool
  post_nchw_shape: tuple[int, ...]
  physical_shape: tuple[int, ...]


@dataclasses.dataclass(frozen=True)
class TpuIoOptimizationPlan:
  """Resolved TPU I/O optimization wrapper and updated specs."""

  wrapped_fn: Callable[..., Any]
  args_spec: Sequence[Any]
  kwargs_spec: dict[str, Any]
  has_input_optimization: bool
  has_output_optimization: bool
  transposed_input_leaf_indices: frozenset[int]


def _has_static_shape(shape: Sequence[Any]) -> bool:
  return all(
      isinstance(d, int) and not jax_export.is_symbolic_dim(d) for d in shape
  )


def _get_op_sharding(
    nsd: jax.sharding.NamedSharding | None,
    aval: jax.core.ShapedArray,
) -> xla_data_pb2.OpSharding | None:
  if nsd is None:
    return None
  hlo_sharding = nsd._to_xla_hlo_sharding(aval.ndim)
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
      or aval.shape[3] > max_transpose_channels
      or aval.shape[3] >= aval.shape[2]
  ):
    return False
  dtype = np.dtype(aval.dtype)
  if not (
      np.issubdtype(dtype, np.integer)
      or np.issubdtype(dtype, np.floating)
      or dtype == jnp.bfloat16
  ):
    return False
  if dtype in (jnp.int4, jnp.uint4, np.dtype(np.bool_)):
    return False
  return dtype.itemsize in (1, 2, 4, 8)


def _collapse_batch_named_sharding(
    nsd: jax.sharding.NamedSharding | None,
) -> jax.sharding.NamedSharding | None:
  """Keeps only the batch dimension (dim 0) partition spec on a NamedSharding."""
  if nsd is None:
    return None
  batch_spec = nsd.spec[:1] if len(nsd.spec) > 0 else ()
  return jax.sharding.NamedSharding(
      nsd.mesh,
      jax.sharding.PartitionSpec(*batch_spec),
      memory_kind=nsd.memory_kind,
  )


def apply_host_transpose_to_user_input_sig(
    input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    num_weight_leaves: int,
    transposed_input_leaf_indices: frozenset[int],
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Sets `host_transpose` on transposed user-input leaves in a signature tree."""
  if not transposed_input_leaf_indices:
    return input_signature
  flat_leaves, tree_def = jax.tree.flatten(
      input_signature,
      is_leaf=lambda x: isinstance(x, shlo_type.ShloTensorSpec),
  )
  updated_leaves = []
  for idx, leaf in enumerate(flat_leaves):
    if idx + num_weight_leaves in transposed_input_leaf_indices:
      updated_leaves.append(
          dataclasses.replace(leaf, host_transpose=NHWC_TO_NCHW_PERM)
      )
    else:
      updated_leaves.append(leaf)
  return jax.tree.unflatten(tree_def, updated_leaves)


def apply_host_transpose_to_full_input_sig(
    full_input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    transposed_input_leaf_indices: frozenset[int],
) -> jaxtyping.PyTree[shlo_type.ShloTensorSpec]:
  """Sets `host_transpose` on transposed leaves in a full function signature."""
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
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None = None,
    num_input_aliased_output: int = 0,
    enable_input_r1_optimization: bool = False,
    enable_output_r1_optimization: bool = False,
    max_transpose_channels: int = 0,
) -> TpuIoOptimizationPlan | None:
  """Builds a wrapped JAX function and specs for TPU I/O shape optimizations."""
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

  in_avals = unoptimized_exported.in_avals
  in_named_shardings = getattr(
      unoptimized_exported, "_in_named_shardings", (None,) * len(in_avals)
  )
  num_user_outputs = unoptimized_exported.out_tree.num_leaves
  out_avals = unoptimized_exported.out_avals[:num_user_outputs]
  out_named_shardings = getattr(
      unoptimized_exported,
      "_out_named_shardings",
      (None,) * len(unoptimized_exported.out_avals),
  )[:num_user_outputs]

  num_weight_leaves = (
      len(jax.tree.leaves(args_spec[0])) if len(args_spec) >= 2 else 0
  )
  user_input_end = len(in_avals) - num_input_aliased_output
  kept_var_indices = set(unoptimized_exported.module_kept_var_idx)

  input_plans: list[_InputLeafPlan | None] = [None] * len(in_avals)
  transposed_indices: set[int] = set()
  has_input_optimization = False

  for idx in range(num_weight_leaves, user_input_end):
    if idx not in kept_var_indices:
      continue
    if (
        model_inputs_layout is not None
        and idx < len(model_inputs_layout)
        and model_inputs_layout[idx] is not None
    ):
      continue
    aval = in_avals[idx]
    if not isinstance(aval, jax.core.ShapedArray) or not _has_static_shape(
        aval.shape
    ):
      continue
    op_sharding = _get_op_sharding(in_named_shardings[idx], aval)
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
      if not isinstance(aval, jax.core.ShapedArray) or not _has_static_shape(
          aval.shape
      ):
        continue
      if len(aval.shape) <= 1:
        continue
      op_sharding = _get_op_sharding(out_named_shardings[out_idx], aval)
      if _should_skip_for_sharding(op_sharding, aval.ndim):
        continue
      output_flatten_flags[out_idx] = True
      has_output_optimization = True

  if not has_input_optimization and not has_output_optimization:
    return None

  flat_specs, in_tree = jax.tree.flatten((tuple(args_spec), kwargs_spec))
  updated_in_shardings: list[jax.sharding.NamedSharding | None] = []
  updated_flat_specs: list[Any] = []
  for idx, spec in enumerate(flat_specs):
    plan = input_plans[idx]
    orig_nsd = in_named_shardings[idx]
    if plan is not None:
      new_nsd = _collapse_batch_named_sharding(orig_nsd)
      updated_in_shardings.append(new_nsd)
      updated_flat_specs.append(
          jax.ShapeDtypeStruct(
              plan.physical_shape, spec.dtype, sharding=new_nsd
          )
      )
    else:
      updated_in_shardings.append(orig_nsd)
      updated_flat_specs.append(spec)

  opt_args_spec, opt_kwargs_spec = jax.tree.unflatten(
      in_tree, updated_flat_specs
  )

  updated_out_shardings: list[jax.sharding.NamedSharding | None] = [
      _collapse_batch_named_sharding(nsd) if flatten else nsd
      for nsd, flatten in zip(out_named_shardings, output_flatten_flags)
  ]

  @functools.wraps(fun_jax)
  def wrapped_fn(*args: Any, **kwargs: Any) -> Any:
    if has_input_optimization:
      flat_args = jax.tree.leaves((args, kwargs))
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

    out = fun_jax(*restored_args, **restored_kwargs)

    if has_output_optimization:
      flat_outs, out_tree = jax.tree.flatten(out)
      flattened_outs = [
          jnp.reshape(y, (-1,)) if flatten else y
          for y, flatten in zip(flat_outs, output_flatten_flags)
      ]
      out = jax.tree.unflatten(out_tree, flattened_outs)
    return out

  jit_kwargs: dict[str, Any] = {}
  if (
      any(nsd is not None for nsd in updated_in_shardings)
      and not opt_kwargs_spec
  ):
    args_in_shardings, _ = jax.tree.unflatten(in_tree, updated_in_shardings)
    jit_kwargs["in_shardings"] = args_in_shardings
  if any(nsd is not None for nsd in updated_out_shardings):
    jit_kwargs["out_shardings"] = jax.tree.unflatten(
        unoptimized_exported.out_tree, updated_out_shardings
    )
  if num_input_aliased_output > 0:
    jit_kwargs["donate_argnums"] = 2
    jit_kwargs["keep_unused"] = True

  return TpuIoOptimizationPlan(
      wrapped_fn=jax.jit(wrapped_fn, **jit_kwargs),
      args_spec=opt_args_spec,
      kwargs_spec=opt_kwargs_spec,
      has_input_optimization=has_input_optimization,
      has_output_optimization=has_output_optimization,
      transposed_input_leaf_indices=frozenset(transposed_indices),
  )

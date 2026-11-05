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

"""A converter from JAX to ML Exported Model (EM)."""

# pylint: disable=protected-access
from collections.abc import Callable, Mapping, Sequence
import dataclasses
import re
from typing import Any

import jax
from jax import export as jax_export
from jax.experimental import layout as jax_layout
import jaxtyping

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import compile_options_util
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import shlo_type
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.core import value
from xla_saved_model._src.jax2xsm import constants
from xla_saved_model._src.jax2xsm import jax_specific_info
from xla_saved_model._src.jax2xsm import tpu_io_optimization
from xla_saved_model._src.jax2xsm import typing
from xla_saved_model._src.jax2xsm import utils


@dataclasses.dataclass
class XlaCompileOptionsSupplementalInfo(
    shlo_function.ShloFunctionSupplementalInfo
):
  """Supplemental info for XLA compile options.

  Attributes:
    xla_compile_options: The XLA compile options.
  """

  xla_compile_options_per_platform: manifest_pb2.CompileOptionsProtoMap

  def serializable_to_proto(
      self,
  ) -> unstructured_data.UnstructuredDataWithExtName:
    """Serializes to an `UnstructuredDataWithExtName`."""
    return unstructured_data.UnstructuredDataWithExtName(
        proto=manifest_pb2.UnstructuredData(
            inlined_bytes=self.xla_compile_options_per_platform.SerializeToString(),
            mime_type=constants.XLA_COMPILE_OPTIONS_MIME_TYPE,
        ),
        ext_name=None,
    )


def jax_exported_to_shlo_fn(
    exported: jax_export.Exported,
    xla_compile_options_per_platform: (
        manifest_pb2.CompileOptionsProtoMap | None
    ) = None,
    model_param_names: Sequence[str] | None = None,
    *,
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None = None,
    num_input_aliased_output: int = 0,
    weights_name: str | None = None,
) -> shlo_function.ShloFunction:
  """Converts a `jax.export.Exported` to an Orbax Model `ShloFunction`.

  Args:
    exported: The `jax.export.Exported` object.
    xla_compile_options_per_platform: XLA compile options to be saved in the
      model artifact, to ensure XLA compilation consistency and reproducibility
      between export time and serving time. Each map entry corresponds to a
      platform type (e.g. TPU, GPU, etc.).
    model_param_names: Optional. A list of the model parameter names in the
      dot-separated key path format (e.g. "params.key.subkey"). If provided,
      only these parameters will be loaded from the checkpoint when the function
      is executed.
    model_inputs_layout: Optional. A sequence of `jax_layout.Layout | None`
      entries specifying the layouts for the model inputs.
    num_input_aliased_output: The number of trailing arguments of `exported`
      that are donated output buffers rather than caller-supplied inputs.
    weights_name: Optional. The name of the weights object in the manifest bound
      to this function.

  Returns:
    An Orbax Model `ShloFunction`.
  """

  in_named_shardings = getattr(
      exported, "_in_named_shardings", (None,) * len(exported.in_avals)
  )
  out_named_shardings = getattr(
      exported, "_out_named_shardings", (None,) * len(exported.out_avals)
  )

  def _get_op_sharding(nsd, aval):
    sd = None
    if nsd is not None:
      sd = nsd._to_xla_hlo_sharding(aval.ndim)
    return utils.hlo_sharding_to_op_sharding(sd)

  in_shardings_hlo = tuple([
      _get_op_sharding(nsd, aval)
      for nsd, aval in zip(in_named_shardings, exported.in_avals)
  ])
  in_placements = tuple(
      utils.sharding_to_placement(nsd) for nsd in in_named_shardings
  )
  out_shardings_hlo = tuple([
      _get_op_sharding(nsd, aval)
      for nsd, aval in zip(out_named_shardings, exported.out_avals)
  ])
  out_placements = tuple(
      utils.sharding_to_placement(nsd) for nsd in out_named_shardings
  )
  num_user_outputs = exported.out_tree.num_leaves
  out_avals = exported.out_avals[:num_user_outputs]
  out_shardings_hlo = out_shardings_hlo[:num_user_outputs]
  out_placements = out_placements[:num_user_outputs]

  # TODO: properly get the name for the input signature.
  shlo_in_sig, jax_in_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          exported.in_avals,
          in_shardings_hlo,
          model_inputs_layout,
          placements=in_placements,
          tree_def=exported.in_tree,
      )
  )
  # Since jax.ShapeDtypeStruct does not have a name field, we assign
  # names to output tensors specs when converting them to ShloTensorSpec by
  # passing `name_leaves=True`. This ensures that the JAX model
  # produces a NamedSignature for its output (e.g., {'results':
  # ShloTensorSpec(...)}), allowing downstream components (e.g., TF data
  # processors) to reference outputs by name in keyword-based pipelines.
  # This will prevent signature mismatches that could otherwise occur, e.g., the
  # JAX model has the output signature like "model_output: ShloTensorSpec(...)",
  # while the following tf data processor has the input signature like
  # "input_model_output: ShloTensorSpec(...)".
  shlo_out_sig, jax_out_sig_refinements = (
      jax_specific_info._to_shlo_spec_tree_and_refinement_tuple(
          out_avals,
          out_shardings_hlo,
          None,
          placements=out_placements,
          tree_def=exported.out_tree,
          name_leaves=True,
      )
  )
  supplemental_info_ = {}
  jax_specific_info_ = jax_specific_info.JaxSpecificInfo(
      name=exported.fun_name,
      input_spec_refinements=jax_in_sig_refinements,
      output_spec_refinements=jax_out_sig_refinements,
      nr_devices=exported.nr_devices,
      ordered_effects=exported.ordered_effects,
      unordered_effects=exported.unordered_effects,
      disabled_safety_checks=tuple(exported.disabled_safety_checks),
      uses_shape_polymorphism=exported.uses_global_constants,
      in_tree=exported.in_tree,
      out_tree=exported.out_tree,
  )
  supplemental_info_[constants.JAX_SPECIFIC_INFO] = jax_specific_info_

  if xla_compile_options_per_platform is not None:
    supplemental_info_[constants.XLA_COMPILE_OPTIONS] = (
        XlaCompileOptionsSupplementalInfo(xla_compile_options_per_platform)
    )

  shlo_func = shlo_function.ShloFunction(
      input_signature=shlo_in_sig,
      output_signature=shlo_out_sig,
      data_names=model_param_names,
      weights_name=weights_name,
      mlir_module_serialized=exported.mlir_module_serialized,
      calling_convention_version=exported.calling_convention_version,
      module_kept_var_idx=exported.module_kept_var_idx,
      lowering_platforms=exported.platforms,
      supplemental_info=supplemental_info_,
      physical_in_dtypes=tuple(
          utils.get_physical_dtype(utils.aval_dtype(v))
          for v in exported.in_avals
      ),
      physical_out_dtypes=tuple(
          utils.get_physical_dtype(utils.aval_dtype(v)) for v in out_avals
      ),
      num_input_aliased_output=num_input_aliased_output,
  )
  return shlo_func


# Matches the `tf.aliasing_output = <result index> : i32` argument attribute
# that JAX emits for a donated argument it managed to alias to a result.
_ALIASING_OUTPUT_RE = re.compile(r"tf\.aliasing_output\s*=\s*(\d+)\s*:\s*i32")


def _with_donated_output_args(
    apply_fn: typing.ApplyFn,
) -> Callable[[Any, Any, Any], Any]:
  """Wraps `apply_fn` so that it takes donated output buffers as a third arg.

  The extra argument carries no data. It exists only so that `jax.jit` can
  donate it and XLA can alias it to a result, which is what lets an execution
  write its outputs into memory chosen by the caller.
  """

  def fn(params, inputs, donated_outputs):
    del donated_outputs  # Only represents memory, never read.
    return apply_fn(params, inputs)

  return fn


def _flat_output_specs(
    apply_fn: typing.ApplyFn, args_spec: Sequence[Any]
) -> tuple[jax.ShapeDtypeStruct, ...]:
  """Returns one spec per output leaf of `apply_fn`, in output order.

  The caller uses these as the donated output buffer specs: one donor per
  output, typed like that output. The flat order matters, because it is the
  order in which XLA matches donated arguments to results.

  The specs deliberately carry only shape and dtype, matching how the real
  input specs are built. XLA matches a donor to a result by per-shard shape,
  dtype and memory kind, so pinning a sharding or layout here that disagreed
  with the one the jit picks for the result would silently cost us the
  aliasing.

  Args:
    apply_fn: The function whose outputs are to be described.
    args_spec: The specs of `apply_fn`'s own arguments, with concrete shapes.

  Returns:
    One spec per output leaf, in output order.
  """
  out_specs = jax.eval_shape(apply_fn, *args_spec)
  return tuple(
      jax.ShapeDtypeStruct(leaf.shape, leaf.dtype)
      for leaf in jax.tree.leaves(out_specs)
  )


def _assert_outputs_are_aliased(
    exported: jax_export.Exported, num_input_aliased_output: int
) -> None:
  """Checks that every output really was aliased to a donated argument.

  JAX drops donation silently in several situations rather than failing: when
  the donated argument's shard shape or layout does not match the result it
  would alias, when `out_shardings` are unspecified on a multi-device mesh, and
  when `jax_debug_nans` is set. In each case the argument is still reported as
  donatable by the runtime (`jax.buffer_donor` and `tf.aliasing_output` are
  indistinguishable there), but nothing is written in place, so the whole point
  of the export is lost with no diagnostic. Catch it here instead.

  Args:
    exported: The freshly exported function.
    num_input_aliased_output: The number of outputs that should be aliased.

  Raises:
    ValueError: If the exported module does not alias every output.
  """
  aliased = {
      int(match)
      for match in _ALIASING_OUTPUT_RE.findall(exported.mlir_module())
  }
  expected = set(range(num_input_aliased_output))
  if aliased != expected:
    raise ValueError(
        "Output donation was requested, but the exported module aliases"
        f" outputs {sorted(aliased)} instead of {sorted(expected)}. JAX drops"
        " donation silently when the donated buffer cannot back the output it"
        " is matched to. The usual causes are unspecified `out_shardings` on a"
        " multi-device mesh, a layout the result cannot use, and"
        " `jax_debug_nans` being enabled."
    )


def _check_args_specs_are_concrete(
    args_specs_list: Sequence[Sequence[Any]],
) -> None:
  """Rejects output donation over symbolic input shapes.

  Donation is only served from AOT executables, and the AOT compiler compiles
  the exported StableHLO as-is: a module with symbolic dimensions fails there
  with an XLA error that says nothing about why. The JIT path, which could
  refine the shapes, rejects donating models. So a donating model exported with
  symbolic shapes could not be served at all, and it is rejected here with a
  message that says what to do instead.

  Args:
    args_specs_list: One `(params_spec, inputs_spec)` per function to export.

  Raises:
    ValueError: If any spec has a symbolic dimension.
  """
  for args_spec in args_specs_list:
    for path, leaf in jax.tree_util.tree_leaves_with_path(args_spec):
      shape = getattr(leaf, "shape", ())
      if any(jax_export.is_symbolic_dim(d) for d in shape):
        raise ValueError(
            "Output donation requires concrete input shapes, but"
            f" {jax.tree_util.keystr(path)} has shape {shape}. Donation is"
            " only served from AOT executables, which are compiled for"
            " concrete shapes. Set `input_polymorphic_shape_symbol_values` to"
            " export one concrete function per value."
        )


def prepare_output_donation(
    apply_fn: typing.ApplyFn,
    args_specs_list: Sequence[Sequence[Any]],
) -> tuple[Callable[[Any, Any, Any], Any], list[tuple[Any, ...]], int]:
  """Sets up `apply_fn` to be exported with its outputs in donated buffers.

  Donation is per positional argument, so the donated buffers go in an argument
  of their own: one donor per output leaf, typed like that output. That is what
  keeps `jax.jit` from matching a real input to an output and leaving a donated
  buffer unaliased. `keep_unused` is required because the function never reads
  them, and JAX would otherwise prune them away along with their aliasing.

  Pass the results to `convert` as `fun_jax`, `args_spec` (one call per entry)
  and `num_input_aliased_output`.

  Args:
    apply_fn: The `(params, inputs)` function to be exported.
    args_specs_list: One `(params_spec, inputs_spec)` per function to export,
      e.g. one per concretized input shape. Every shape must be concrete:
      polymorphic shapes must be concretized first.

  Returns:
    The function to export, `args_specs_list` with the donated output specs
    appended to each entry, and the number of donated output arguments.

  Raises:
    ValueError: If any input shape is symbolic.
  """
  _check_args_specs_are_concrete(args_specs_list)
  donating_args_specs_list = [
      (*args_spec, _flat_output_specs(apply_fn, args_spec))
      for args_spec in args_specs_list
  ]
  num_input_aliased_output = len(donating_args_specs_list[0][-1])
  donating_fn = jax.jit(
      _with_donated_output_args(apply_fn),
      donate_argnums=2,
      keep_unused=True,
  )
  return donating_fn, donating_args_specs_list, num_input_aliased_output


def _infer_num_weight_leaves(
    args_spec: Sequence[Any],
    kwargs_spec: Mapping[str, Any],
    model_param_names: Sequence[str] | None,
    weights_name: str | None,
    num_input_aliased_output: int,
) -> int:
  """Infers leading weight leaf count in `(args_spec, kwargs_spec)`."""
  if model_param_names is not None:
    return len(model_param_names)
  if weights_name is not None and args_spec:
    return len(jax.tree.leaves(args_spec[0]))
  num_user_pos_args = len(args_spec) - (
      1 if num_input_aliased_output > 0 else 0
  )
  if num_user_pos_args >= 2 or (num_user_pos_args == 1 and bool(kwargs_spec)):
    return len(jax.tree.leaves(args_spec[0]))
  return 0


def convert(
    fun_jax: Callable[..., Any],
    args_spec: Sequence[Any],
    kwargs_spec: dict[str, Any],
    jax_export_kwargs: Mapping[str, Any] | None = None,
    xla_compile_options_per_platform: (
        manifest_pb2.CompileOptionsProtoMap | None
    ) = None,
    model_param_names: Sequence[str] | None = None,
    model_inputs_layout: Sequence[jax_layout.Layout | None] | None = None,
    platforms: Sequence[manifest_pb2.Platform] | None = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
    num_input_aliased_output: int = 0,
    weights_name: str | None = None,
    *,
    enable_input_r1_optimization: bool = False,
    enable_output_r1_optimization: bool = False,
    max_transpose_channels: int = 0,
    num_weight_leaves: int | None = None,
) -> shlo_function.ShloFunction:
  """Converts a JAX function to an Orbax Model `ShloFunction`.

  For example usage of `convert_to_shlo_fn`, please refer to the
  main_lib_test.py file.

  Args:
    fun_jax: The JAX function to be converted. It should be a jitted function.
      If it's not, we will jit it by simply `jax.jit(fun_jax)`, which means
      `fun_jax` can not contain any sharding annotations that are not
      "FullyReplicated".
    args_spec: A sequence of pytrees of {class}`jax.ShapeDtypeStruct`, or values
      with `.shape` and `.dtype` attributes, or result of calling
      `jax_export.symbolic_args_specs()`. These will be used to trace the
      function in jax export. As a result, the ordering of params and input
      within the specs here should match the jax function that user provides.
    kwargs_spec: A dictionary of keyword arguments to pytrees of
      {class}`jax.ShapeDtypeStruct`, or values with `.shape` and `.dtype`
      attributes, or result of calling `jax_export.symbolic_args_specs()`. These
      will be used to trace the function in jax export.
    jax_export_kwargs: Keyword arguments for `jax.export.export`. See
      https://docs.jax.dev/en/latest/_autosummary/jax.export.export.html#jax.export.export.
    xla_compile_options_per_platform: XLA compile options to be saved in the
      model artifact, to ensure XLA compilation consistency and reproducibility
      between export time and serving time. Each map entry corresponds to a
      platform type (e.g. TPU, GPU, etc.).
    model_param_names: Optional. A list of the model parameter names in the
      dot-separated key path format (e.g. "params.key.subkey"). If provided,
      only these parameters will be loaded from the checkpoint when the function
      is executed.
    model_inputs_layout: Optional. A sequence of `jax_layout.Layout | None`
      entries specifying the layouts for the model inputs (weights and user
      inputs).
    platforms: Optional sequence of platforms for compatibility.
    native_serialization_disabled_checks: Optional disabled checks for
      compatibility.
    num_input_aliased_output: The number of trailing entries of `args_spec` that
      are donated output buffers rather than caller-supplied inputs. `fun_jax`
      must already be jitted with those arguments donated.
    weights_name: Optional. The name of the weights object in the manifest bound
      to this function.
    enable_input_r1_optimization: If True, flattens eligible static rank >= 2
      user inputs to 1-D at the StableHLO entry point and reshapes them back
      inside the function. Enabling any export-time TPU I/O optimization makes
      export-time optimization authoritative, so the serving runtime skips its
      legacy `StablehloTpuOptimizer` MLIR passes for the exported function.
    enable_output_r1_optimization: If True, flattens eligible static rank >= 2
      outputs to 1-D before returning from the StableHLO function.
    max_transpose_channels: If > 0, transposes eligible 4-D NHWC user inputs
      with `C <= max_transpose_channels` and `C < W` to NCHW at the StableHLO
      entry point and transposes them back to NHWC on device.
    num_weight_leaves: Optional explicit count of leading flattened weight
      leaves in `(args_spec, kwargs_spec)`. If None, inferred from
      `model_param_names`, `weights_name`, and `(args_spec, kwargs_spec)`.

  Returns:
    An Orbax Model `ShloFunction`.

  Raises:
    ValueError: If `enable_output_r1_optimization` is enabled alongside output
      donation (`num_input_aliased_output > 0`), or if output donation fails to
      alias all outputs.
  """
  if not hasattr(fun_jax, "trace"):
    fun_jax = jax.jit(fun_jax)

  jax_export_kwargs = dict(jax_export_kwargs or {})
  if platforms is not None:
    jax_export_kwargs["platforms"] = platforms
  if native_serialization_disabled_checks:
    jax_export_kwargs["disabled_checks"] = native_serialization_disabled_checks
  if not jax_export_kwargs.get("platforms", None):
    jax_export_kwargs["platforms"] = ["tpu"]
  elif "platforms" in jax_export_kwargs and jax_export_kwargs["platforms"]:
    jax_export_kwargs["platforms"] = [
        manifest_pb2.Platform.Name(p).lower() if isinstance(p, int) else str(p)
        for p in jax_export_kwargs["platforms"]
    ]

  exported_creator = jax_export.export(
      fun_jax,  # pyrefly: ignore[bad-argument-type]
      **jax_export_kwargs,
  )

  utils.assert_not_in_jax_transformation()

  exported = exported_creator(*args_spec, **kwargs_spec)
  if num_input_aliased_output:
    _assert_outputs_are_aliased(exported, num_input_aliased_output)
  if num_weight_leaves is not None:
    resolved_num_weight_leaves = num_weight_leaves
  else:
    resolved_num_weight_leaves = _infer_num_weight_leaves(
        args_spec,
        kwargs_spec,
        model_param_names,
        weights_name,
        num_input_aliased_output,
    )
  opt_plan = tpu_io_optimization.build_tpu_io_optimization_plan(
      fun_jax,
      args_spec,
      kwargs_spec,
      exported,
      num_weight_leaves=resolved_num_weight_leaves,
      model_inputs_layout=model_inputs_layout,
      num_input_aliased_output=num_input_aliased_output,
      enable_input_r1_optimization=enable_input_r1_optimization,
      enable_output_r1_optimization=enable_output_r1_optimization,
      max_transpose_channels=max_transpose_channels,
  )
  if opt_plan is None:
    return jax_exported_to_shlo_fn(
        exported,
        xla_compile_options_per_platform,
        model_param_names=model_param_names,
        model_inputs_layout=model_inputs_layout,
        num_input_aliased_output=num_input_aliased_output,
        weights_name=weights_name,
    )

  unoptimized_shlo_fn = jax_exported_to_shlo_fn(
      exported,
      xla_compile_options_per_platform,
      model_param_names=model_param_names,
      model_inputs_layout=model_inputs_layout,
      num_input_aliased_output=num_input_aliased_output,
      weights_name=weights_name,
  )
  opt_exported_creator = jax_export.export(
      opt_plan.wrapped_fn,  # pyrefly: ignore[bad-argument-type]
      **jax_export_kwargs,
  )
  opt_exported = opt_exported_creator(
      *opt_plan.args_spec, **opt_plan.kwargs_spec
  )
  if num_input_aliased_output:
    _assert_outputs_are_aliased(opt_exported, num_input_aliased_output)

  opt_shlo_fn = jax_exported_to_shlo_fn(
      opt_exported,
      xla_compile_options_per_platform,
      model_param_names=model_param_names,
      model_inputs_layout=model_inputs_layout,
      num_input_aliased_output=num_input_aliased_output,
      weights_name=weights_name,
  )
  physical_in_sig = opt_shlo_fn.input_signature
  physical_out_sig = opt_shlo_fn.output_signature
  if opt_plan.transposed_input_leaf_indices:
    physical_in_sig = (
        tpu_io_optimization.apply_host_transpose_to_full_input_sig(
            physical_in_sig,
            opt_plan.transposed_input_leaf_indices,
        )
    )
  opt_shlo_fn.input_signature = unoptimized_shlo_fn.input_signature
  opt_shlo_fn.output_signature = unoptimized_shlo_fn.output_signature
  if constants.JAX_SPECIFIC_INFO in unoptimized_shlo_fn.supplemental_info:
    merged_supplemental = dict(opt_shlo_fn.supplemental_info)
    merged_supplemental[constants.JAX_SPECIFIC_INFO] = (
        unoptimized_shlo_fn.supplemental_info[constants.JAX_SPECIFIC_INFO]
    )
    opt_shlo_fn.supplemental_info = merged_supplemental
  if opt_plan.has_input_optimization:
    opt_shlo_fn.optimized_input_signature = physical_in_sig
  if opt_plan.has_output_optimization:
    opt_shlo_fn.optimized_output_signature = physical_out_sig
  return opt_shlo_fn


def convert_path_to_value(
    path: str,
    mime_type: str | None = None,
    version: str | None = None,
    loader_type: manifest_pb2.LoaderType | None = None,
) -> value.ExternalValue:
  return value.ExternalValue(
      data=manifest_pb2.UnstructuredData(
          file_system_location=manifest_pb2.FileSystemLocation(
              string_path=path
          ),
          mime_type=mime_type,
          version=version,
      ),
      loader_type=loader_type,
  )


def convert_apply_fn_bundle_to_shlo(
    apply_fn_bundle: typing.ApplyFnBundle,
    params: jaxtyping.PyTree | None,
    input_signature: jaxtyping.PyTree[shlo_type.ShloTensorSpec],
    model_param_names: Sequence[str] | None = None,
    jax_mesh: jax.sharding.Mesh | None = None,
    weights_name: str | None = None,
) -> Sequence[shlo_function.ShloFunction]:
  """Converts an `ApplyFnBundle` into one or more `ShloFunction` instances.

  Args:
    apply_fn_bundle: Bundle containing `apply_fn`, polymorphic shape settings,
      output donation flag, and TPU I/O optimization flags.
    params: Model parameter PyTree (or `None` when weightless).
    input_signature: Caller-facing user input `ShloTensorSpec` PyTree.
    model_param_names: Optional dot-separated parameter names matching `params`
      leaves.
    jax_mesh: Optional JAX device mesh used for XLA compile options generation.
    weights_name: Optional manifest weights key bound to the exported functions.

  Returns:
    A tuple of exported `ShloFunction` objects (one per concretized shape bucket
    when `input_polymorphic_shape_symbol_values` is provided, else a single-item
    tuple).

  Raises:
    ValueError: If incompatible options are configured (e.g.,
      `use_input_aliasing=True` with `enable_output_r1_optimization=True` or
      unconcretized symbolic shapes).
  """
  args_specs = jax_export.symbolic_args_specs(
      args=utils.to_jax_spec((params, input_signature)),
      shapes_specs=(
          "...",
          apply_fn_bundle.input_polymorphic_shapes,
      ),
      constraints=apply_fn_bundle.polymorphic_constraints,
  )
  if apply_fn_bundle.input_polymorphic_shape_symbol_values is not None:
    args_specs_list = utils._concretize_polymorphic_shapes(
        args_specs,
        apply_fn_bundle.input_polymorphic_shape_symbol_values,
    )
  else:
    args_specs_list = [args_specs]

  fun_jax = apply_fn_bundle.apply_fn
  num_input_aliased_output = 0
  if apply_fn_bundle.use_input_aliasing:
    if apply_fn_bundle.enable_output_r1_optimization:
      raise ValueError(
          "Output R1 optimization is incompatible with output donation"
          " (use_input_aliasing=True)."
      )
    fun_jax, args_specs_list, num_input_aliased_output = (
        prepare_output_donation(apply_fn_bundle.apply_fn, args_specs_list)
    )

  num_weight_leaves = (
      len(jax.tree.leaves(params)) if params is not None else 0
  )
  compile_options_per_platform = (
      compile_options_util.generate_xla_compile_options(
          native_serialization_platforms=apply_fn_bundle.jax_export_kwargs.get(
              "platforms", None
          ),
          jax_mesh=jax_mesh,
      )
  )
  return tuple(
      convert(
          fun_jax=fun_jax,
          args_spec=args_spec,
          kwargs_spec={},
          jax_export_kwargs=apply_fn_bundle.jax_export_kwargs,
          xla_compile_options_per_platform=compile_options_per_platform,
          model_param_names=model_param_names,
          num_input_aliased_output=num_input_aliased_output,
          weights_name=weights_name,
          enable_input_r1_optimization=(
              apply_fn_bundle.enable_input_r1_optimization
          ),
          enable_output_r1_optimization=(
              apply_fn_bundle.enable_output_r1_optimization
          ),
          max_transpose_channels=apply_fn_bundle.max_transpose_channels,
          num_weight_leaves=num_weight_leaves,
      )
      for args_spec in args_specs_list
  )

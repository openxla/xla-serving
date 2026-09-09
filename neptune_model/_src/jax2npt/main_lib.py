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
from collections.abc import Mapping
import dataclasses
from typing import Any, Dict, Sequence

import jax
from jax import export as jax_export
from jax.experimental import layout as jax_layout
import jaxtyping

from neptune_model._src.core import compile_options_util
from neptune_model._src.core import shlo_function
from neptune_model._src.core import shlo_type
from neptune_model._src.core import unstructured_data
from neptune_model._src.core import value
from neptune_model._src.jax2npt import constants
from neptune_model._src.jax2npt import jax_specific_info
from neptune_model._src.jax2npt import typing
from neptune_model._src.jax2npt import utils
from neptune_model.protos import manifest_pb2


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
    model_inputs_layout: Sequence[jax_layout.Layout] | None = None,
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
    model_inputs_layout: Optional. A sequence of `jax_layout.Layout` objects
      specifying the layouts for the model inputs.

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
          exported.in_tree,
          placements=in_placements,
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
          exported.out_tree,
          placements=out_placements,
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
  )
  return shlo_func


def convert(
    fun_jax: typing.ApplyFn,
    args_spec: Sequence[Any],
    kwargs_spec: Dict[str, Any],
    jax_export_kwargs: Mapping[str, Any] | None = None,
    xla_compile_options_per_platform: (
        manifest_pb2.CompileOptionsProtoMap | None
    ) = None,
    model_param_names: Sequence[str] | None = None,
    model_inputs_layout: Sequence[jax_layout.Layout] | None = None,
    platforms: Sequence[manifest_pb2.Platform] | None = None,
    native_serialization_disabled_checks: Sequence[
        jax_export.DisabledSafetyCheck
    ] = (),
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
    model_inputs_layout: Optional. A sequence of `jax_layout.Layout` objects
      specifying the layouts for the model inputs (weights and user inputs)
    platforms: Optional sequence of platforms for compatibility.
    native_serialization_disabled_checks: Optional disabled checks for
      compatibility.

  Returns:
    An Orbax Model `ShloFunction`.
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
  return jax_exported_to_shlo_fn(
      exported,
      xla_compile_options_per_platform,
      model_param_names=model_param_names,
      model_inputs_layout=model_inputs_layout,
  )


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
) -> Sequence[shlo_function.ShloFunction]:
  """Converts an ApplyFnBundle to SHLO functions."""
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
          fun_jax=apply_fn_bundle.apply_fn,
          args_spec=args_spec,
          kwargs_spec={},
          jax_export_kwargs=apply_fn_bundle.jax_export_kwargs,
          xla_compile_options_per_platform=compile_options_per_platform,
          model_param_names=model_param_names,
      )
      for args_spec in args_specs_list
  )

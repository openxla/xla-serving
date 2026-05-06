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

from neptune_model._src.core import shlo_function
from neptune_model._src.core import unstructured_data
from neptune_model._src.core import value
from neptune_model._src.jax2npt import constants
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
  # Fallback to named shardings if HLO shardings are missing (JAX 0.4.31+)
  has_named_shardings = getattr(exported, "_has_named_shardings", False)
  in_named_shardings = getattr(
      exported, "_in_named_shardings", (None,) * len(exported.in_avals)
  )
  out_named_shardings = getattr(
      exported, "_out_named_shardings", (None,) * len(exported.out_avals)
  )

  def _get_op_sharding(sd, nsd, aval):
    if sd is None and has_named_shardings and nsd is not None:
      sd = nsd._to_xla_hlo_sharding(aval.ndim)
    return utils.hlo_sharding_to_op_sharding(sd)

  in_shardings_hlo = tuple([
      _get_op_sharding(sd, nsd, aval)
      for sd, nsd, aval in zip(
          exported.in_shardings_hlo, in_named_shardings, exported.in_avals
      )
  ])
  out_shardings_hlo = tuple([
      _get_op_sharding(sd, nsd, aval)
      for sd, nsd, aval in zip(
          exported.out_shardings_hlo, out_named_shardings, exported.out_avals
      )
  ])
  # TODO: properly get the name for the input signature.
  shlo_in_sig = utils.to_shlo_spec_tree(
      exported.in_avals,
      in_shardings_hlo,
      model_inputs_layout,
      exported.in_tree,
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
  shlo_out_sig = utils.to_shlo_spec_tree(
      exported.out_avals,
      out_shardings_hlo,
      None,
      exported.out_tree,
      name_leaves=True,
  )
  supplemental_info_ = {}

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
          utils.get_physical_dtype(utils.aval_dtype(v))
          for v in exported.out_avals
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

  Returns:
    An Orbax Model `ShloFunction`.
  """
  if not hasattr(fun_jax, "trace"):
    fun_jax = jax.jit(fun_jax)

  exported_creator = jax_export.export(
      fun_jax,
      **(jax_export_kwargs or {}),
  )

  utils.assert_not_in_jax_transformation()

  exported = exported_creator(*args_spec, **kwargs_spec)
  return jax_exported_to_shlo_fn(
      exported,
      model_param_names=model_param_names,
      model_inputs_layout=model_inputs_layout,
  )


def convert_path_to_value(
    path: str, mime_type: str | None = None, version: str | None = None
) -> value.ExternalValue:
  return value.ExternalValue(
      data=manifest_pb2.UnstructuredData(
          file_system_location=manifest_pb2.FileSystemLocation(
              string_path=path
          ),
          mime_type=mime_type,
          version=version,
      ),
  )

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

"""Utilities for building Manifest proto."""

from collections.abc import Iterable, Iterator, Mapping, Sequence
import dataclasses
import inspect
import io
import pathlib
import re
from typing import Any, Protocol, TypeGuard

from absl import logging

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import constants
from xla_saved_model._src.core import device_assignment
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import function
from xla_saved_model._src.core import saveable
from xla_saved_model._src.core import serializable_function
from xla_saved_model._src.core import shlo_function
from xla_saved_model._src.core import type_proto_utils
from xla_saved_model._src.core import unstructured_data
from xla_saved_model._src.core import value


def resolve_relative_path(
    fsl: manifest_pb2.FileSystemLocation,
    manifest: manifest_pb2.Manifest,
) -> str:
  """Resolves relative file path string from a FileSystemLocation proto."""
  if not fsl.HasField("string_path") and not fsl.HasField("frequent_path_idx"):
    raise ValueError(
        "FileSystemLocation has neither string_path nor frequent_path_idx."
    )
  if fsl.HasField("string_path"):
    return fsl.string_path
  idx = fsl.frequent_path_idx
  if idx < 0:
    raise ValueError(f"Invalid frequent_path_idx {idx} in manifest.")
  try:
    return manifest.frequent_paths[idx]
  except IndexError:
    raise ValueError(
        f"Invalid frequent_path_idx {idx} in manifest."
    ) from None


def get_functions(
    manifest: manifest_pb2.Manifest,
) -> Iterator[tuple[str, manifest_pb2.Function]]:
  """Yields (object_name, function) from Manifest TopLevelObjects."""
  for name, obj in manifest.objects.items():
    if obj.HasField("function"):
      yield name, obj.function
    elif obj.HasField("poly_fn"):
      num_fns = len(obj.poly_fn.concrete_functions)
      for i, fn in enumerate(obj.poly_fn.concrete_functions):
        yield f"{name}_{i}" if num_fns > 1 else name, fn


def read_unstructured_data_bytes(
    data: manifest_pb2.UnstructuredData,
    source_dir: str,
    manifest: manifest_pb2.Manifest | None = None,
    recorded_external_paths: set[str] | None = None,
) -> bytes:
  """Reads the payload of an `UnstructuredData`, inlined or on file.

  Args:
    data: The data to read.
    source_dir: Directory the data's file system location is relative to.
    manifest: The manifest holding `data`, needed to resolve a
      `frequent_path_idx`.
    recorded_external_paths: If given, the relative path of a file read is
      added to it.

  Returns:
    The payload bytes.

  Raises:
    ValueError: If the data is unset, or its path is absolute, contains '..' or
      cannot be resolved.
    FileNotFoundError: If the referenced file does not exist.
  """
  match data.WhichOneof("data"):
    case "inlined_bytes":
      return data.inlined_bytes
    case "inlined_string":
      return data.inlined_string.encode("utf-8")
    case "file_system_location":
      pass
    case case_name:
      raise ValueError(f"Unsupported UnstructuredData data field: {case_name}")

  fsl = data.file_system_location
  if fsl.HasField("frequent_path_idx") and manifest is None:
    raise ValueError("A manifest is needed to resolve frequent_path_idx.")
  rel_path = resolve_relative_path(fsl, manifest or manifest_pb2.Manifest())
  if pathlib.PurePath(rel_path).is_absolute():
    raise ValueError(
        f'File system location must be relative, got "{rel_path}"'
    )
  if ".." in pathlib.PurePath(rel_path).parts:
    raise ValueError(f"File system location cannot contain '..': {rel_path}")
  path = str(pathlib.Path(source_dir) / rel_path)
  if not file_utils.exists(path):
    raise FileNotFoundError(f'Referenced file does not exist: "{path}"')
  if recorded_external_paths is not None:
    recorded_external_paths.add(rel_path)

  with file_utils.open_file(path, "rb") as f:
    if fsl.offset_in_byte:
      f.seek(
          fsl.offset_in_byte,
          io.SEEK_END if fsl.offset_in_byte < 0 else io.SEEK_SET,
      )
    return (
        f.read(fsl.size_in_byte) if fsl.HasField("size_in_byte") else f.read()
    )


def extract_stablehlo_payload(
    shlo_proto: manifest_pb2.UnstructuredData,
    manifest: manifest_pb2.Manifest,
    source_dir: str,
    recorded_external_paths: set[str] | None = None,
) -> bytes:
  """Extracts StableHLO payload bytes and optionally records external paths."""
  if shlo_proto.WhichOneof("data") is None:
    raise ValueError(
        "Cannot recompile model: StableHLO payload was stripped in a prior"
        " AOT-only export."
    )
  return read_unstructured_data_bytes(
      shlo_proto, source_dir, manifest, recorded_external_paths
  )


# Note: Single-device portable executables decouple binary from host device IDs.
# Multi-device execution in IFRT/XLA requires topology-bound executables.
@dataclasses.dataclass(frozen=True)
class CompiledArtifact:
  """The resulting artifact from an Ahead-Of-Time (AOT) compilation hook."""

  pjrt_executable_bytes: bytes
  kept_var_idx: Sequence[int]
  target_topology_label: str
  executable_version: bytes


class CompilerHook(Protocol):
  """Protocol for an Ahead-Of-Time (AOT) compiler callback.

  Invoked during manifest generation to compile serialized StableHLO modules
  into hardware-specific executable artifacts (e.g. PJRT executables).
  """

  def __call__(
      self,
      mlir_module_serialized: bytes,
      module_kept_var_idx: Sequence[int],
      num_weights: int,
      /,
  ) -> Iterable[CompiledArtifact]:
    """Compiles a serialized MLIR module into compiled artifacts.

    Args:
      mlir_module_serialized: Serialized StableHLO MLIR bytecode/module.
      module_kept_var_idx: Indices of kept variables/arguments in the module.
      num_weights: Number of weights (data names) passed to the function.

    Returns:
      An iterable of `CompiledArtifact` objects containing compiled executable
      payloads and associated topology metadata.
    """
    ...


def _call_compiler_hook(
    compiler_hook: CompilerHook,
    mlir_module_serialized: bytes,
    module_kept_var_idx: Sequence[int],
    num_weights: int,
    *,
    optimized_at_export: bool,
) -> Iterable[CompiledArtifact]:
  """Invokes `compiler_hook`, passing `optimized_at_export` when supported."""
  try:
    params = inspect.signature(compiler_hook).parameters
  except (TypeError, ValueError):
    params = {}
  hook_any: Any = compiler_hook
  opt_param = params.get("optimized_at_export")
  if opt_param is not None:
    if opt_param.kind == inspect.Parameter.POSITIONAL_ONLY:
      return hook_any(
          mlir_module_serialized,
          module_kept_var_idx,
          num_weights,
          optimized_at_export,
      )
    return hook_any(
        mlir_module_serialized,
        module_kept_var_idx,
        num_weights,
        optimized_at_export=optimized_at_export,
    )
  return compiler_hook(
      mlir_module_serialized, module_kept_var_idx, num_weights
  )


def _build_function(
    fn: function.Function,
    target_dir: str,
    name: str,
    visibility: manifest_pb2.Visibility,
    save_shlo_to_file: bool = False,
    compiler_hook: CompilerHook | None = None,
    include_stablehlo: bool = True,
) -> manifest_pb2.Function:
  """Builds a `manifest_pb2.Function` proto from a `Function` object.

  Args:
    fn: The `Function` object to build into a proto.
    target_dir: Target directory where external payload files are written.
    name: Function name in the manifest.
    visibility: Visibility level (PUBLIC or PRIVATE).
    save_shlo_to_file: If True, writes StableHLO MLIR payload to a separate file
      rather than inlining it in the manifest.
    compiler_hook: Optional `CompilerHook` callback to compile StableHLO modules
      into AOT compiled executables.
    include_stablehlo: If True, includes the StableHLO payload in the exported
      function proto.

  Returns:
    A populated `manifest_pb2.Function` proto.
  """

  fn_proto = manifest_pb2.Function()
  fn_proto.visibility = visibility

  if fn.data_names:
    fn_proto.data_names.extend(fn.data_names)
  if fn.weights_name:
    fn_proto.weights_name = fn.weights_name

  fn_proto.signature.CopyFrom(
      type_proto_utils.to_function_signature_proto(
          fn.input_signature, fn.output_signature
      )
  )

  if isinstance(fn, shlo_function.ShloFunction):
    stable_hlo_body = fn_proto.body.stable_hlo_body
    if include_stablehlo:
      if save_shlo_to_file:
        shlo_data = manifest_pb2.UnstructuredData(
            inlined_bytes=fn.mlir_module_serialized,
            mime_type=constants.SHLO_MIME_TYPE,
            version=constants.SHLO_VERSION,
        )
        filename = unstructured_data.build_relative_filepath_from_extension(
            name, "shlo"
        )
        shlo_file_data = unstructured_data.write_inlined_data_to_file(
            shlo_data, target_dir, filename
        )
        stable_hlo_body.stable_hlo.CopyFrom(shlo_file_data)
      else:
        stable_hlo_body.stable_hlo.inlined_bytes = fn.mlir_module_serialized
        stable_hlo_body.stable_hlo.mime_type = constants.SHLO_MIME_TYPE
        stable_hlo_body.stable_hlo.version = constants.SHLO_VERSION
    stable_hlo_body.calling_convention_version = fn.calling_convention_version

    stable_hlo_body.lowering_platforms.extend(fn.lowering_platforms)

    stable_hlo_body.module_kept_var_idx.extend(fn.module_kept_var_idx)

    stable_hlo_body.num_input_aliased_output = fn.num_input_aliased_output

    if fn.optimized_input_signature is not None:
      stable_hlo_body.optimized_input_signature.CopyFrom(
          type_proto_utils.to_type_proto(fn.optimized_input_signature)
      )
    if fn.optimized_output_signature is not None:
      stable_hlo_body.optimized_output_signature.CopyFrom(
          type_proto_utils.to_type_proto(fn.optimized_output_signature)
      )

    # Serialize and add all supplementals to the function body proto.
    for supp_name, supp in (fn.supplemental_info or {}).items():
      supp = supp.serializable_to_proto()
      supp_proto = supp.proto

      if supp.ext_name:
        filename = unstructured_data.build_relative_filepath_from_extension(
            name + "_" + supp_name + "_supplemental", supp.ext_name
        )
        supp_proto = unstructured_data.write_inlined_data_to_file(
            supp_proto, target_dir, filename
        )
      stable_hlo_body.supplemental_info[supp_name].CopyFrom(supp_proto)

    if compiler_hook:
      num_weights = len(fn.data_names or [])
      safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", name)
      optimized_at_export = (
          fn.optimized_input_signature is not None
          or fn.optimized_output_signature is not None
      )
      artifacts = _call_compiler_hook(
          compiler_hook,
          fn.mlir_module_serialized,
          fn.module_kept_var_idx,
          num_weights,
          optimized_at_export=optimized_at_export,
      )
      for artifact in artifacts:
        safe_label = re.sub(
            r"[^a-zA-Z0-9_-]", "_", artifact.target_topology_label
        )
        filename = f"{safe_name}_{safe_label}_pjrt_executable.bin"
        pjrt_proto = manifest_pb2.UnstructuredData(
            inlined_bytes=artifact.pjrt_executable_bytes,
            mime_type=constants.IFRT_EXECUTABLE_MIME_TYPE,
        )
        pjrt_proto = unstructured_data.write_inlined_data_to_file(
            pjrt_proto, target_dir, filename
        )
        compiled_exec_proto = stable_hlo_body.compiled_executables.add()
        compiled_exec_proto.aot_executable.CopyFrom(pjrt_proto)
        compiled_exec_proto.target_topology_label = (
            artifact.target_topology_label
        )
        compiled_exec_proto.aot_kept_var_idx.extend(artifact.kept_var_idx)
        compiled_exec_proto.executable_version = artifact.executable_version

    # Fail-fast payload validation assertion
    has_stable_hlo = (
        stable_hlo_body.HasField("stable_hlo") if include_stablehlo else False
    )
    has_aot_executables = bool(stable_hlo_body.compiled_executables)

    if not (has_stable_hlo or has_aot_executables):
      raise ValueError(
          f"Function '{name}' contains no executable payload. When"
          " `include_stablehlo=False`, `compiler_hook` must produce at least"
          " one AOT compiled executable artifact."
      )

  elif isinstance(fn, serializable_function.SerializableFunction):
    body_proto = fn.body.proto

    if fn.body.ext_name is not None:
      relative_filepath = (
          unstructured_data.build_relative_filepath_from_extension(
              name, fn.body.ext_name, subfolder=fn.body.subfolder
          )
      )
      body_proto = unstructured_data.write_inlined_data_to_file(
          body_proto, target_dir, relative_filepath
      )

    fn_proto.body.other.CopyFrom(body_proto)

  else:
    raise ValueError(f"Unsupported subclass of `Function`: {type(fn)}")

  return fn_proto


def _build_device_assignment_by_coords_proto(
    device_assignments: Sequence[device_assignment.DeviceAssignment],
) -> manifest_pb2.DeviceAssignmentByCoords:
  """Builds a DeviceAssignmentByCoords proto from device assignments.

  Args:
    device_assignments: A sequence of DeviceAssignment objects.

  Returns:
    A DeviceAssignmentByCoords proto.
  """
  proto = manifest_pb2.DeviceAssignmentByCoords()

  for assignment in device_assignments:
    device_proto = proto.devices.add(id=assignment.id)
    if assignment.coords is not None:
      device_proto.coords.extend(assignment.coords)
    if assignment.core_on_chip is not None:
      device_proto.core_on_chip = assignment.core_on_chip

  return proto


def _is_seq_of_functions(
    obj: saveable.Saveable,
) -> TypeGuard[Sequence[function.Function]]:
  """Checks if the object is a non-empty sequence of `Function`s."""
  return (
      isinstance(obj, Sequence)
      and not isinstance(obj, (str, bytes))
      and bool(obj)
      and all(isinstance(elem, function.Function) for elem in obj)
  )


def build_manifest_proto(
    obm_module: Mapping[str, saveable.Saveable],
    target_dir: str,
    supplementals: Mapping[str, manifest_pb2.UnstructuredData] | None = None,
    visibilities: Mapping[str, manifest_pb2.Visibility] | None = None,
    device_assignments: (
        Sequence[device_assignment.DeviceAssignment] | None
    ) = None,
    save_shlo_to_file: bool = False,
    compiler_hook: CompilerHook | None = None,
    include_stablehlo: bool = True,
) -> manifest_pb2.Manifest:
  """Builds a manifest proto from an OBM module.

  Args:
    obm_module: A mapping from object name to saveable module components.
    target_dir: Target directory where external payload files are written.
    supplementals: Optional global supplemental unstructured data entries.
    visibilities: Optional mapping from object name to visibility.
    device_assignments: Optional sequence of device assignments.
    save_shlo_to_file: If True, writes StableHLO MLIR payloads to separate
      files.
    compiler_hook: Optional `CompilerHook` callback to compile StableHLO modules
      into AOT compiled executables.
    include_stablehlo: If True, includes StableHLO payloads in the manifest.

  Returns:
    A populated `manifest_pb2.Manifest` proto.
  """

  visibilities = visibilities or {}
  manifest = manifest_pb2.Manifest()

  for name, obj in obm_module.items():
    visibility = visibilities.get(name, manifest_pb2.PUBLIC)

    if isinstance(obj, function.Function):
      manifest.objects[name].function.CopyFrom(
          _build_function(
              obj,
              target_dir,
              name,
              visibility,
              save_shlo_to_file,
              compiler_hook,
              include_stablehlo,
          )
      )

    elif _is_seq_of_functions(obj):
      fn_protos = [
          _build_function(
              fn,
              target_dir,
              f"__{name}_{i}",
              visibility,
              save_shlo_to_file,
              compiler_hook,
              include_stablehlo,
          )
          for i, fn in enumerate(obj)
      ]
      manifest.objects[name].poly_fn.concrete_functions.extend(fn_protos)

    elif isinstance(obj, value.ExternalValue):
      if obj.type:
        raise NotImplementedError(
            "Serializing `ExternalValue.type` is not supported yet."
        )
      manifest.objects[name].value.external.data.CopyFrom(obj.data)
      if obj.loader_type is not None:
        manifest.objects[name].value.external.loader_type = obj.loader_type

    else:
      raise ValueError(
          f"Unsupported module value type at key {name}: {type(obj)}"
      )

  if supplementals:
    for name, supp in supplementals.items():
      manifest.supplemental_info[name].CopyFrom(supp)

  if device_assignments:
    manifest.device_assignment_by_coords.CopyFrom(
        _build_device_assignment_by_coords_proto(device_assignments)
    )

  logging.vlog(3, "Final manifest proto: %s", manifest)

  return manifest

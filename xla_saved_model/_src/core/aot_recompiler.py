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

"""Ahead-Of-Time (AOT) Recompiler for XLA SavedModels."""

from collections.abc import Sequence
import pathlib
import re
import time

from absl import logging

from xla_saved_model.protos import manifest_pb2
from xla_saved_model._src.core import constants
from xla_saved_model._src.core import file_utils
from xla_saved_model._src.core import manifest_utils
from xla_saved_model._src.core import offline_compiler
from xla_saved_model._src.core import persistence_lib

TargetConfig = offline_compiler.TargetConfig


def _resolve_output_version_path(
    output_model_path: str | pathlib.Path,
    explicit_version: int | None = None,
    auto_timestamp: bool = True,
) -> tuple[str, str]:
  """Resolves (base_dir, target_version_dir) for output model export."""
  path = pathlib.Path(output_model_path)
  is_version = path.name.isdigit()
  base_dir = path.parent if is_version else path
  if explicit_version is not None:
    target_dir = base_dir / str(explicit_version)
  elif is_version:
    target_dir = path
  elif auto_timestamp:
    target_dir = base_dir / str(int(time.time()))
  else:
    base_dir = path.parent
    target_dir = path
  return str(base_dir), str(target_dir)


def _update_manifest_with_compiled_artifacts(
    obj_name: str,
    fn: manifest_pb2.Function,
    manifest: manifest_pb2.Manifest,
    artifacts: Sequence[manifest_utils.CompiledArtifact],
    target_dir: str,
) -> None:
  """Writes compiled executables to disk and updates manifest entries."""
  if not artifacts:
    return

  if not fn.body.HasField("stable_hlo_body"):
    raise ValueError(
        f"Function '{obj_name}' does not contain a stable_hlo_body."
    )

  new_labels = {a.target_topology_label for a in artifacts}
  body = fn.body.stable_hlo_body
  target_path = pathlib.Path(target_dir)

  # Garbage collect old binaries if replacing an existing target_topology_label
  preserved = []
  for entry in body.compiled_executables:
    if entry.target_topology_label in new_labels:
      if entry.aot_executable.HasField("file_system_location"):
        path = manifest_utils.resolve_relative_path(
            entry.aot_executable.file_system_location, manifest
        )
        old_full_path = str(target_path / path)
        if file_utils.exists(old_full_path):
          file_utils.remove(old_full_path)
    else:
      preserved.append(entry)

  del body.compiled_executables[:]
  body.compiled_executables.extend(preserved)

  sanitized_obj = re.sub(r"[^a-zA-Z0-9_-]", "_", obj_name)
  for artifact in artifacts:
    sanitized_label = re.sub(
        r"[^a-zA-Z0-9_-]", "_", artifact.target_topology_label
    )
    filename = f"{sanitized_obj}_{sanitized_label}_pjrt_executable.bin"

    with file_utils.open_file(str(target_path / filename), "wb") as f:
      f.write(artifact.pjrt_executable_bytes)

    entry = body.compiled_executables.add()
    entry.aot_executable.file_system_location.string_path = (
        pathlib.PurePosixPath(filename).as_posix()
    )
    entry.aot_executable.mime_type = constants.IFRT_EXECUTABLE_MIME_TYPE
    entry.executable_version = artifact.executable_version
    entry.target_topology_label = artifact.target_topology_label
    entry.aot_kept_var_idx.extend(artifact.kept_var_idx)


def _recompile_to_temp(
    input_model_path: str,
    tmp_dir: str,
    configs: Sequence[TargetConfig],
    include_stablehlo: bool,
) -> None:
  """Executes recompilation staging inside tmp_dir."""
  logging.info(
      "Preserving model assets from %s to %s", input_model_path, tmp_dir
  )
  file_utils.copytree(input_model_path, tmp_dir)
  tmp_path = pathlib.Path(tmp_dir)

  manifest = persistence_lib.load(tmp_dir)
  shlo_rel_paths_to_delete: set[str] = set()
  compiler_hook = offline_compiler.create_offline_compiler_hook(configs)

  for obj_name, fn in manifest_utils.get_functions(manifest):
    if not fn.body.HasField("stable_hlo_body"):
      continue
    body = fn.body.stable_hlo_body
    if not body.HasField("stable_hlo"):
      raise ValueError(
          f"Cannot recompile function '{obj_name}' because the input model"
          " was exported as an AOT-only artifact (StableHLO payload was"
          " stripped)."
      )

    mlir_payload = manifest_utils.extract_stablehlo_payload(
        body.stable_hlo, manifest, tmp_dir, shlo_rel_paths_to_delete
    )
    optimized_at_export = body.HasField(
        "optimized_input_signature"
    ) or body.HasField("optimized_output_signature")
    artifacts = list(
        compiler_hook(
            mlir_payload,
            body.module_kept_var_idx,
            len(fn.data_names),
            optimized_at_export=optimized_at_export,
        )
    )
    _update_manifest_with_compiled_artifacts(
        obj_name, fn, manifest, artifacts, tmp_dir
    )

  if not include_stablehlo:
    logging.info("Stripping StableHLO payloads from output manifest")
    for _, fn in manifest_utils.get_functions(manifest):
      if fn.body.HasField("stable_hlo_body"):
        fn.body.stable_hlo_body.ClearField("stable_hlo")
    for rel_path in shlo_rel_paths_to_delete:
      full_path = str(tmp_path / rel_path)
      if file_utils.exists(full_path):
        file_utils.remove(full_path)

  manifest_path = str(tmp_path / constants.MANIFEST_FILE_PATH)
  with file_utils.open_file(manifest_path, "wb") as f:
    f.write(manifest.SerializeToString())


def _promote_temp_dir_to_target(tmp_dir: str, target_version_dir: str) -> None:
  """Promotes staged tmp_dir to target_version_dir atomically."""
  if file_utils.exists(target_version_dir):
    file_utils.rmtree(target_version_dir)

  try:
    pathlib.Path(tmp_dir).replace(target_version_dir)
  except OSError:
    file_utils.copytree(tmp_dir, target_version_dir)
    file_utils.rmtree(tmp_dir)


def recompile(
    input_model_path: str,
    output_model_path: str,
    target_configs: Sequence[TargetConfig] | TargetConfig,
    *,
    include_stablehlo: bool = True,
    version: int | None = None,
    auto_timestamp: bool = True,
    overwrite: bool = False,
) -> str:
  """Recompiles an exported XLA SavedModel for target accelerator topologies."""
  configs = (
      [target_configs]
      if isinstance(target_configs, TargetConfig)
      else list(target_configs)
  )
  if not configs:
    raise ValueError("target_configs must not be empty.")

  if not file_utils.exists(input_model_path):
    raise FileNotFoundError(
        f"Input model path does not exist: {input_model_path}"
    )

  base_dir, target_version_dir = _resolve_output_version_path(
      output_model_path, explicit_version=version, auto_timestamp=auto_timestamp
  )
  input_resolved = pathlib.Path(input_model_path).resolve()
  if input_resolved in (
      pathlib.Path(output_model_path).resolve(),
      pathlib.Path(base_dir).resolve(),
      pathlib.Path(target_version_dir).resolve(),
  ):
    raise ValueError(
        "input_model_path and output_model_path cannot be identical."
    )

  if file_utils.exists(target_version_dir) and not overwrite:
    raise FileExistsError(
        f"Output version directory already exists: {target_version_dir}"
    )

  if not file_utils.exists(base_dir):
    file_utils.mkdir_p(base_dir)

  target_path = pathlib.Path(target_version_dir)
  tmp_dir = str(target_path.with_name(f"{target_path.name}.tmp"))
  if file_utils.exists(tmp_dir):
    file_utils.rmtree(tmp_dir)

  try:
    _recompile_to_temp(input_model_path, tmp_dir, configs, include_stablehlo)
    _promote_temp_dir_to_target(tmp_dir, target_version_dir)
    logging.info(
        "XSE AOT recompilation completed successfully at %s",
        target_version_dir,
    )
    return target_version_dir
  finally:
    if file_utils.exists(tmp_dir):
      file_utils.rmtree(tmp_dir)

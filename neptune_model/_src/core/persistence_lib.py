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

"""Saving to disk.

This file contains facilities that can save an OBM module to disk or load a
persisted OBM module.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib

from absl import logging

from neptune_model._src.core import constants
from neptune_model._src.core import device_assignment
from neptune_model._src.core import file_utils
from neptune_model._src.core import manifest_utils
from neptune_model._src.core import metadata
from neptune_model._src.core import saveable
from neptune_model._src.core import unstructured_data
from neptune_model.protos import manifest_pb2


@dataclasses.dataclass
class GlobalSupplemental:
  """GlobalSupplemental info to be saved in the manifest.pb.

  Attributes:
    data: The global supplemental info to be saved.
    save_as: An optional relative file name. If set, `data` will be saved as a
      separate file using this file name (and in this case the supplemental info
      mustn't be a `file_location` already).
  """

  data: manifest_pb2.UnstructuredData

  save_as: str | None = None


@dataclasses.dataclass
class SaveOptions:
  """Options for saving to FSM SavedModel.

  This function may be used in the `options` argument in functions that
  save an FSM SavedModel.

  Attributes:
    version: The serialization format version. With version >= 2 it generates
      manifest.pb whereas with version <= 1 it generates saved_model.pb.
    supplementals: Optional. A map of UnstructuredData to be saved in the
      manifest as the global supplemental info.
    visibility: Optional. A mapping from function name to its visibility (e.g.,
      `manifest_pb2.PUBLIC`, `manifest_pb2.PRIVATE`). If this parameter is not
      provided, all functions will be public. If only a subset of functions are
      provided in the mapping, the rest will be public by default.
    device_assignment_by_coords: Optional. A sequence of DeviceAssignment to be
      saved in the manifest.pb.
    save_shlo_to_file: Optional. If true, save the SHLO in a separate file
      instead of inlining it in the manifest.pb.
    include_stablehlo: Optional. A boolean flag indicating whether to include
      the StableHLO payload in the exported manifest. Defaults to True.
    compiler_hook: Optional. A `manifest_utils.CompilerHook` callback invoked to
      compile external artifacts (e.g. PJRT executables) that takes serialized
      MLIR bytecode, kept variable indices, and the number of weights, and
      returns an iterable of `manifest_utils.CompiledArtifact`s.
  """

  version: int | None = None
  supplementals: Mapping[str, GlobalSupplemental] | None = None
  visibility: Mapping[str, manifest_pb2.Visibility] | None = None
  device_assignment_by_coords: (
      Sequence[device_assignment.DeviceAssignment] | None
  ) = None
  save_shlo_to_file: bool = False
  include_stablehlo: bool = True
  compiler_hook: manifest_utils.CompilerHook | None = None


def save(
    module: Mapping[str, saveable.Saveable],
    target_dir: str,
    options: SaveOptions,
) -> None:
  """Saves module `module` in the target directory `target_dir`."""
  if options.version is None or options.version < 2:
    raise ValueError('Version must be >= 2')

  logging.info('Save version: %d', options.version)
  # Generate and export the manifest proto.
  supplemental_info = None
  if options.supplementals:
    supplemental_info = {
        name: _save_supplemental(supp, target_dir)
        for name, supp in options.supplementals.items()
    }

  manifest_proto = manifest_utils.build_manifest_proto(
      module,
      target_dir,
      supplementals=supplemental_info,
      visibilities=options.visibility,
      device_assignments=options.device_assignment_by_coords,
      save_shlo_to_file=options.save_shlo_to_file,
      compiler_hook=options.compiler_hook,
      include_stablehlo=options.include_stablehlo,
  )

  manifest_path = pathlib.Path(target_dir) / constants.MANIFEST_FILE_PATH
  file_utils.mkdir_p(str(manifest_path.parent))
  with file_utils.open_file(str(manifest_path), 'wb') as f:
    f.write(manifest_proto.SerializeToString())

  model_version = metadata.ModelVersion(
      version=constants.MANIFEST_VERSION,
      mime_type=constants.MANIFEST_MIME_TYPE,
      manifest_file_path=constants.MANIFEST_FILE_PATH,
  )
  # Write the metadata files to detect and parse Neptune/Orbax Model. The
  # version files should be THE LAST files to be written. They are used to
  # validate the export and identify a Model. We write both flavors for
  # backwards compatibility.
  model_version.save(
      str(pathlib.Path(target_dir) / constants.MODEL_VERSION_FILENAME)
  )

  legacy_model_version = metadata.ModelVersion(
      version=constants.MANIFEST_VERSION,
      mime_type=constants.MANIFEST_LEGACY_MIME_TYPE,
      manifest_file_path=constants.MANIFEST_FILE_PATH,
  )
  legacy_model_version.save(
      str(pathlib.Path(target_dir) / constants.MODEL_VERSION_LEGACY_FILENAME)
  )


def _load_model_version(
    saved_state_dir: str,
) -> metadata.ModelVersion:
  """Loads the model version from the saved state directory."""

  current_path = (
      pathlib.Path(saved_state_dir) / constants.MODEL_VERSION_FILENAME
  )
  if file_utils.exists(str(current_path)):
    # pylint: disable=g-unsafe-pickle-load
    return metadata.ModelVersion.load(str(current_path))

  legacy_path = (
      pathlib.Path(saved_state_dir) / constants.MODEL_VERSION_LEGACY_FILENAME
  )
  if file_utils.exists(str(legacy_path)):
    # pylint: disable=g-unsafe-pickle-load
    return metadata.ModelVersion.load(str(legacy_path))

  raise FileNotFoundError(
      f'No model version file found in {saved_state_dir}. Expected either'
      f' {constants.MODEL_VERSION_FILENAME} or'
      f' {constants.MODEL_VERSION_LEGACY_FILENAME}.'
  )


def load(saved_state_dir: str) -> manifest_pb2.Manifest:
  """Discovers and loads the manifest in the saved state directory."""
  model_version = _load_model_version(saved_state_dir)

  # TODO: Once there is more than one version of the model
  # we will need to check the list of supported versions and customize the
  # loading process accordingly.
  if model_version.version != constants.MANIFEST_VERSION:
    raise ValueError(f'Unsupported manifest version "{model_version.version}"')
  if (
      model_version.mime_type != constants.MANIFEST_MIME_TYPE
      and model_version.mime_type != constants.MANIFEST_LEGACY_MIME_TYPE
  ):
    raise ValueError(f'Unsupported manifest type "{model_version.mime_type}"')
  if pathlib.Path(model_version.manifest_file_path).is_absolute():
    raise ValueError(
        'Manifest file path must be relative, got'
        f' "{model_version.manifest_file_path}"'
    )

  manifest_path = (
      pathlib.Path(saved_state_dir) / model_version.manifest_file_path
  )
  with file_utils.open_file(str(manifest_path), 'rb') as f:
    return manifest_pb2.Manifest.FromString(f.read())


def _save_supplemental(
    supplemental: GlobalSupplemental,
    target_dir: str,
) -> manifest_pb2.UnstructuredData:
  """Returns the supplemental data, either inlined or saved to disk."""
  if supplemental.save_as is None:
    return supplemental.data

  return unstructured_data.write_inlined_data_to_file(
      supplemental.data,
      target_dir,
      supplemental.save_as,
  )

"""Saving to disk.

This file contains facilities that can save an OBM module to disk or load a
persisted OBM module.
"""

from collections.abc import Mapping, Sequence
import dataclasses
import pathlib

from absl import logging

from .third_party.neptune.model._src.core import constants
from .third_party.neptune.model._src.core import device_assignment
from .third_party.neptune.model._src.core import file_utils
from .third_party.neptune.model._src.core import manifest_utils
from .third_party.neptune.model._src.core import metadata
from .third_party.neptune.model._src.core import saveable
from .third_party.neptune.model._src.core import unstructured_data
from .third_party.neptune.protos import manifest_pb2


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
  """

  version: int | None = None
  supplementals: Mapping[str, GlobalSupplemental] | None = None
  visibility: Mapping[str, manifest_pb2.Visibility] | None = None
  device_assignment_by_coords: (
      Sequence[device_assignment.DeviceAssignment] | None
  ) = None


def save(
    module: dict[str, saveable.Saveable],
    target_dir: str,
    options: SaveOptions,
) -> None:
  """Saves module `module` in the target directory `target_dir`."""
  if options.version is None or options.version < 2:
    raise ValueError('Version must be >= 2')

  logging.info('Save version: %d', options.version)
  # Generate and export the manifest proto.
  supplemental_info = None
  if options.supplementals is not None:
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
  # Write the main metadata to detect and parse an Orbax Model. The version file
  # should be THE LAST file to be written. It is used to validate the export and
  # identify an Orbax Model.
  model_version.save(
      str(pathlib.Path(target_dir) / constants.MODEL_VERSION_FILENAME)
  )


def load(saved_state_dir: str) -> manifest_pb2.Manifest:
  """Discovers and loads the manifest in the saved state directory."""
  model_version = metadata.ModelVersion.load(
      str(pathlib.Path(saved_state_dir) / constants.MODEL_VERSION_FILENAME)
  )

  # TODO: Once there is more than one version of the model
  # we will need to check the list of supported versions and customize the
  # loading process accordingly.
  if model_version.version != constants.MANIFEST_VERSION:
    raise ValueError(f'Unsupported manifest version "{model_version.version}"')
  if model_version.mime_type != constants.MANIFEST_MIME_TYPE:
    raise ValueError(f'Unsupported manifest type "{model_version.mime_type}"')

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

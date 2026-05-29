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

"""Library for saving assets to Neptune model."""

import collections
from collections.abc import Mapping, Sequence
import os
import pathlib
import re

from neptune_model._src.core import file_utils
from neptune_model._src.core import persistence_lib
from neptune_model.protos import asset_map_pb2
from neptune_model.protos import manifest_pb2

ASSETS_SUBDIR = 'assets'
ASSET_MAP_MIME_TYPE = 'application/protobuf; type=neptune_assets.AssetMap'
ASSET_MAP_VERSION = '0.0.1'
ASSET_MAP_SUPPLEMENTAL_NAME = 'asset_map'


def _normalize_file_name(file_name: str) -> str:
  """Strips numeric suffixes from a filename.

  For example, 'foo_1.txt' becomes 'foo.txt'. This helps group files
  for conflict resolution.

  Args:
    file_name: The base name of file.

  Returns:
    The normalized file name.
  """
  base, ext = os.path.splitext(file_name)
  # Strip numeric suffixes (_1, _123, _1_2, etc.) to normalize filenames
  # like 'foo_1.txt' and 'foo.txt' to the same base 'foo.txt' so that the
  # conflict can be resolved efficiently with the self._next_index_map.
  while re.search(r'_\d+$', base):
    base = re.sub(r'_\d+$', '', base)
  return f'{base}{ext}'


class _AssetMapBuilder:
  """Helper class to build AssetMap efficiently."""

  def __init__(self):
    self._asset_map = asset_map_pb2.AssetMap()
    # Maps unique/sanitized filenames to their original source paths.
    # Used to detect filename collisions and ensure unique asset filenames.
    self._auxiliary_file_map: dict[str, str] = {}
    # Stores the next available index for a given base filename,
    # defaulting to 1 if the base filename hasn't been seen before.
    self._next_index_map: dict[str, int] = collections.defaultdict(lambda: 1)

  @property
  def asset_map(self) -> asset_map_pb2.AssetMap:
    return self._asset_map

  def add_asset(self, source_path: str, subdir: str) -> None:
    """Adds an asset to the map, resolving name conflicts.

    If the base filename of the source_path is already used by a different
    source path in the map, a unique indexed filename (e.g., 'file_1.ext')
    is generated and used.

    Args:
      source_path: The original path of the asset file.
      subdir: The subdirectory to save the asset to relative to the model export
        path.
    """
    # The asset has been added before, skip.
    if source_path in self._asset_map.assets:
      return

    file_name = os.path.basename(source_path)
    unique_file_name = _normalize_file_name(file_name)

    # If the file_name is already in use by a different source path,
    # or if we've previously generated indexed names for this base,
    # we need to find a unique indexed name.
    if unique_file_name in self._auxiliary_file_map:
      base, ext = os.path.splitext(unique_file_name)
      i = self._next_index_map[base]
      unique_file_name = f'{base}_{i}{ext}'
      self._next_index_map[base] += 1

    # Add the unique file name to the maps.
    self._auxiliary_file_map[unique_file_name] = source_path
    self._asset_map.assets[source_path] = os.path.join(
        subdir, ASSETS_SUBDIR, unique_file_name
    )


def build_asset_map(
    asset_bundle_name: str,
    asset_source_paths: Sequence[str],
) -> asset_map_pb2.AssetMap:
  """Builds an AssetsMap proto from a set of asset source paths.

  Args:
    asset_bundle_name: The name of the asset bundle.
    asset_source_paths: A sequence of asset source paths.

  Returns:
    An AssetMap proto containing the asset source paths and their destination
    paths relative to the asset bundle name.
  """
  builder = _AssetMapBuilder()
  for source_path in sorted(set(asset_source_paths)):
    builder.add_asset(source_path, asset_bundle_name)
  return builder.asset_map


def save_assets(
    supplemental_name: str,
    assets: asset_map_pb2.AssetMap | Sequence[str],
    export_path: str,
) -> Mapping[str, persistence_lib.GlobalSupplemental]:
  """Saves assets under the model export path.

  Args:
    supplemental_name: The name of the supplemental to save the assets map to.
    assets: An AssetMap proto containing the asset source paths and their
      destination paths relative to the model export path, or a sequence of
      asset source paths.
    export_path: Directory to save the assets to.

  Returns:
    A GlobalSupplemenal object containing the saved assets map. The asset
    map maps original asset source paths to their destination paths relative
    to the model export path.
  """

  if not re.fullmatch(r'^[a-zA-Z0-9-_]+$', supplemental_name):
    raise ValueError(
        'Supplemental name must contain only alphanumeric characters, '
        'underscores and hyphens'
    )

  if isinstance(assets, Sequence):
    asset_map_builder = _AssetMapBuilder()
    for src_path in assets:
      asset_map_builder.add_asset(src_path, supplemental_name)
    asset_map = asset_map_builder.asset_map
  else:
    asset_map = assets

  if not asset_map.assets.items():
    return {}

  for source_path, dest_relative_path in asset_map.assets.items():
    resolved_source_path = pathlib.Path(source_path).resolve()
    resolved_dest_path = pathlib.Path(export_path, dest_relative_path).resolve()

    if not resolved_dest_path.is_relative_to(export_path):
      raise ValueError(
          f'Asset "{resolved_source_path}" cannot be saved to'
          f' "{resolved_dest_path}" as this would move it'
          ' outside of the model export path'
      )

    file_utils.mkdir_p(str(resolved_dest_path.parent))
    file_utils.copy(str(resolved_source_path), str(resolved_dest_path))

  return {
      supplemental_name: persistence_lib.GlobalSupplemental(
          data=manifest_pb2.UnstructuredData(
              inlined_bytes=asset_map.SerializeToString(),
              mime_type=ASSET_MAP_MIME_TYPE,
              version=ASSET_MAP_VERSION,
          ),
          save_as=supplemental_name + '_asset_map.pb',
      )
  }

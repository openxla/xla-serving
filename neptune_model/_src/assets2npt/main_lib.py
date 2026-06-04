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

from collections.abc import Mapping
import os
import pathlib

import blake3

from neptune_model._src.core import file_utils

ASSETS_SUBDIR = 'assets'


def file_hexdigest(path: str) -> str:
  """Returns the BLAKE3 hexdigest of the file at the given path."""
  hasher = blake3.blake3()
  with open(path, 'rb') as f:
    while chunk := f.read(2 * 1024 * 1024):  # 2 MiB
      hasher.update(chunk)
  return hasher.hexdigest()


class AssetMap:
  """A content-addressable mapping of model assets."""

  def __init__(self):
    self._assets: dict[str, str] = {}

  @property
  def assets(self) -> Mapping[str, str]:
    """A mapping from source paths to destination paths.

    The destination paths are relative to the model export path.
    """
    return self._assets

  def add_asset(self, source_path: str) -> str:
    """Adds an asset and returns its content-addressable relative destination path.

    If the asset source path was already added, returns the cached destination
    path without re-hashing.

    Args:
      source_path: The source path of the asset.

    Returns:
      The destination path of the asset relative to the export path.
    """
    if dest_path := self._assets.get(source_path):
      return dest_path

    hexdigest = file_hexdigest(source_path)
    dest_relative_path = os.path.join(ASSETS_SUBDIR, hexdigest)
    self._assets[source_path] = dest_relative_path
    return dest_relative_path


def save_assets(
    asset_map: AssetMap,
    export_path: str,
) -> None:
  """Saves assets under the model export path.

  Args:
    asset_map: An AssetMap maintaining the content-addressable mapping.
    export_path: Directory to save the assets to.
  """
  export_path = pathlib.Path(export_path).resolve()

  if not asset_map.assets:
    return

  source_path_by_dest_path = {}
  for source_path, dest_relative_path in asset_map.assets.items():
    resolved_source_path = pathlib.Path(source_path).resolve()
    resolved_dest_path = pathlib.Path(export_path, dest_relative_path).resolve()
    if not resolved_dest_path.is_relative_to(export_path):
      raise ValueError(
          f'Asset "{resolved_source_path}" cannot be saved to'
          f' "{resolved_dest_path}" as this would move it'
          ' outside of the model export path'
      )
    source_path_by_dest_path[resolved_dest_path] = resolved_source_path

  for dest_path, source_path in source_path_by_dest_path.items():
    if not dest_path.exists():
      file_utils.mkdir_p(str(dest_path.parent))
      file_utils.copy(str(source_path), str(dest_path))

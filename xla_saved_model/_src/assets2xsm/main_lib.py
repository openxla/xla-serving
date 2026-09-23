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

"""Library for saving assets to XLA SavedModel."""

from collections.abc import Mapping
import os
import pathlib

from xla_saved_model._src.core import file_utils

ASSETS_SUBDIR = 'assets'


class AssetMap:
  """A mapping of model assets maintaining unique destination paths."""

  def __init__(self, bundle_name: str):
    self._assets: dict[str, str] = {}
    self._used_dest_paths: set[str] = set()
    self._bundle_name = bundle_name

    if not self._bundle_name:
      raise ValueError('Bundle name must not be empty')

  @property
  def bundle_name(self) -> str:
    return self._bundle_name

  @property
  def assets(self) -> Mapping[str, str]:
    """A mapping from source paths to destination paths.

    The destination paths are relative to the model export path.
    """
    return self._assets

  def add_asset(self, source_path: str) -> str:
    """Adds an asset and returns its relative destination path.

    If the asset source path was already added, returns the cached destination
    path.

    Args:
      source_path: The source path of the asset.

    Returns:
      The destination path of the asset relative to the export path.
    """
    if dest_path := self._assets.get(source_path):
      return dest_path

    resolved_source_path = pathlib.Path(source_path).resolve()
    stem = resolved_source_path.stem
    ext = resolved_source_path.suffix

    dest_relative_path = os.path.join(ASSETS_SUBDIR, f'{stem}{ext}')
    counter = 1
    while dest_relative_path in self._used_dest_paths:
      dest_relative_path = os.path.join(ASSETS_SUBDIR, f'{stem}_{counter}{ext}')
      counter += 1

    self._assets[source_path] = dest_relative_path
    self._used_dest_paths.add(dest_relative_path)
    return dest_relative_path


def save_assets(
    asset_map: AssetMap,
    export_path: str,
) -> None:
  """Saves assets under the model export path.

  Args:
    asset_map: An AssetMap maintaining the asset mapping.
    export_path: Directory to save the assets to.
  """
  if not asset_map.assets:
    return

  bundle_path = pathlib.Path(export_path, asset_map.bundle_name).resolve()
  if file_utils.exists(str(bundle_path)):
    raise ValueError(
        f'Asset bundle {bundle_path!r} already exists in {export_path!r}'
    )

  source_path_by_dest_path = {}
  for source_path, dest_relative_path in asset_map.assets.items():
    resolved_source_path = pathlib.Path(source_path).resolve()
    resolved_dest_path = pathlib.Path(bundle_path, dest_relative_path).resolve()
    if not resolved_dest_path.is_relative_to(bundle_path):
      raise ValueError(
          f'Asset {resolved_source_path!r} cannot be saved to'
          f' {resolved_dest_path!r} as this would move it'
          ' outside of the model export path'
      )
    source_path_by_dest_path[resolved_dest_path] = resolved_source_path

  for dest_path, source_path in source_path_by_dest_path.items():
    if file_utils.exists(str(dest_path)):
      # Asset destination paths should be unique per bundle by construction.
      raise ValueError(
          f'Asset {source_path!r} cannot be saved to {dest_path!r} as this'
          ' path already exists'
      )
    file_utils.mkdir_p(str(dest_path.parent))
    file_utils.copy(str(source_path), str(dest_path))

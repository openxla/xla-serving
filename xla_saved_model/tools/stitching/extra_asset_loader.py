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

"""Tools for loading the extra assets saved with a model."""

import os
import pathlib

from xla_saved_model._src.core import file_utils
from xla_saved_model._src.export import warmup


def find_extra_asset_path(model_dir: pathlib.Path | str, name: str) -> str:
  """Returns the path of an extra asset saved with a model.

  Extra assets are files that a model keeps in its `assets.extra` directory for
  tools to read, rather than for serving.

  Args:
    model_dir: Directory where the model was saved.
    name: File name of the asset.

  Returns:
    The path of the asset.

  Raises:
    ValueError: If `name` is not a file name.
    FileNotFoundError: If the model has no such asset.
  """
  if name in ("", os.curdir, os.pardir) or os.path.basename(name) != name:
    raise ValueError(f"The asset name must be a file name, got {name!r}.")
  path = os.path.join(str(model_dir), warmup.ASSETS_EXTRA_DIR, name)
  if not file_utils.exists(path):
    raise FileNotFoundError(f"The model has no extra asset '{name}': {path}")
  return path


def load_extra_asset(model_dir: pathlib.Path | str, name: str) -> bytes:
  """Reads an extra asset saved with a model.

  Args:
    model_dir: Directory where the model was saved.
    name: File name of the asset.

  Returns:
    The contents of the asset.

  Raises:
    ValueError: If `name` is not a file name.
    FileNotFoundError: If the model has no such asset.
  """
  with file_utils.open_file(find_extra_asset_path(model_dir, name), "rb") as f:
    return f.read()


__all__ = [
    "find_extra_asset_path",
    "load_extra_asset",
]

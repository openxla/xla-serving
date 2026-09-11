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

"""File utilities."""

import contextlib
import os
import shutil

_file_opener = open
_mkdir_p = lambda path: os.makedirs(path, exist_ok=True)
_copy = lambda src, dst, overwrite=False: (
    shutil.copyfile(src, dst)
    if (overwrite or not os.path.exists(dst))
    else (_ for _ in ()).throw(FileExistsError(f"Destination {dst} exists"))
)
_copytree = shutil.copytree
_exists = os.path.exists
_isdir = os.path.isdir
_remove = os.remove
_rename = lambda src, dst, overwrite=True: (
    os.replace(src, dst) if overwrite else os.rename(src, dst)
)
_rmtree = shutil.rmtree


@contextlib.contextmanager
def open_file(filename: str, mode: str):
  """Opens a file with the given filename and mode."""
  f = _file_opener(filename, mode)
  try:
    yield f
  finally:
    f.close()


def mkdir_p(path: str) -> None:
  """Creates a directory, creating parent directories as needed."""
  _mkdir_p(path)


def copy(source: str, dest: str, overwrite: bool = False) -> None:
  _copy(source, dest, overwrite=overwrite)


def copytree(source: str, dest: str) -> None:
  _copytree(source, dest)


def exists(path: str) -> bool:
  return _exists(path)


def isdir(path: str) -> bool:
  return _isdir(path)


def remove(path: str) -> None:
  _remove(path)


def rmtree(path: str) -> None:
  _rmtree(path)


def rename(source: str, dest: str, overwrite: bool = True) -> None:
  _rename(source, dest, overwrite=overwrite)

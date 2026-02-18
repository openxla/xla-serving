"""File utilities."""

import contextlib
import os
import shutil


_file_opener = open
_mkdir_p = lambda path: os.makedirs(path, exist_ok=True)
_copy = shutil.copyfile

# BEGIN GOOGLE-INTERNAL
from google3.pyglib import gfile  # pylint: disable=g-import-not-at-top
from google3.pyglib.contrib.gpathlib import gpath  # pylint: disable=g-import-not-at-top

_file_opener = gfile.Open


def _gfile_mkdir_p(path: str) -> None:
  dir_path = gpath.GPath(path)
  dir_path.mkdir(parents=True, exist_ok=True)


_mkdir_p = _gfile_mkdir_p

_copy = gfile.Copy
# END GOOGLE-INTERNAL


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


def copy(source: str, dest: str) -> None:
  _copy(source, dest)

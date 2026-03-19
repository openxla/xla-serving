"""File utilities."""

import contextlib
import os
import shutil


_file_opener = open
_mkdir_p = lambda path: os.makedirs(path, exist_ok=True)
_copy = shutil.copyfile


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

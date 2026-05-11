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
"""Setup file for neptune_model. Stages and compiles protos before building."""

import pathlib
import re
import shutil
import sys
from grpc_tools import protoc
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
PACKAGE_NAME = "neptune_model"
PROTOS_SRC_DIR = ROOT_DIR / "protos"
PROTOS_STAGING_DIR = ROOT_DIR / "build" / "protos"

PROTOS_OLD_PREFIX = "protos/"
PROTOS_NEW_PREFIX = "neptune_model/protos/"


def stage_protos():
  """Updates protos to use Python-specific import paths."""
  if not PROTOS_SRC_DIR.exists():
    sys.exit(f"Proto source directory not found: {PROTOS_SRC_DIR}")

  if PROTOS_STAGING_DIR.exists():
    shutil.rmtree(PROTOS_STAGING_DIR)

  pkg_stage_root = PROTOS_STAGING_DIR / PACKAGE_NAME
  pkg_stage_root.mkdir(parents=True)

  staged = []

  for src in PROTOS_SRC_DIR.rglob("*.proto"):
    rel = src.relative_to(ROOT_DIR)
    dst = pkg_stage_root / rel
    dst.parent.mkdir(parents=True, exist_ok=True)

    contents = src.read_bytes()
    contents = contents.replace(
        f'import "{PROTOS_OLD_PREFIX}'.encode(),
        f'import "{PROTOS_NEW_PREFIX}'.encode(),
    )
    dst.write_bytes(contents)

    staged.append(dst.relative_to(PROTOS_STAGING_DIR))

  return staged


def compile_protos(staged):
  """Runs protoc from grpcio-tools on staged protos."""
  if not staged:
    return

  args = [
      "protoc",
      f"--proto_path={PROTOS_STAGING_DIR}",
      f"--python_out={ROOT_DIR}",
      f"--pyi_out={ROOT_DIR}",
  ]
  args.extend(str(p) for p in staged)

  rc = protoc.main(args)
  if rc != 0:
    sys.exit(f"protoc failed with exit code {rc}")


def ensure_init_files():
  """Ensures that __init__.py files exist in generated proto directories."""

  for d in ROOT_DIR.rglob(f"{PROTOS_NEW_PREFIX}/**"):
    if d.is_dir():
      init_py_path = d / "__init__.py"
      if not init_py_path.exists():
        init_py_path.touch()


class GenerateProtosAndBuild(build_py):
  """Generates Python protos prior to building."""

  def run(self):
    staged_protos = stage_protos()
    compile_protos(staged_protos)
    ensure_init_files()

    # Remind setuptools about newly generated files.
    self.distribution_packages = find_packages(where=".")
    self.packages = self.distribution_packages

    super().run()


with (ROOT_DIR / PACKAGE_NAME / "__init__.py").open("r", encoding="utf-8") as f:
  version_match = re.search(
      r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE
  )
  if version_match:
    __version__ = version_match.group(1)
  else:
    __version__ = "unknown"
    sys.exit("Unable to find __version__ string in __init__.py")

setup(
    version=__version__,
    packages=find_packages(where="."),
    package_data={
        PACKAGE_NAME: ["**/*.pyi"],
    },
    include_package_data=True,
    cmdclass={
        "build_py": GenerateProtosAndBuild,
    },
)

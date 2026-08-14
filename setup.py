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
import subprocess
import sys

import grpc_tools
from grpc_tools import protoc
from setuptools import find_packages
from setuptools import setup
from setuptools.command.build_py import build_py

ROOT_DIR = pathlib.Path(__file__).parent.resolve()
PACKAGE_NAME = "neptune_model"
PROTOS_SRC_DIR = ROOT_DIR / "protos"
PROTOS_STAGING_DIR = ROOT_DIR / "build" / "protos"

# Relocates imports to ensure that generated Python protos use
# Python-specific import paths withihn the built package.
NEPTUNE_PROTOS_IMPORT_PREFIX_MAPPING = {
    "protos/": "neptune_model/protos/",
    "xla/": "neptune_model/protos/xla/",
}

XLA_PROTOS_IMPORT_PREFIX_MAPPING = {
    "xla/": "neptune_model/protos/xla/",
}

# All of xla names used by neptune_model protos now need to point to the
# neptune_model.protos.xla namespace.
NEPTUNE_PROTOS_NAMES_MAPPING = {
    "xla.": "neptune_model.xla.",
}

XLA_PROTOS_PACKAGE_MAPPING = {
    "package xla;": "package neptune_model.xla;",
    "xla.": "neptune_model.xla.",
}


def rewrite_proto_file(src, dst, import_mappings, name_mappings):
  """Rewrites a proto file by applying prefix and package/name mappings."""
  dst.parent.mkdir(parents=True, exist_ok=True)

  out = []
  for line in src.read_text().splitlines(keepends=True):
    if line.lstrip().startswith("import "):
      for old_prefix, new_prefix in import_mappings.items():
        line = line.replace(old_prefix, new_prefix)
    else:
      for old_package_ref, new_package_ref in name_mappings.items():
        line = line.replace(old_package_ref, new_package_ref)

    out.append(line)

  dst.write_text("".join(out))


def stage_xla_protos(target_dir):
  """Stages XLA protos from the XLA Bazel repository."""

  bazel_target = "@xla//xla:xla_data_proto"
  subprocess.run(
      ["bazel", "fetch", bazel_target],
      check=True,
      cwd=ROOT_DIR,
  )

  output_base_dir = (
      subprocess.run(
          ["bazel", "info", "output_base"],
          check=True,
          capture_output=True,
          cwd=ROOT_DIR,
      )
      .stdout.decode()
      .strip()
  )

  xla_base_dir = pathlib.Path(output_base_dir) / "external" / "xla+" / "xla"

  staged = []
  for src in xla_base_dir.rglob("*.proto"):
    dst = target_dir / src.relative_to(xla_base_dir)
    rewrite_proto_file(
        src,
        dst,
        XLA_PROTOS_IMPORT_PREFIX_MAPPING,
        XLA_PROTOS_PACKAGE_MAPPING,
    )
    staged.append(dst)

  return staged


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
    rewrite_proto_file(
        src,
        dst,
        NEPTUNE_PROTOS_IMPORT_PREFIX_MAPPING,
        NEPTUNE_PROTOS_NAMES_MAPPING,
    )
    staged.append(dst.relative_to(PROTOS_STAGING_DIR))

  # Place XLA protos under the the package root "protos/xla"
  # subdirectory to ensure they're importable via 'neptune_model.protos.xla'.
  staged.extend(stage_xla_protos(pkg_stage_root / "protos" / "xla"))

  # Stage test-only files located inside the package root directory.
  pkg_root = ROOT_DIR / PACKAGE_NAME
  for src in pkg_root.rglob("*_test.proto"):
    rel_to_pkg = src.relative_to(pkg_root)
    dst = pkg_stage_root / rel_to_pkg
    rewrite_proto_file(
        src,
        dst,
        NEPTUNE_PROTOS_IMPORT_PREFIX_MAPPING,
        NEPTUNE_PROTOS_NAMES_MAPPING,
    )
    staged.append(dst.relative_to(PROTOS_STAGING_DIR))

  return staged


def compile_protos(staged):
  """Runs protoc from grpcio-tools on staged protos."""
  if not staged:
    return

  args = [
      "protoc",
      "--proto_path="
      + str(pathlib.Path(grpc_tools.__file__).parent / "_proto"),
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

  for d in (ROOT_DIR / PACKAGE_NAME / "protos").rglob("**"):
    if d.is_dir():
      init_py_path = d / "__init__.py"
      if not init_py_path.exists():
        init_py_path.touch()


class GenerateProtosAndBuild(build_py):
  """Generates Python protos prior to building."""

  def find_package_modules(self, package, package_dir):
    modules = super().find_package_modules(package, package_dir)
    return [
        (pkg, mod, path)
        for pkg, mod, path in modules
        if not mod.endswith("_test")
        and mod != "test_utils"
        and not mod.endswith("_test_pb2")
    ]

  def run(self):
    staged_protos = stage_protos()
    compile_protos(staged_protos)
    ensure_init_files()

    # Remind setuptools about newly generated files.
    self.distribution_packages = find_packages(where=".")
    self.packages = self.distribution_packages

    super().run()


with (ROOT_DIR / "version.bzl").open("r", encoding="utf-8") as f:
  version_match = re.search(
      r"^VERSION\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE
  )
  if version_match:
    __version__ = version_match.group(1)
  else:
    __version__ = "unknown"
    sys.exit("Unable to find VERSION string in version.bzl")

setup(
    version=__version__,
    packages=find_packages(where="."),
    package_data={
        PACKAGE_NAME: ["**/*.pyi"],
    },
    exclude_package_data={
        PACKAGE_NAME: ["**/*_test_pb2.pyi"],
    },
    include_package_data=True,
    cmdclass={
        "build_py": GenerateProtosAndBuild,
    },
)

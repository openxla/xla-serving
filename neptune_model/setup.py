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

import os
import re
import shutil
import stat
import subprocess
from setuptools import find_packages, setup
from setuptools.command.build_py import build_py

this_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.environ.get("NEPTUNE_ROOT_DIR") or os.path.abspath(
    os.path.join(this_dir, "..")
)

# 1. Create PLACEHOLDERS for both folders
# We need both to exist so setuptools recognizes them as packages initially
for folder in ["xla", "protos"]:
  path = os.path.join(this_dir, folder)
  os.makedirs(path, exist_ok=True)
  with open(os.path.join(path, "__init__.py"), "w") as f:
    f.write(f"# Generated {folder.upper()} Namespace\n")


class BazelBuildProtos(build_py):

  def run(self):
    if not os.path.exists(os.path.join(root_dir, "MODULE.bazel")):
      print("\n" + "=" * 50)
      print("🛑 BUILD ISOLATION ERROR 🛑")
      print(f"root_dir: {root_dir}")
      print(
          "MODULE.bazel not found! If you ran `pip install .`, pip copied this"
      )
      print(
          "folder to an isolated temp directory and left the parent folder"
          " behind."
      )
      print("Please run `pip install --no-build-isolation .` instead.")
      print("=" * 50 + "\n")
      print("Warning: MODULE.bazel not found. Skipping build.")
    else:
      bazel_bin = shutil.which("bazelisk") or shutil.which("bazel")
      # Add this safety check!
      if bazel_bin is None:
        raise RuntimeError(
            "Bazel executable not found! Please ensure 'bazel' or 'bazelisk' "
            "is installed and available in your system PATH."
        )

      subprocess.check_call([bazel_bin, "build", "//protos:all"], cwd=root_dir)

      gen_root = os.path.join(root_dir, "bazel-bin")
      for root, _, files in os.walk(gen_root):
        for f in files:
          if f.endswith(("_pb2.py", "_pb2.pyi")):
            source_path = os.path.join(root, f)

            # 2. Logic to decide WHERE to copy
            relative_path = os.path.relpath(root, gen_root)
            parts = relative_path.split(os.sep)

            # STRIP BAZEL EXTERNAL PATHS:
            # If the path starts with 'external/xla~/' (or any repo), drop the first two folders
            if parts[0] == "external" and len(parts) > 2:
              parts = parts[2:]

            pkg_name = "xla" if "xla" in f or "xla" in root else "protos"
            if not parts or parts[0] != pkg_name:
              parts = [pkg_name] + parts
            target_dir = os.path.join(this_dir, *parts)

            # FIX 1 & 2: Create intermediate folders AND proper init files safely
            current_path = this_dir
            for part in parts:
              if part and part != ".":
                current_path = os.path.join(current_path, part)
                os.makedirs(current_path, exist_ok=True)
                init_file = os.path.join(current_path, "__init__.py")
                if not os.path.exists(init_file):
                  with open(init_file, "w") as init_f:
                    init_f.write("# Generated Namespace\n")

            dest_path = os.path.join(target_dir, f)

            # Standard Permission Fix
            if os.path.exists(dest_path):
              os.chmod(dest_path, stat.S_IWUSR | stat.S_IRUSR)
              os.remove(dest_path)

            shutil.copyfile(source_path, dest_path)
            os.chmod(
                dest_path,
                stat.S_IRUSR | stat.S_IWUSR | stat.S_IRGRP | stat.S_IROTH,
            )

            # ==========================================
            # NEW: Fix generated import paths dynamically
            # ==========================================
            with open(dest_path, "r", encoding="utf-8") as file:
              content = file.read()

            # 1. Catch deeply nested 'from' imports (e.g., from xla.pjrt.proto import...)
            content = content.replace("from xla.", "from neptune_model.xla.")
            content = content.replace(
                "from protos.", "from neptune_model.protos."
            )

            # 2. Catch standard 'import' statements (e.g., import xla.pjrt.proto)
            content = content.replace(
                "import xla.", "import neptune_model.xla."
            )
            content = content.replace(
                "import protos.", "import neptune_model.protos."
            )

            # 3. Catch exact top-level imports just in case Bazel generates them
            content = content.replace(
                "from xla import ", "from neptune_model.xla import "
            )
            content = content.replace(
                "from protos import ", "from neptune_model.protos import "
            )

            with open(dest_path, "w", encoding="utf-8") as file:
              file.write(content)
            # ==========================================
            print(f"Bundled {f} into {os.path.basename(target_dir)}/")

      # FIX 3: Defeat the Race Condition!
      # Now that Bazel is done and all init files exist, find the nested packages
      # and force setuptools to add them to the wheel right before wrapping it up.
      from setuptools import find_namespace_packages

      new_pkgs = [
          f"neptune_model.{pkg}"
          for pkg in find_namespace_packages(where=".")
          if pkg.startswith("xla") or pkg.startswith("protos")
      ]
      for pkg in new_pkgs:
        if pkg not in self.distribution.packages:
          self.distribution.packages.append(pkg)

    super().run()

init_path = os.path.join(os.path.dirname(__file__), "__init__.py")

with open(init_path, "r", encoding="utf-8") as f:
  version_match = re.search(
      r"^__version__\s*=\s*['\"]([^'\"]+)['\"]", f.read(), re.MULTILINE
  )
  if version_match:
    __version__ = version_match.group(1)
  else:
    raise RuntimeError("Unable to find __version__ string in __init__.py")


discovered_packages = [
    f"neptune_model.{pkg}" for pkg in find_packages(where=".")
]

setup(
    version=__version__,
    packages=["neptune_model"] + discovered_packages,
    package_dir={
        "neptune_model": ".",
        "neptune_model.xla": "xla",
        "neptune_model.protos": "protos",
    },
    package_data={
        "": ["*.py", "*.pyi"],
    },
    include_package_data=True,
    cmdclass={"build_py": BazelBuildProtos},
)

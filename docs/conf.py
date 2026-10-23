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

"""Sphinx configuration for the XLA Serving Engine documentation.

Build locally with:

    pip install -r docs/requirements.txt
    sphinx-build -W -b html docs docs/_build/html
"""

import pathlib
import re

_REPO_ROOT = pathlib.Path(__file__).resolve().parent.parent
_GITHUB_URL = "https://github.com/AI-Hypercomputer/xla-serving"


def _read_version() -> str:
  """Reads VERSION from version.bzl so the docs never drift from the wheels."""
  text = (_REPO_ROOT / "version.bzl").read_text()
  match = re.search(r'VERSION\s*=\s*"([^"]+)"', text)
  return match.group(1) if match else "unknown"


# -- Project information -----------------------------------------------------

project = "XLA Serving Engine"
author = "XLA Serving Engine Authors"
copyright = "2026, Google LLC"  # pylint: disable=redefined-builtin
release = _read_version()
version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "myst_parser",
    "sphinx_design",
    "sphinx_copybutton",
    "sphinxcontrib.mermaid",
]

# `google/` holds internal-only material that is stripped from the public
# repository; excluding it keeps an internal checkout building cleanly too.
exclude_patterns = ["_build", "google/**", "Thumbs.db", ".DS_Store"]

source_suffix = {".md": "markdown"}
master_doc = "index"

myst_enable_extensions = [
    "attrs_inline",
    "colon_fence",
    "deflist",
    "fieldlist",
    "substitution",
]
myst_heading_anchors = 3
myst_substitutions = {
    "version": release,
    "github": _GITHUB_URL,
}

# Strip prompts when copying shell/python snippets.
copybutton_prompt_text = r">>> |\.\.\. |\$ "
copybutton_prompt_is_regexp = True

# -- HTML output -------------------------------------------------------------

html_theme = "sphinx_book_theme"
html_title = f"XLA Serving Engine {release}"
html_static_path = ["_static"]
html_css_files = ["custom.css"]
html_last_updated_fmt = ""

html_theme_options = {
    "repository_url": _GITHUB_URL,
    "repository_branch": "main",
    "path_to_docs": "docs",
    "use_repository_button": True,
    "use_issues_button": True,
    "use_edit_page_button": True,
    "use_download_button": False,
    "home_page_in_toc": True,
    "show_navbar_depth": 1,
    "show_toc_level": 2,
    "navigation_with_keys": False,
    "pygments_light_style": "tango",
    "pygments_dark_style": "monokai",
}

# Render mermaid diagrams client-side; no local binary needed on RTD.
# Option names follow sphinxcontrib-mermaid >= 1.0.
mermaid_output_format = "raw"
mermaid_fullscreen = False
mermaid_d3_zoom = False
mermaid_height = "auto"
mermaid_light_theme = "neutral"
mermaid_dark_theme = "dark"
mermaid_init_config = {
    "startOnLoad": False,
    "themeVariables": {"fontSize": "15px"},
    "flowchart": {
        "curve": "basis",
        "padding": 14,
        "nodeSpacing": 28,
        "rankSpacing": 36,
        "subGraphTitleMargin": {"top": 6, "bottom": 10},
    },
}

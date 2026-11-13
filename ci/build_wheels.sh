#!/bin/bash
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
#
# Builds and validates XLA Serving Python wheels from the repository root.
#
# Inputs (environment variables):
#   XSE_PYTHON_VERSION: Python major.minor to build for (3.12, 3.13, or 3.14).
#   XSE_RC_VERSION: Release candidate suffix (e.g. rc0). When set, builds a
#     release: both VERSION and VERSION<rc> wheels, where VERSION comes from
#     version.bzl. When empty, builds a nightly VERSION.devYYYYMMDD wheel set
#     using the HEAD commit date in America/Los_Angeles.
#   XSE_WHEEL_OUTDIR: Optional output directory (defaults to <repo>/dist).
#   XSE_WRITE_TO_BAZEL_REMOTE_CACHE: Optional; 1 to write to the remote cache.
set -exu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT
readonly PURE_WHEEL_PYTHON="3.12"
readonly PYPI_MIRROR_URL="https://us-python.pkg.dev/ml-oss-artifacts-published/pypi-mirror/simple/"

: "${XSE_PYTHON_VERSION:?XSE_PYTHON_VERSION must be set (e.g. 3.12, 3.13, 3.14)}"
RC_VERSION="${XSE_RC_VERSION:-}"
readonly RC_VERSION

cd "${REPO_ROOT}"

BASE_VERSION="$(sed -nE 's/^VERSION[[:space:]]*=[[:space:]]*["'\'']([^"'\'']+)["'\''].*/\1/p' version.bzl)"
readonly BASE_VERSION
if [[ -z "${BASE_VERSION}" ]]; then
  echo "ERROR: Could not parse VERSION from version.bzl." >&2
  exit 1
fi

if [[ -n "${RC_VERSION}" ]]; then
  if [[ ! "${RC_VERSION}" =~ ^rc[0-9]+$ ]]; then
    echo "ERROR: Invalid XSE_RC_VERSION '${RC_VERSION}' (expected rcN)." >&2
    exit 1
  fi
  if [[ ! "${BASE_VERSION}" =~ ^[0-9]+\.[0-9]+\.[0-9]+(\.[0-9]+)?$ ]]; then
    echo "ERROR: version.bzl VERSION '${BASE_VERSION}' is not a release version." >&2
    exit 1
  fi
  TARGET_VERSION="${BASE_VERSION}"
else
  git config --global --add safe.directory "${REPO_ROOT}" || true
  commit_ts="$(git show -s --format=%ct HEAD)"
  nightly_date="$(TZ='America/Los_Angeles' date -d "@${commit_ts}" +%Y%m%d)"
  TARGET_VERSION="${BASE_VERSION}.dev${nightly_date}"
fi
readonly TARGET_VERSION

set_version_bzl() {
  local new_version="$1"
  sed -i -E "s/^VERSION[[:space:]]*=[[:space:]]*[\"'][^\"']+[\"']/VERSION = \"${new_version}\"/" version.bzl
  grep -q "^VERSION = \"${new_version}\"$" version.bzl
}

build_wheel_set() {
  local target_version="$1"
  local out_dir="$2"
  local warm_test_wheel="${3:-false}"
  if [[ "${warm_test_wheel}" == "true" ]]; then
    bazel build "${BAZEL_FLAGS[@]}" //data:xla_serving_data_test
  fi
  set_version_bzl "${target_version}"
  bazel build "${BAZEL_FLAGS[@]}" //data:xla_serving_data //runtime/python:xla_saved_model_runner
  cp bazel-bin/data/xla_serving_data-"${target_version}"-*.whl "${out_dir}/"
  cp bazel-bin/runtime/python/xla_saved_model_runner-"${target_version}"-*.whl "${out_dir}/"
  if [[ "${XSE_PYTHON_VERSION}" == "${PURE_WHEEL_PYTHON}" ]]; then
    local pure_out_dir
    pure_out_dir="$(mktemp -d)"
    uv build --no-build-isolation --wheel --out-dir "${pure_out_dir}"
    cp "${pure_out_dir}"/xla_saved_model-"${target_version}"-*.whl "${out_dir}/"
    rm -rf "${pure_out_dir}"
  fi
}

REAL_BAZEL="$(command -v bazel || echo /usr/local/bin/bazel)"
readonly REAL_BAZEL

SHIM_DIR="$(mktemp -d)"
VENV_DIR="$(mktemp -d)"
readonly SHIM_DIR VENV_DIR
trap 'rm -rf "${SHIM_DIR}" "${VENV_DIR}"' EXIT

cat > "${SHIM_DIR}/bazel" <<EOF
#!/bin/bash
exec "${REAL_BAZEL}" --nosystem_rc "\$@"
EOF
chmod +x "${SHIM_DIR}/bazel"

if [[ "${XSE_PYTHON_VERSION}" == "${PURE_WHEEL_PYTHON}" ]]; then
  uv venv --clear --python "/usr/bin/python${XSE_PYTHON_VERSION}" "${VENV_DIR}"
  export VIRTUAL_ENV="${VENV_DIR}"
  export PATH="${SHIM_DIR}:${VENV_DIR}/bin:${PATH}"
  uv pip install --index-url "${PYPI_MIRROR_URL}" --require-hashes \
    --only-binary=:all: -r "${SCRIPT_DIR}/build_requirements_py312.txt"
else
  export PATH="${SHIM_DIR}:${PATH}"
fi

CACHE_CONFIG="remote_cache_readonly"
if [[ "${XSE_WRITE_TO_BAZEL_REMOTE_CACHE:-0}" == "1" ]]; then
  CACHE_CONFIG="remote_cache_readwrite"
fi
BAZEL_FLAGS=(
  -c opt
  "--@rules_python//python/config_settings:python_version=${XSE_PYTHON_VERSION}"
  "--config=${CACHE_CONFIG}"
  --output_filter=DONT_MATCH_ANYTHING
  --ui_event_filters=-warning
  --show_progress_rate_limit=30
  --curses=no
)
if [[ -n "${RC_VERSION}" ]]; then
  # Release builds never reuse remote cache hits.
  BAZEL_FLAGS+=(--noremote_accept_cached)
fi
readonly BAZEL_FLAGS

WHEEL_OUT_DIR="${XSE_WHEEL_OUTDIR:-${REPO_ROOT}/dist}"
readonly WHEEL_OUT_DIR
mkdir -p "${WHEEL_OUT_DIR}"

build_wheel_set "${TARGET_VERSION}" "${WHEEL_OUT_DIR}" "true"
if [[ -n "${RC_VERSION}" ]]; then
  build_wheel_set "${TARGET_VERSION}${RC_VERSION}" "${WHEEL_OUT_DIR}" "false"
fi

expected_wheels=2
if [[ "${XSE_PYTHON_VERSION}" == "${PURE_WHEEL_PYTHON}" ]]; then
  expected_wheels=3
fi
if [[ -n "${RC_VERSION}" ]]; then
  expected_wheels=$((expected_wheels * 2))
fi
shopt -s nullglob
built_wheels=("${WHEEL_OUT_DIR}"/*.whl)
shopt -u nullglob
if (( ${#built_wheels[@]} != expected_wheels )); then
  echo "ERROR: Expected ${expected_wheels} wheels in ${WHEEL_OUT_DIR}, found ${#built_wheels[@]}." >&2
  ls -la "${WHEEL_OUT_DIR}" >&2
  exit 1
fi

if command -v twine >/dev/null 2>&1; then
  twine check --strict "${built_wheels[@]}"
else
  /usr/bin/python3 -m twine check --strict "${built_wheels[@]}"
fi

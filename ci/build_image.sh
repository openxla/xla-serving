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
# Builds and pushes a single XLA Serving OCI image variant via Bazel.
set -exu -o pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
readonly SCRIPT_DIR
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
readonly REPO_ROOT

: "${XSE_DOCKER_VARIANT:?XSE_DOCKER_VARIANT must be set}"
: "${XSE_IMAGE_REPOSITORY:?XSE_IMAGE_REPOSITORY must be set}"
: "${XSE_IMAGE_TAG:?XSE_IMAGE_TAG must be set}"

cd "${REPO_ROOT}"

case "${XSE_DOCKER_VARIANT}" in
  cpu-jax-only)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher"
    VARIANT_BAZEL_FLAGS=(-c opt)
    ;;
  cpu-jax-with-tf)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher"
    VARIANT_BAZEL_FLAGS=(-c opt --define ENABLE_TF=1)
    ;;
  gpu-jax-cuda12)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher_gpu"
    VARIANT_BAZEL_FLAGS=(-c opt --config=cuda_v12 --verbose_failures)
    ;;
  gpu-jax-cuda12-with-tf)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher_gpu"
    VARIANT_BAZEL_FLAGS=(
      -c opt --config=cuda_v12 --define ENABLE_TF=1 --verbose_failures
    )
    ;;
  gpu-jax-cuda13)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher_gpu"
    VARIANT_BAZEL_FLAGS=(-c opt --config=cuda_v13 --verbose_failures)
    ;;
  gpu-jax-cuda13-with-tf)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher_gpu"
    VARIANT_BAZEL_FLAGS=(
      -c opt --config=cuda_v13 --define ENABLE_TF=1 --verbose_failures
    )
    ;;
  tpu-jax-only)
    DOCKER_TARGET="//runtime/serving/gcp/server:xla_serving_pusher_tpu"
    VARIANT_BAZEL_FLAGS=(-c opt)
    ;;
  *)
    echo "ERROR: Unsupported XSE_DOCKER_VARIANT: ${XSE_DOCKER_VARIANT}" >&2
    exit 1
    ;;
esac
readonly DOCKER_TARGET VARIANT_BAZEL_FLAGS

CACHE_CONFIG="remote_cache_readonly"
if [[ "${XSE_WRITE_TO_BAZEL_REMOTE_CACHE:-0}" == "1" ]]; then
  CACHE_CONFIG="remote_cache_readwrite"
fi
CACHE_FLAGS=("--config=${CACHE_CONFIG}")
if [[ "${XSE_RELEASE_BUILD:-false}" == "true" ]]; then
  CACHE_FLAGS+=(--noremote_accept_cached)
fi
readonly CACHE_FLAGS

bazel --nosystem_rc run "${VARIANT_BAZEL_FLAGS[@]}" \
  "${CACHE_FLAGS[@]}" \
  --local_ram_resources=80000 \
  --output_filter=DONT_MATCH_ANYTHING \
  --show_progress_rate_limit=30 \
  --curses=no \
  "${DOCKER_TARGET}" -- \
  --repository="${XSE_IMAGE_REPOSITORY}" \
  --tag="${XSE_IMAGE_TAG}"

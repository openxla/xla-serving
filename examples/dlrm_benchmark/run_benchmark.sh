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
# End-to-end orchestration of the XSE vs. pure-Python JAX serving benchmark,
# and of the XSE "CompositeProcessor (JCE) vs. TensorFlow preprocessor"
# comparison (xse-tf-* arms; need a server built with --define=ENABLE_TF=1).
#
# One invocation: exports the model once, then for every arm starts the server
# pinned to --server_cpus, verifies numerical parity, runs the closed-loop and
# open-loop sweeps with the load generator pinned to --client_cpus, stops the
# server and finally renders results/REPORT.md.
#
# Works from a  checkout (binaries under blaze-bin) and from the OSS
# xla-serving repository (python -m examples.dlrm_benchmark.*, run from an
# import root under <work_dir>/pysrc that contains only `examples`, so that
# the checkout's proto-less xla_saved_model/ source tree does not shadow the
# installed wheel). See README.md.
#
# TPU: the OSS xla_serving_main_tpu (binary or the distroless tpu-jax-only
# image, which contains no libtpu.so) loads the TPU PJRT plugin from
# $TPU_LIBRARY_PATH at startup and aborts before main() if it is unset. The
# script resolves the host's libtpu.so (--tpu_library_path, $TPU_LIBRARY_PATH,
# the venv's `libtpu` package from jax[tpu], /lib/libtpu.so), bind-mounts it
# into the container and sets TPU_LIBRARY_PATH for either launch mode.
#
# `docker run` does not inherit the caller's environment, so the script also
# forwards every exported TPU_* variable (except TPU_LIBRARY_PATH) and
# LIBTPU_INIT_ARGS into the container. On a VM that was not created by the
# Cloud TPU API (gcloud compute instances create --machine-type ct6e-...) the
# instance has no `tpu-env` metadata and libtpu would spin on the metadata
# server; the script then derives the single-host topology variables
# (TPU_SKIP_MDS_QUERY=1, TPU_ACCELERATOR_TYPE, TPU_CHIPS_PER_HOST_BOUNDS, ...)
# from the machine type. See configure_tpu_runtime_env and README.md.

set -euo pipefail

usage() {
  cat <<EOF
Usage: $0 --platform <cpu|cuda|tpu> [options]

Required:
  --platform P            Accelerator of the VM: cpu | cuda | tpu.

XSE server (one of; defaults to the locally built binary in  mode):
  --xse_bin PATH          xla_serving_main_{cpu,gpu,tpu} binary. For the
                          xse-tf-* arms it must be built with
                          --define=ENABLE_TF=1 ( cpu: done by this
                          script; the same binary then serves the JCE arms).
  --xse_image IMAGE       Docker image, e.g.
                          gcr.io/neptune-serving-engine/gpu-jax-cuda12:nightly-YYYYMMDD
                          (xse-tf-* arms: the *-with-tf variant, e.g.
                          gpu-jax-cuda12-with-tf; there is no TPU TF image).
  --tpu_library_path PATH Host libtpu.so for the OSS TPU server (image or
                          binary). Default: \$TPU_LIBRARY_PATH, else the venv's
                          libtpu package (jax[tpu]), else /lib/libtpu.so.
                          Other exported TPU_* variables and LIBTPU_INIT_ARGS
                          are forwarded into the container; on a VM without
                          tpu-env metadata (created with gcloud compute
                          instances create) the single-host topology set is
                          derived from the machine type when unset.

Fairness / placement:
  --server_cpus LIST      CPU list for the server (taskset / --cpuset-cpus).
  --client_cpus LIST      CPU list for the load generator (taskset).
  --num_threads N         Server request threads for *all* arms (default 8).
  --num_batch_threads N   Batch execution threads for xse-batch, xse-tf-*-batch,
                          py-batch and py-mp-batch (default: --num_threads;
                          py-mp-batch splits them over its frontends, at
                          least 1 each).
  --preprocessor_threads N
                          Internal parallelism of the CPU embedding lookup,
                          same for both preprocessors (default 8 = both
                          built-in defaults): JCE lookup worker pool
                          (export --jce_max_cpu_slots) and the TFRT work
                          queue of the TF preprocessor (server
                          --tfrt_num_threads, passed only when != 8 so that
                          older images still start).
  --batch_timeout_micros  Dynamic batching wait for the *-batch arms (1000).
  --max_enqueued_batches N
                          Batch queue admission control for the *-batch arms
                          (default 64): XSE max_enqueued_batches and the
                          Python batchers' equivalent (py-mp-batch splits the
                          budget over its frontends); beyond it requests fail
                          fast with UNAVAILABLE instead of queueing.
  --batch_sizes LIST      Exported batch sizes = PAD_UP buckets for every arm
                          (default: powers of two 1..4096).
  --num_processes N       Frontend processes of the py-mp arms (default:
                          --num_threads). Each loads its own copy of the
                          tables: needs N x table size of RAM.
  --channels_per_worker N gRPC connections per load generator worker, all
                          arms (default 8). SO_REUSEPORT (py-mp) balances
                          connections, not requests.
  --ring_mb N             Shared-memory ring per py-mp frontend (default 256).

Model:
  --embedding_size N      Embedding dimension (default 128).
  --max_vocab_size N      Cap per-table vocab; 0 = full ~6.2M rows (default 0).
  --skip_export           Reuse <work_dir>/model from a previous run.

Sweep:
  --arms LIST             Subset of
                            xse-batch, xse-nobatch          (JCE / CompositeProcessor)
                            xse-tf-batch, xse-tf-nobatch    (TF SparseSegmentSum lookup)
                            xse-tf-gather-batch, xse-tf-gather-nobatch
                                                            (TF ResourceGather+Sum lookup)
                            py, py-batch, py-mp, py-mp-batch
                          Default: the two JCE and the four Python arms
                          ("xse" = xse-batch). "CompositeProcessor vs TF"
                          preset: --arms xse-batch,xse-nobatch,xse-tf-batch,xse-tf-nobatch
                          (add the xse-tf-gather-* arms for the dense-gather
                          TF variant). The xse-tf-* arms need a TF-enabled
                          server, see above.
  --request_batch_sizes   Examples per request (default 1,16,128).
  --concurrencies LIST    Closed-loop in-flight requests (default 1,4,16,64,256).
  --open_loop_fractions   Open-loop QPS as fraction of peak (default 0.5,0.8).
  --duration S            Measured seconds per point (default 30).
  --warmup S              Discarded seconds per point (default 5).
  --repeats N             Repeats per point, median reported (default 3).
  --num_workers N         Load generator processes (default 8).

Misc:
  --work_dir DIR          Default /tmp/dlrm_benchmark.
  --port N                Server port (default 8500).
  --no_build               mode: do not run blaze build.
  --dry_run               Print the commands instead of running them.
EOF
  exit 1
}

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
PLATFORM=""
XSE_BIN=""
XSE_IMAGE=""
TPU_LIBRARY_PATH_HOST=""
SERVER_CPUS=""
CLIENT_CPUS=""
NUM_THREADS=8
NUM_BATCH_THREADS=""   # default: NUM_THREADS
PREPROCESSOR_THREADS=8 # JCE max_cpu_slots == XSE's TFRT work queue default
BATCH_TIMEOUT_MICROS=1000
MAX_ENQUEUED_BATCHES=64
BATCH_SIZES=""         # default: the exporter's (powers of two)
NUM_PROCESSES=""       # default: NUM_THREADS
CHANNELS_PER_WORKER=8
RING_MB=256
EMBEDDING_SIZE=128
MAX_VOCAB_SIZE=0
SKIP_EXPORT=0
ARMS="xse-batch,xse-nobatch,py,py-batch,py-mp,py-mp-batch"
REQUEST_BATCH_SIZES="1,16,128"
CONCURRENCIES="1,4,16,64,256"
OPEN_LOOP_FRACTIONS="0.5,0.8"
DURATION=30
WARMUP=5
REPEATS=3
NUM_WORKERS=8
WORK_DIR="/tmp/dlrm_benchmark"
PORT=8500
NO_BUILD=0
DRY_RUN=0
MODEL_NAME="dlrm"
# Where the host's libtpu.so is mounted inside the container.
DOCKER_LIBTPU="/usr/lib/libtpu.so"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --platform) PLATFORM="$2"; shift 2 ;;
    --xse_bin) XSE_BIN="$2"; shift 2 ;;
    --xse_image) XSE_IMAGE="$2"; shift 2 ;;
    --tpu_library_path) TPU_LIBRARY_PATH_HOST="$2"; shift 2 ;;
    --server_cpus) SERVER_CPUS="$2"; shift 2 ;;
    --client_cpus) CLIENT_CPUS="$2"; shift 2 ;;
    --num_threads) NUM_THREADS="$2"; shift 2 ;;
    --num_batch_threads) NUM_BATCH_THREADS="$2"; shift 2 ;;
    --preprocessor_threads) PREPROCESSOR_THREADS="$2"; shift 2 ;;
    --batch_timeout_micros) BATCH_TIMEOUT_MICROS="$2"; shift 2 ;;
    --max_enqueued_batches) MAX_ENQUEUED_BATCHES="$2"; shift 2 ;;
    --batch_sizes) BATCH_SIZES="$2"; shift 2 ;;
    --num_processes) NUM_PROCESSES="$2"; shift 2 ;;
    --channels_per_worker) CHANNELS_PER_WORKER="$2"; shift 2 ;;
    --ring_mb) RING_MB="$2"; shift 2 ;;
    --embedding_size) EMBEDDING_SIZE="$2"; shift 2 ;;
    --max_vocab_size) MAX_VOCAB_SIZE="$2"; shift 2 ;;
    --skip_export) SKIP_EXPORT=1; shift ;;
    --arms) ARMS="$2"; shift 2 ;;
    --request_batch_sizes) REQUEST_BATCH_SIZES="$2"; shift 2 ;;
    --concurrencies) CONCURRENCIES="$2"; shift 2 ;;
    --open_loop_fractions) OPEN_LOOP_FRACTIONS="$2"; shift 2 ;;
    --duration) DURATION="$2"; shift 2 ;;
    --warmup) WARMUP="$2"; shift 2 ;;
    --repeats) REPEATS="$2"; shift 2 ;;
    --num_workers) NUM_WORKERS="$2"; shift 2 ;;
    --work_dir) WORK_DIR="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no_build) NO_BUILD=1; shift ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage ;;
    *) echo "Unknown argument: $1" >&2; usage ;;
  esac
done
: "${NUM_BATCH_THREADS:=${NUM_THREADS}}"
: "${NUM_PROCESSES:=${NUM_THREADS}}"

# `xse` was the name of the batched XSE arm before the arms were spelled out;
# keep accepting it (results land under results/xse-batch).
if [[ ",${ARMS}," == *",xse,"* ]]; then
  echo "note: arm 'xse' is now called 'xse-batch'" >&2
  ARMS="$(echo ",${ARMS}," | sed 's/,xse,/,xse-batch,/; s/^,//; s/,$//')"
fi
# Which TF-preprocessor exports the selected arms need (lookup x batching);
# NEED_TF=1 if any xse-tf-* arm is selected (TF-enabled server required).
NEED_TF=0
NEED_TF_FUSED_BATCHED=0; NEED_TF_FUSED_UNBATCHED=0
NEED_TF_GATHER_BATCHED=0; NEED_TF_GATHER_UNBATCHED=0
for arm in ${ARMS//,/ }; do
  case "${arm}" in
    xse-batch|xse-nobatch|py|py-batch|py-mp|py-mp-batch) ;;
    xse-tf-batch) NEED_TF=1; NEED_TF_FUSED_BATCHED=1 ;;
    xse-tf-nobatch) NEED_TF=1; NEED_TF_FUSED_UNBATCHED=1 ;;
    xse-tf-gather-batch) NEED_TF=1; NEED_TF_GATHER_BATCHED=1 ;;
    xse-tf-gather-nobatch) NEED_TF=1; NEED_TF_GATHER_UNBATCHED=1 ;;
    *) echo "Unknown arm ${arm}" >&2; usage ;;
  esac
done

case "${PLATFORM}" in
  cpu|cuda|tpu) ;;
  *) echo "--platform must be cpu, cuda or tpu" >&2; usage ;;
esac

# Pure argument validation, so it fails before the build / docker pull: there
# is no prebuilt TPU image with TF support (see "TF support of the server").
if [[ "${NEED_TF}" == 1 && -n "${XSE_IMAGE}" && "${PLATFORM}" == "tpu" ]]; then
  cat >&2 <<EOF
The xse-tf-* arms need an XSE server built with TensorFlow support, and no
prebuilt TPU image with TF exists (push-nightly-serving-images.yml publishes
cpu-jax-with-tf and gpu-jax-cuda1{2,3}-with-tf, but only tpu-jax-only).
Build the TPU server from source with TF enabled and pass it with --xse_bin:
  bazel build -c opt --define=ENABLE_TF=1 //runtime/serving/gcp/server:xla_serving_main_tpu
or drop the xse-tf-* arms. See README.md, "TF preprocessor vs. CompositeProcessor".
EOF
  [[ "${DRY_RUN}" == 1 ]] || exit 1
fi

# ---------------------------------------------------------------------------
# Locate the tools:  (blaze-bin) or OSS (python -m).
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PKG_REL="third_party/neptune/examples/dlrm_benchmark"
MODE="oss"
if [[ "${SCRIPT_DIR}" == *//${PKG_REL} ]]; then
  MODE=""
  G3_ROOT="${SCRIPT_DIR%/${PKG_REL}}"
  BIN_DIR="${G3_ROOT}/blaze-bin/${PKG_REL}"
  SERVER_BIN_DIR="${G3_ROOT}/blaze-bin/third_party/neptune/runtime/serving/gcp/server"
else
  OSS_ROOT="${SCRIPT_DIR%/examples/dlrm_benchmark}"
fi

# Runs one of the python tools of this directory with the given flags.
#
# OSS mode runs `python3 -m examples.dlrm_benchmark.<name>` from PY_ROOT (set
# below once WORK_DIR is known), a directory containing only a symlink to the
# checkout's `examples/`. `python -m` puts the cwd first on sys.path; running
# from the repo root would resolve `xla_saved_model` to the checkout's source
# tree, which has no generated `protos/` package, instead of the installed
# wheel ("ModuleNotFoundError: No module named 'xla_saved_model.protos'").
tool() {
  local name="$1"; shift
  if [[ "${MODE}" == "" ]]; then
    "${BIN_DIR}/${name}" "$@"
  else
    (cd "${PY_ROOT}" && python3 -m "examples.dlrm_benchmark.${name}" "$@")
  fi
}

# Same as `tool` but returns the argv (for taskset / backgrounding).
tool_argv() {
  local name="$1"; shift
  if [[ "${MODE}" == "" ]]; then
    echo "${BIN_DIR}/${name}" "$@"
  else
    echo python3 -m "examples.dlrm_benchmark.${name}" "$@"
  fi
}

# Prints the arguments shell-quoted where needed (spaces, quotes, $, ...), so
# that --dry_run output stays copy-pasteable; plain flags, paths, comma lists
# and the <PLACEHOLDER> markers are printed as they are.
quoted() {
  local arg out="" plain='^[[:alnum:]_./:=,@%+<>-]*$'
  for arg in "$@"; do
    if [[ -n "${arg}" && "${arg}" =~ ${plain} ]]; then
      out+=" ${arg}"
    else
      out+=" $(printf '%q' "${arg}")"
    fi
  done
  echo "${out# }"
}

run() {
  if [[ "$1" == "tool" ]]; then
    # Show the resolved command (and, in OSS mode, the import root) rather
    # than the `tool` helper so that --dry_run output is copy-pasteable.
    if [[ "${MODE}" == "" ]]; then
      echo "+ $(tool_argv "${@:2}")" >&2
    else
      echo "+ (cd ${PY_ROOT} && $(tool_argv "${@:2}"))" >&2
    fi
  else
    # e.g. a forwarded LIBTPU_INIT_ARGS with several flags needs quotes.
    echo "+ $(quoted "$@")" >&2
  fi
  if [[ "${DRY_RUN}" == 0 ]]; then
    "$@"
  fi
}

case "${PLATFORM}" in
  cpu) XSE_TARGET="xla_serving_main_cpu" ;;
  cuda) XSE_TARGET="xla_serving_main_gpu" ;;
  tpu) XSE_TARGET="xla_serving_main_tpu" ;;
esac

if [[ "${MODE}" == "" && "${NO_BUILD}" == 0 && "${DRY_RUN}" == 0 ]]; then
  targets=("//${PKG_REL}:all")
  build_flags=(-c opt)
  if [[ -z "${XSE_BIN}" && -z "${XSE_IMAGE}" && "${PLATFORM}" == "cpu" ]]; then
    targets+=("//third_party/neptune/runtime/serving/gcp/server:${XSE_TARGET}")
  fi
  # A TF preprocessor needs the server built with TF support. --define does
  # not change the output directory, so the TF-enabled binary lands at the
  # same blaze-bin path and serves the JCE arms of this run too: every XSE arm
  # runs on one and the same binary.
  if [[ "${NEED_TF}" == 1 ]]; then
    build_flags+=(--define=ENABLE_TF=1)
  fi
  (cd "${G3_ROOT}" && SKYBUILD=1 blaze build "${build_flags[@]}" "${targets[@]}")
fi

if [[ -z "${XSE_BIN}" && -z "${XSE_IMAGE}" ]]; then
  if [[ "${MODE}" == "" ]]; then
    XSE_BIN="${SERVER_BIN_DIR}/${XSE_TARGET}"
  elif [[ -x "${OSS_ROOT}/bazel-bin/runtime/serving/gcp/server/${XSE_TARGET}" ]]; then
    XSE_BIN="${OSS_ROOT}/bazel-bin/runtime/serving/gcp/server/${XSE_TARGET}"
  fi
fi
if [[ -z "${XSE_IMAGE}" && ! -x "${XSE_BIN}" && "${DRY_RUN}" == 0 ]]; then
  echo "XSE server binary not found (${XSE_BIN:-unset}); pass --xse_bin or" \
    "--xse_image. See README.md for how to build it." >&2
  exit 1
fi
if [[ -n "${XSE_IMAGE}" && "${DRY_RUN}" == 0 ]]; then
  if ! command -v docker >/dev/null; then
    echo "--xse_image requires docker on PATH." >&2
    exit 1
  fi
  if ! docker info >/dev/null 2>&1; then
    echo "docker is installed but not usable by $(id -un); add yourself to the" \
      "docker group (sudo usermod -aG docker $(id -un); re-login) or run as root." >&2
    exit 1
  fi
  # Pull up front (docker run would too) so that env.txt can record the digest
  # and the model export does not run before a bad tag is detected.
  if ! docker image inspect "${XSE_IMAGE}" >/dev/null 2>&1; then
    run docker pull "${XSE_IMAGE}"
  fi
fi

# ---------------------------------------------------------------------------
# TF support of the server (xse-tf-* arms).
# ---------------------------------------------------------------------------
# A JAX-only server (default build, *-jax-only images) fails to load a model
# whose preprocessor is a TF SavedModel:
#   No `OptionalFunctionHandler` is found for key: ...
# wait_for_port stops on that line; the checks below catch the common cases
# before the (long) export. XSE_TF_SUPPORT goes into env.txt.
XSE_TF_SUPPORT="n/a"
if [[ "${NEED_TF}" == 1 ]]; then
  XSE_TF_SUPPORT="unknown"
  if [[ -n "${XSE_IMAGE}" ]]; then
    # (--platform tpu with an image was already rejected above.)
    if [[ "${XSE_IMAGE}" == *-with-tf* ]]; then
      XSE_TF_SUPPORT="yes (image name)"
    else
      echo "warning: --arms contains xse-tf-* arms but --xse_image ${XSE_IMAGE}" \
        "does not look like a TF-enabled image (*-with-tf, e.g." \
        "gcr.io/neptune-serving-engine/gpu-jax-cuda12-with-tf:nightly-YYYYMMDD);" \
        "a JAX-only server cannot load the TF preprocessor." >&2
      XSE_TF_SUPPORT="no (image name)"
    fi
  elif [[ -x "${XSE_BIN}" ]]; then
    # The TF SavedModel runner is only linked into ENABLE_TF=1 builds; its
    # load-time log message is a cheap marker.
    if grep -q -a -m1 'Using TFRT to load TF SavedModel' "${XSE_BIN}" 2>/dev/null; then
      XSE_TF_SUPPORT="yes (binary)"
    else
      echo "warning: --arms contains xse-tf-* arms but ${XSE_BIN} does not look" \
        "like it was built with --define=ENABLE_TF=1; a JAX-only server cannot" \
        "load the TF preprocessor (rebuild with that define, or drop the" \
        "xse-tf-* arms)." >&2
      XSE_TF_SUPPORT="no (binary)"
    fi
  fi
fi

# ---------------------------------------------------------------------------
# libtpu for the OSS TPU server.
# ---------------------------------------------------------------------------
# The OSS xla_serving_main_tpu registers the TPU PJRT plugin dynamically: a
# static initializer dlopen()s $TPU_LIBRARY_PATH and CHECK-fails before main()
# when it is unset ("Check failed: RegisterDynamicPjrtPlugin(kTpuPjrtName,
# "TPU_LIBRARY_PATH") is OK (NOT_FOUND: Environment variable TPU_LIBRARY_PATH
# is not set"). The tpu-jax-only image is distroless and ships no libtpu.so,
# so the host's copy is bind-mounted into the container. The  binary
# links libtpu statically and needs none of this.
#
# Sets TPU_LIBRARY_PATH_HOST from, in order: --tpu_library_path,
# $TPU_LIBRARY_PATH, the venv's `libtpu` package (pulled in by jax[tpu]),
# /lib/libtpu.so (present on some TPU VM images).
resolve_tpu_library_path() {
  local candidate=""
  if [[ -z "${TPU_LIBRARY_PATH_HOST}" && -n "${TPU_LIBRARY_PATH:-}" ]]; then
    TPU_LIBRARY_PATH_HOST="${TPU_LIBRARY_PATH}"
  fi
  if [[ -z "${TPU_LIBRARY_PATH_HOST}" ]]; then
    candidate="$(python3 - 2>/dev/null <<'PY'
import os
import libtpu
try:
  print(libtpu.get_library_path())
except AttributeError:  # older wheels
  print(os.path.join(os.path.dirname(libtpu.__file__), 'libtpu.so'))
PY
)" || candidate=""
    if [[ -n "${candidate}" && -f "${candidate}" ]]; then
      TPU_LIBRARY_PATH_HOST="${candidate}"
    fi
  fi
  if [[ -z "${TPU_LIBRARY_PATH_HOST}" && -f /lib/libtpu.so ]]; then
    TPU_LIBRARY_PATH_HOST="/lib/libtpu.so"
  fi
  # Canonical path (symlinked venvs, /lib -> /usr/lib) for the bind mount and
  # env.txt.
  if [[ -f "${TPU_LIBRARY_PATH_HOST}" ]]; then
    TPU_LIBRARY_PATH_HOST="$(readlink -f "${TPU_LIBRARY_PATH_HOST}")"
  fi
}

# ---------------------------------------------------------------------------
# TPU runtime environment (topology) for the server.
# ---------------------------------------------------------------------------
# libtpu takes its topology from TPU_CHIPS_PER_HOST_BOUNDS / TPU_HOST_BOUNDS /
# TPU_ACCELERATOR_TYPE / TPU_WORKER_ID and, when they are unset, from the GCE
# instance metadata attribute `tpu-env`. Only VMs created by the Cloud TPU API
# (gcloud compute tpus tpu-vm create) carry that attribute. On a VM created
# with `gcloud compute instances create --machine-type ct6e-standard-1t` the
# lookup returns 404; libtpu retries it 30 x 1 s per variable, so the server
# is silent for minutes and then logs
#   Failed to convert value for key 'CHIPS_PER_HOST_BOUNDS' in 'tpu-env' to
#   string: INTERNAL: Failed to fetch URL after 30 tries (http status: 404)
# TPU_SKIP_MDS_QUERY=1 disables the metadata lookups, and the single-host
# values derived below are the ones documented for TPU containers without
# `tpu-env` (https://cloud.google.com/dataflow/docs/tpu/use-tpus).
#
# `docker run` does not inherit the caller's environment, so whatever the host
# has (or this preflight derives) is passed to the container explicitly:
# TPU_FORWARD_ENV holds NAME=VALUE for every exported TPU_* variable except
# TPU_LIBRARY_PATH (the container gets its own, DOCKER_LIBTPU) plus
# LIBTPU_INIT_ARGS. Host-image specifics such as
# TPU_VISIBLE_DEVICE_PATHS=/dev/vfio/0 or
# LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true (stock Ubuntu
# image: vfio-bound chip, no tpunetd) are forwarded but never derived. The
# --xse_bin and Python arms inherit the shell environment as usual.
TPU_FORWARD_ENV=()
TPU_ENV_METADATA="unknown"  # present | missing | unknown      (env.txt)
TPU_ENV_SOURCE="none"       # host | metadata | derived(<machine type>) | none
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance"

# Prints "<TPU_ACCELERATOR_TYPE> <TPU_CHIPS_PER_HOST_BOUNDS>" for a single-host
# ct* machine type; fails for anything else.
tpu_topology_for_machine_type() {
  local machine_type="$1" accel chips bounds
  case "${machine_type}" in
    ct6e-standard-[148]t) accel="v6e-" ;;
    ct5lp-hightpu-[148]t) accel="v5litepod-" ;;
    # 4 chips x 2 TensorCores: the accelerator type counts cores.
    ct5p-hightpu-4t) echo "v5p-8 2,2,1"; return 0 ;;
    ct4p-hightpu-4t) echo "v4-8 2,2,1"; return 0 ;;
    *) return 1 ;;
  esac
  chips="${machine_type##*-}"
  chips="${chips%t}"
  case "${chips}" in
    1) bounds="1,1,1" ;;
    4) bounds="2,2,1" ;;
    8) bounds="2,4,1" ;;
  esac
  echo "${accel}${chips} ${bounds}"
}

# Prints the HTTP status of a metadata path (000 when unreachable / no curl).
metadata_status() {
  # Test hook: DLRM_BENCHMARK_FAKE_METADATA=<tpu-env status>:<machine type>.
  if [[ -n "${DLRM_BENCHMARK_FAKE_METADATA:-}" ]]; then
    echo "${DLRM_BENCHMARK_FAKE_METADATA%%:*}"
    return 0
  fi
  command -v curl >/dev/null || { echo 000; return 0; }
  # curl prints the (000) code even when it fails, so do not echo a second one.
  local code
  code="$(curl -s -m 2 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' \
    "${METADATA_URL}/$1" 2>/dev/null)" || true
  echo "${code:-000}"
}

# Prints the VM's machine type (last path component), or nothing.
metadata_machine_type() {
  if [[ -n "${DLRM_BENCHMARK_FAKE_METADATA:-}" ]]; then
    echo "${DLRM_BENCHMARK_FAKE_METADATA#*:}"
    return 0
  fi
  local full
  full="$(curl -sf -m 2 -H 'Metadata-Flavor: Google' \
    "${METADATA_URL}/machine-type" 2>/dev/null || true)"
  echo "${full##*/}"
}

configure_tpu_runtime_env() {
  local status machine_type topology accel bounds name
  if [[ -n "${TPU_SKIP_MDS_QUERY:-}" ]]; then
    # The host is already configured (e.g. /etc/environment); nothing to
    # derive, just forward.
    TPU_ENV_SOURCE="host"
  else
    status="$(metadata_status attributes/tpu-env)"
    case "${status}" in
      200) TPU_ENV_METADATA="present"; TPU_ENV_SOURCE="metadata" ;;
      404)
        TPU_ENV_METADATA="missing"
        machine_type="$(metadata_machine_type)"
        if topology="$(tpu_topology_for_machine_type "${machine_type}")"; then
          read -r accel bounds <<< "${topology}"
          # User-set values win; only fill the gaps.
          export TPU_SKIP_MDS_QUERY=1
          : "${TPU_HOST_BOUNDS:=1,1,1}"; export TPU_HOST_BOUNDS
          : "${TPU_WORKER_HOSTNAMES:=localhost}"; export TPU_WORKER_HOSTNAMES
          : "${TPU_WORKER_ID:=0}"; export TPU_WORKER_ID
          : "${TPU_ACCELERATOR_TYPE:=${accel}}"; export TPU_ACCELERATOR_TYPE
          : "${TPU_CHIPS_PER_HOST_BOUNDS:=${bounds}}"; export TPU_CHIPS_PER_HOST_BOUNDS
          TPU_ENV_SOURCE="derived(${machine_type})"
          echo "This VM has no tpu-env instance metadata (not created by the Cloud TPU" \
            "API); using the single-host TPU topology for ${machine_type}:" \
            "TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE}" \
            "TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}" \
            "TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS} TPU_WORKER_ID=${TPU_WORKER_ID}" \
            "TPU_WORKER_HOSTNAMES=${TPU_WORKER_HOSTNAMES}" >&2
        else
          cat >&2 <<EOF
This VM has no tpu-env instance metadata (it was not created by the Cloud TPU
API) and its machine type "${machine_type:-unknown}" is not a known single-host
TPU type (ct6e-standard-{1,4,8}t, ct5lp-hightpu-{1,4,8}t, ct5p-hightpu-4t).
Without the topology libtpu retries the metadata server 30 x 1 s per lookup
and the TPU server ends with
  Failed to convert value for key 'CHIPS_PER_HOST_BOUNDS' in 'tpu-env' to
  string: INTERNAL: Failed to fetch URL after 30 tries (http status: 404)

Export the topology for this machine before running the script, e.g. for a
single v6e chip:
  export TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
  export TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0
(table: https://cloud.google.com/dataflow/docs/tpu/use-tpus), or create the VM
with the Cloud TPU API / the TPU image family; see README.md.
EOF
          if [[ "${DRY_RUN}" == 0 ]]; then
            exit 1
          fi
          echo "warning: continuing without TPU topology variables (--dry_run)." >&2
        fi
        ;;
      *) ;;  # not on GCE or no answer: leave libtpu to its own devices.
    esac
  fi
  # Collect what the container must see. `compgen -e` lists exported names.
  while IFS= read -r name; do
    [[ "${name}" == "TPU_LIBRARY_PATH" ]] && continue
    TPU_FORWARD_ENV+=("${name}=${!name:-}")
  done < <(compgen -e | grep -E '^(TPU_|LIBTPU_INIT_ARGS$)' | sort || true)
}

if [[ "${PLATFORM}" == "tpu" ]]; then
  resolve_tpu_library_path
  configure_tpu_runtime_env
  needs_libtpu=0
  [[ -n "${XSE_IMAGE}" || "${MODE}" == "oss" ]] && needs_libtpu=1
  if [[ "${DRY_RUN}" == 0 && -n "${TPU_LIBRARY_PATH_HOST}" \
        && ! -f "${TPU_LIBRARY_PATH_HOST}" ]]; then
    echo "libtpu.so not found at ${TPU_LIBRARY_PATH_HOST}" \
      "(from --tpu_library_path / \$TPU_LIBRARY_PATH)." >&2
    exit 1
  fi
  if [[ "${needs_libtpu}" == 1 && -z "${TPU_LIBRARY_PATH_HOST}" ]]; then
    if [[ "${DRY_RUN}" == 0 ]]; then
      cat >&2 <<EOF
No libtpu.so found for the TPU server.

The OSS xla_serving_main_tpu (${XSE_IMAGE:-${XSE_BIN}}) loads the TPU PJRT
plugin from \$TPU_LIBRARY_PATH when it starts and aborts without it:
  Check failed: RegisterDynamicPjrtPlugin(kTpuPjrtName, "TPU_LIBRARY_PATH") is OK
  (NOT_FOUND: Environment variable TPU_LIBRARY_PATH is not set. ...)
The tpu-jax-only image does not contain libtpu.so, so the host's copy is
mounted into the container.

Install it into the active venv (it is part of jax[tpu]; see README.md):
  uv pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
or point the script at an existing one:
  --tpu_library_path /path/to/libtpu.so   (or export TPU_LIBRARY_PATH=...)
EOF
      exit 1
    fi
    echo "warning: no libtpu.so found; using the <TPU_LIBRARY_PATH> placeholder" \
      "in the printed commands (--dry_run)." >&2
    TPU_LIBRARY_PATH_HOST="<TPU_LIBRARY_PATH>"
  fi
fi
# `env TPU_LIBRARY_PATH=...` prefix for a TPU server started as a binary.
SERVER_ENV=()
if [[ "${PLATFORM}" == "tpu" && -n "${TPU_LIBRARY_PATH_HOST}" ]]; then
  SERVER_ENV=(env "TPU_LIBRARY_PATH=${TPU_LIBRARY_PATH_HOST}")
fi

if [[ -n "${SERVER_CPUS}" ]]; then
  SERVER_TASKSET=(taskset -c "${SERVER_CPUS}")
else
  SERVER_TASKSET=()
fi
if [[ -n "${CLIENT_CPUS}" ]]; then
  CLIENT_TASKSET=(taskset -c "${CLIENT_CPUS}")
else
  CLIENT_TASKSET=()
fi

mkdir -p "${WORK_DIR}"
# Absolute: the model directory is bind-mounted into the docker container.
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
MODEL_DIR="${WORK_DIR}/model"
RESULTS_DIR="${WORK_DIR}/results"
LOG_DIR="${WORK_DIR}/logs"
BATCHED_EXPORT="${MODEL_DIR}/xse_batched"
UNBATCHED_EXPORT="${MODEL_DIR}/xse_unbatched"
# TF-preprocessor exports (xse-tf-* arms), by lookup implementation.
TF_FUSED_BATCHED_EXPORT="${MODEL_DIR}/xse_tf_fused_batched"
TF_FUSED_UNBATCHED_EXPORT="${MODEL_DIR}/xse_tf_fused_unbatched"
TF_GATHER_BATCHED_EXPORT="${MODEL_DIR}/xse_tf_gather_batched"
TF_GATHER_UNBATCHED_EXPORT="${MODEL_DIR}/xse_tf_gather_unbatched"
WEIGHTS_DIR="${MODEL_DIR}/baseline_weights"
mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"
SERVER_ADDR="localhost:${PORT}"
DOCKER_NAME="dlrm-benchmark-xse-$$"

# OSS: isolated import root for `python -m examples.dlrm_benchmark.*` (see the
# comment on `tool`). A symlink rather than a copy, so it never goes stale.
PY_ROOT=""
if [[ "${MODE}" == "oss" ]]; then
  PY_ROOT="${WORK_DIR}/pysrc"
  mkdir -p "${PY_ROOT}"
  ln -sfn "${OSS_ROOT}/examples" "${PY_ROOT}/examples"
  if [[ "${DRY_RUN}" == 0 ]] && ! (cd "${PY_ROOT}" && \
      python3 -c 'import xla_saved_model.protos.grpc.predict_pb2' 2>/dev/null); then
    cat >&2 <<EOF
The installed xla_saved_model package has no generated protos
(import xla_saved_model.protos.grpc.predict_pb2 failed from ${PY_ROOT}).
This is what the benchmark tools fail with as
"ModuleNotFoundError: No module named 'xla_saved_model.protos'".

Install the release wheel into the active venv (see README.md):
  gh release download --repo AI-Hypercomputer/xla-serving --pattern '*.whl' --dir wheels
  uv pip install --force-reinstall wheels/xla_saved_model-*.whl

An editable install (\`pip install -e .\`) only contains the generated protos
if \`bazel\` was on PATH when it was installed (setup.py fetches the XLA protos
with it; pip merely warns when that fails). To build from the checkout
instead, with bazel available:
  uv build --wheel && uv pip install --force-reinstall dist/xla_saved_model-*.whl
EOF
    exit 1
  fi
fi

echo "mode=${MODE} platform=${PLATFORM} work_dir=${WORK_DIR}"
echo "arms=${ARMS} B=${REQUEST_BATCH_SIZES} C=${CONCURRENCIES}" \
  "repeats=${REPEATS} duration=${DURATION}s"

# ---------------------------------------------------------------------------
# Environment snapshot (for the report's reproducibility section).
# ---------------------------------------------------------------------------
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname)"
  echo "platform: ${PLATFORM}"
  echo "mode: ${MODE}"
  echo "nproc: $(nproc)"
  echo "server_cpus: ${SERVER_CPUS:-all}"
  echo "client_cpus: ${CLIENT_CPUS:-all}"
  echo "num_threads: ${NUM_THREADS}"
  echo "num_batch_threads: ${NUM_BATCH_THREADS}"
  echo "preprocessor_threads: ${PREPROCESSOR_THREADS}"
  echo "batch_timeout_micros: ${BATCH_TIMEOUT_MICROS}"
  echo "max_enqueued_batches: ${MAX_ENQUEUED_BATCHES}"
  echo "batch_sizes: ${BATCH_SIZES:-default (powers of two 1..4096)}"
  echo "num_processes: ${NUM_PROCESSES}"
  echo "channels_per_worker: ${CHANNELS_PER_WORKER}"
  echo "num_workers: ${NUM_WORKERS}"
  echo "ring_mb: ${RING_MB}"
  echo "xse: ${XSE_IMAGE:-${XSE_BIN}}"
  echo "xse_tf_arms: ${NEED_TF}"
  echo "xse_tf_support: ${XSE_TF_SUPPORT}"
  if [[ -n "${XSE_IMAGE}" ]] && docker image inspect "${XSE_IMAGE}" >/dev/null 2>&1; then
    # Pin the exact image behind a moving tag such as :nightly.
    docker image inspect --format \
      'xse_image_id: {{.Id}}{{"\n"}}xse_image_digests: {{join .RepoDigests ","}}' \
      "${XSE_IMAGE}" 2>/dev/null || true
  fi
  grep -m1 'model name' /proc/cpuinfo || true
  awk '/MemTotal|MemAvailable/ {printf "%s %.1f GiB\n", tolower($1), $2/1048576}' /proc/meminfo 2>/dev/null || true
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
  python3 -c 'import jax; print("jax:", jax.__version__); print("devices:", jax.devices())' 2>/dev/null || true
  if [[ "${NEED_TF}" == 1 ]]; then
    # The TF that built the preprocessor SavedModel (the server runs its own).
    python3 -c 'import tensorflow as tf; print("tensorflow:", tf.__version__)' 2>/dev/null || true
  fi
  if [[ "${PLATFORM}" == "tpu" ]]; then
    echo "tpu_library_path: ${TPU_LIBRARY_PATH_HOST:-unset}"
    python3 -c 'import importlib.metadata as m; print("libtpu:", m.version("libtpu"))' 2>/dev/null || true
    echo "tpu_env_metadata: ${TPU_ENV_METADATA}"
    echo "tpu_env_source: ${TPU_ENV_SOURCE}"
    echo "tpu_env: ${TPU_FORWARD_ENV[*]:-none}"
  fi
} > "${RESULTS_DIR}/env.txt" || true

# ---------------------------------------------------------------------------
# Export (once).
# ---------------------------------------------------------------------------
# The same batch buckets, batch threads and queue bound go into the XSE
# exports (where they are baked into BatchOptions; the TF exporter shares the
# JCE exporter's BatchOptions builder) and to the Python batchers (flags), so
# all batched arms run the identical batching configuration.
#
# The TF-preprocessor exports are built from the weights the JCE exporter
# wrote (tables.npz / params.npz), so every arm serves the same bytes and
# parity_check applies unchanged. Only the variants the selected arms need are
# exported (each full-size TF export is another ~3.2 GB on disk).
if [[ "${SKIP_EXPORT}" == 0 ]]; then
  rm -rf "${BATCHED_EXPORT}" "${UNBATCHED_EXPORT}" "${WEIGHTS_DIR}" \
    "${TF_FUSED_BATCHED_EXPORT}" "${TF_FUSED_UNBATCHED_EXPORT}" \
    "${TF_GATHER_BATCHED_EXPORT}" "${TF_GATHER_UNBATCHED_EXPORT}"
  batch_export_flags=(
    --platforms="${PLATFORM}"
    --batch_timeout_micros="${BATCH_TIMEOUT_MICROS}"
    --num_batch_threads="${NUM_BATCH_THREADS}"
    --max_enqueued_batches="${MAX_ENQUEUED_BATCHES}"
  )
  export_flags=(
    --batched_export_path="${BATCHED_EXPORT}"
    --unbatched_export_path="${UNBATCHED_EXPORT}"
    --baseline_weights_dir="${WEIGHTS_DIR}"
    --embedding_size="${EMBEDDING_SIZE}"
    --max_vocab_size="${MAX_VOCAB_SIZE}"
    --jce_max_cpu_slots="${PREPROCESSOR_THREADS}"
    "${batch_export_flags[@]}"
  )
  [[ -n "${BATCH_SIZES}" ]] && export_flags+=(--batch_sizes="${BATCH_SIZES}")
  run tool export_benchmark_model "${export_flags[@]}"

  # TF exports: one exporter run per lookup implementation (the tables are
  # turned into tf.Variables once per run and shared by its two exports).
  tf_export() {
    local lookup="$1" batched_path="$2" unbatched_path="$3"
    local flags=(--weights_dir="${WEIGHTS_DIR}" --tf_lookup="${lookup}"
      "${batch_export_flags[@]}")
    [[ -n "${batched_path}" ]] && flags+=(--batched_export_path="${batched_path}")
    [[ -n "${unbatched_path}" ]] && flags+=(--unbatched_export_path="${unbatched_path}")
    run tool export_tf_benchmark_model "${flags[@]}"
  }
  if [[ "${NEED_TF_FUSED_BATCHED}" == 1 || "${NEED_TF_FUSED_UNBATCHED}" == 1 ]]; then
    tf_export fused \
      "$([[ "${NEED_TF_FUSED_BATCHED}" == 1 ]] && echo "${TF_FUSED_BATCHED_EXPORT}")" \
      "$([[ "${NEED_TF_FUSED_UNBATCHED}" == 1 ]] && echo "${TF_FUSED_UNBATCHED_EXPORT}")"
  fi
  if [[ "${NEED_TF_GATHER_BATCHED}" == 1 || "${NEED_TF_GATHER_UNBATCHED}" == 1 ]]; then
    tf_export gather \
      "$([[ "${NEED_TF_GATHER_BATCHED}" == 1 ]] && echo "${TF_GATHER_BATCHED_EXPORT}")" \
      "$([[ "${NEED_TF_GATHER_UNBATCHED}" == 1 ]] && echo "${TF_GATHER_UNBATCHED_EXPORT}")"
  fi
fi

# ---------------------------------------------------------------------------
# Server lifecycle.
# ---------------------------------------------------------------------------
SERVER_PID=""
# 1 while the XSE container is the current server (SERVER_PID is then the
# container's host pid, which is 0 once it has exited).
SERVER_IN_DOCKER=0
MONITOR_PID=""

stop_monitor() {
  if [[ -n "${MONITOR_PID}" ]]; then
    kill -TERM "${MONITOR_PID}" 2>/dev/null || true
    for _ in $(seq 1 20); do
      kill -0 "${MONITOR_PID}" 2>/dev/null || break
      sleep 0.5
    done
    kill -KILL "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
    MONITOR_PID=""
  fi
}

stop_server() {
  stop_monitor
  if [[ "${SERVER_IN_DOCKER}" == 1 ]]; then
    # -a: also remove a container that already exited (e.g. crashed at startup).
    if docker ps -aq -f "name=^${DOCKER_NAME}$" | grep -q .; then
      docker rm -f "${DOCKER_NAME}" >/dev/null 2>&1 || true
    fi
    SERVER_IN_DOCKER=0
  fi
  # An exited container reports pid 0; never signal that (it is our own
  # process group).
  if [[ -n "${SERVER_PID}" && "${SERVER_PID}" != 0 ]]; then
    kill -TERM "${SERVER_PID}" 2>/dev/null || true
    for _ in $(seq 1 30); do
      kill -0 "${SERVER_PID}" 2>/dev/null || break
      sleep 1
    done
    kill -KILL "${SERVER_PID}" 2>/dev/null || true
    wait "${SERVER_PID}" 2>/dev/null || true
  fi
  SERVER_PID=""
}
trap stop_server EXIT INT TERM

# True while the server started by start_server is still running.
server_alive() {
  if [[ "${SERVER_IN_DOCKER}" == 1 ]]; then
    [[ "$(docker inspect -f '{{.State.Running}}' "${DOCKER_NAME}" 2>/dev/null)" == "true" ]]
  else
    [[ -n "${SERVER_PID}" && "${SERVER_PID}" != 0 ]] && kill -0 "${SERVER_PID}" 2>/dev/null
  fi
}

# Explains the "No `OptionalFunctionHandler` is found for key" load error: the
# model has a TF preprocessor but the server was built without TF support.
tf_handler_hint() {
  local log="$1" line
  line="$(grep -m1 'No `OptionalFunctionHandler` is found for key' "${log}" 2>/dev/null)" || return 0
  cat >&2 <<EOF

The server could not load the model's TensorFlow preprocessor:
  ${line}
It is a JAX-only build. The xse-tf-* arms need a server with TF support:
  : blaze build -c opt --define=ENABLE_TF=1 //third_party/neptune/runtime/serving/gcp/server:${XSE_TARGET}
           (run_benchmark.sh does this itself for --platform cpu without --xse_bin)
  OSS:     bazel build -c opt [--config=cuda_v12] --define=ENABLE_TF=1 //runtime/serving/gcp/server:${XSE_TARGET}
           or a *-with-tf image (cpu-jax-with-tf, gpu-jax-cuda12-with-tf; none for TPU).
See README.md, "TF preprocessor vs. CompositeProcessor".
EOF
}

wait_for_port() {
  local log="$1"
  local deadline=$(( $(date +%s) + 1800 ))
  until (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; do
    if ! server_alive; then
      echo "Server exited before opening port ${PORT}; see ${log}" >&2
      sleep 1  # let the log redirect / `docker logs -f` flush
      echo "--- last lines of ${log} ---" >&2
      tail -n 20 "${log}" >&2 2>/dev/null || true
      tf_handler_hint "${log}"
      return 1
    fi
    # A JAX-only server does not always exit on this; do not wait it out.
    if grep -q 'No `OptionalFunctionHandler` is found for key' "${log}" 2>/dev/null; then
      echo "--- last lines of ${log} ---" >&2
      tail -n 20 "${log}" >&2 2>/dev/null || true
      tf_handler_hint "${log}"
      return 1
    fi
    # libtpu keeps running (with no topology) after giving up on the metadata
    # server, so the check above never fires; do not wait out the deadline.
    if grep -q 'Failed to fetch URL after 30 tries' "${log}" 2>/dev/null; then
      echo "TPU server cannot read its topology from the GCE metadata server" \
        "(no tpu-env attribute: this VM was not created by the Cloud TPU API)." >&2
      echo "Export TPU_SKIP_MDS_QUERY=1 and the TPU_* topology variables for" \
        "this machine type (see README.md, \"TPU runtime environment\");" \
        "the script forwards them into the container." >&2
      echo "--- last lines of ${log} ---" >&2
      tail -n 20 "${log}" >&2 2>/dev/null || true
      return 1
    fi
    if [[ $(date +%s) -gt ${deadline} ]]; then
      echo "Timed out waiting for port ${PORT}" >&2
      return 1
    fi
    sleep 1
  done
}

start_xse() {
  local model_path="$1" log="$2"
  local flags=(
    --model_name="${MODEL_NAME}"
    --orbax_grpc_server_platform="${PLATFORM}"
    --port="${PORT}"
    --use_insecure_grpc_credentials=true
    --warmup_server=true
    --thread_pool_num_threads="${NUM_THREADS}"
  )
  # The TF preprocessor's TFRT work queue; the server's built-in default is 8
  # (= the JCE default) and images older than the flag reject unknown flags,
  # so it is only passed when the benchmark actually changes it. Passed for
  # every XSE arm (the JCE arms ignore it) so all arms get one command line.
  if [[ "${PREPROCESSOR_THREADS}" != 8 ]]; then
    flags+=(--tfrt_num_threads="${PREPROCESSOR_THREADS}")
  fi
  if [[ -n "${XSE_IMAGE}" ]]; then
    local docker_args=(--name "${DOCKER_NAME}" --net=host --ulimit memlock=-1
      --mount "type=bind,src=${model_path},dst=/tmp/model")
    [[ -n "${SERVER_CPUS}" ]] && docker_args+=(--cpuset-cpus "${SERVER_CPUS}")
    case "${PLATFORM}" in
      cuda) docker_args+=(--gpus all -e PJRT_DEVICE=cuda
              -e NVIDIA_VISIBLE_DEVICES=all
              -e NVIDIA_DRIVER_CAPABILITIES=all,compute,utility
              -e LD_LIBRARY_PATH=/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64) ;;
      # --privileged exposes /dev/accel* (/dev/vfio); the image has no
      # libtpu.so, so the host's is mounted and TPU_LIBRARY_PATH points at it.
      # The host's TPU_* / LIBTPU_INIT_ARGS (see configure_tpu_runtime_env)
      # are forwarded because docker run does not inherit the environment.
      tpu) docker_args+=(--privileged
             --mount "type=bind,src=${TPU_LIBRARY_PATH_HOST},dst=${DOCKER_LIBTPU},readonly"
             -e "TPU_LIBRARY_PATH=${DOCKER_LIBTPU}")
           local kv
           for kv in "${TPU_FORWARD_ENV[@]}"; do
             docker_args+=(-e "${kv}")
           done ;;
    esac
    run docker run -d "${docker_args[@]}" "${XSE_IMAGE}" \
      --model_path=/tmp/model "${flags[@]}"
    [[ "${DRY_RUN}" == 1 ]] && return 0
    SERVER_IN_DOCKER=1
    docker logs -f "${DOCKER_NAME}" > "${log}" 2>&1 &
    SERVER_PID="$(docker inspect -f '{{.State.Pid}}' "${DOCKER_NAME}")"
  else
    echo "+ ${SERVER_TASKSET[*]} ${SERVER_ENV[*]} ${XSE_BIN} --model_path=${model_path} ${flags[*]}" >&2
    [[ "${DRY_RUN}" == 1 ]] && return 0
    "${SERVER_TASKSET[@]}" "${SERVER_ENV[@]}" "${XSE_BIN}" --model_path="${model_path}" \
      "${flags[@]}" > "${log}" 2>&1 &
    SERVER_PID=$!
  fi
}

start_python() {
  local log="$1"; shift
  local argv
  read -r -a argv <<< "$(tool_argv python_server \
    --weights_dir="${WEIGHTS_DIR}" --port="${PORT}" --device="${PLATFORM}" \
    --model_name="${MODEL_NAME}" --num_threads="${NUM_THREADS}" \
    --alsologtostderr "$@")"
  echo "+ ${SERVER_TASKSET[*]} ${argv[*]}" >&2
  [[ "${DRY_RUN}" == 1 ]] && return 0
  if [[ "${MODE}" == "" ]]; then
    "${SERVER_TASKSET[@]}" "${argv[@]}" > "${log}" 2>&1 &
  else
    (cd "${PY_ROOT}" && exec "${SERVER_TASKSET[@]}" "${argv[@]}") > "${log}" 2>&1 &
  fi
  SERVER_PID=$!
}

# Warns when N private copies of the tables (py-mp arms) will not fit.
check_memory_for_processes() {
  local tables="${WEIGHTS_DIR}/tables.npz" bytes avail need
  [[ -f "${tables}" ]] || return 0
  bytes="$(stat -c %s "${tables}" 2>/dev/null || echo 0)"
  avail="$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo 2>/dev/null || echo 0)"
  # npz is stored uncompressed, so the file size is the in-memory size; plus
  # ~4 GiB for JAX/numpy imports, the rings and the device owner.
  need=$(( NUM_PROCESSES * bytes + 4 * 1024 * 1024 * 1024 ))
  if [[ "${avail}" -gt 0 && "${need}" -gt "${avail}" ]]; then
    echo "warning: --num_processes=${NUM_PROCESSES} frontends each load a private" \
      "copy of the tables ($(( bytes / 1048576 )) MiB): ~$(( need / 1073741824 )) GiB" \
      "needed, $(( avail / 1073741824 )) GiB available. Lower --num_processes" \
      "or --max_vocab_size, or expect swapping/OOM." >&2
  fi
}

start_server() {
  local arm="$1" log="${LOG_DIR}/$1.server.log"
  local batch_flags=(--dynamic_batching
    --batch_timeout_micros="${BATCH_TIMEOUT_MICROS}"
    --num_batch_threads="${NUM_BATCH_THREADS}"
    --max_enqueued_batches="${MAX_ENQUEUED_BATCHES}")
  local mp_flags=(--num_processes="${NUM_PROCESSES}" --ring_mb="${RING_MB}")
  case "${arm}" in
    xse-batch) start_xse "${BATCHED_EXPORT}" "${log}" ;;
    xse-nobatch) start_xse "${UNBATCHED_EXPORT}" "${log}" ;;
    xse-tf-batch) start_xse "${TF_FUSED_BATCHED_EXPORT}" "${log}" ;;
    xse-tf-nobatch) start_xse "${TF_FUSED_UNBATCHED_EXPORT}" "${log}" ;;
    xse-tf-gather-batch) start_xse "${TF_GATHER_BATCHED_EXPORT}" "${log}" ;;
    xse-tf-gather-nobatch) start_xse "${TF_GATHER_UNBATCHED_EXPORT}" "${log}" ;;
    py) start_python "${log}" --nodynamic_batching ;;
    py-batch) start_python "${log}" "${batch_flags[@]}" ;;
    py-mp)
      check_memory_for_processes
      start_python "${log}" --nodynamic_batching "${mp_flags[@]}" ;;
    py-mp-batch)
      check_memory_for_processes
      start_python "${log}" "${batch_flags[@]}" "${mp_flags[@]}" ;;
    *) echo "Unknown arm ${arm}" >&2; exit 1 ;;
  esac
  [[ "${DRY_RUN}" == 1 ]] && return 0
  echo "Waiting for ${arm} (pid ${SERVER_PID}) on port ${PORT}..."
  wait_for_port "${log}"
}

start_monitor() {
  local output="$1"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  # Background the binary itself (not the `tool` shell function) so that
  # MONITOR_PID is the python process and signals reach it directly. It runs
  # on the client CPUs: sampling PSS over a multi-process server tree walks
  # /proc/<pid>/smaps_rollup for every process, and that must not compete
  # with the server whose CPU time is being measured.
  local argv
  read -r -a argv <<< "$(tool_argv monitor --pid="${SERVER_PID}" \
    --output="${output}" --accelerator="${PLATFORM}")"
  if [[ "${MODE}" == "" ]]; then
    "${CLIENT_TASKSET[@]}" "${argv[@]}" >> "${LOG_DIR}/monitor.log" 2>&1 &
  else
    (cd "${PY_ROOT}" && exec "${CLIENT_TASKSET[@]}" "${argv[@]}") \
      >> "${LOG_DIR}/monitor.log" 2>&1 &
  fi
  MONITOR_PID=$!
}

# Prints the best closed-loop QPS of <arm> at request batch size <B>.
peak_qps() {
  local arm="$1" b="$2"
  python3 - "${RESULTS_DIR}/${arm}" "${b}" <<'PY'
import glob, json, os, sys
d, b = sys.argv[1], int(sys.argv[2])
best = 0.0
for p in glob.glob(os.path.join(d, f'closed_B{b}_*.json')):
  if p.endswith('.monitor.json'):
    continue
  r = json.load(open(p))
  if r.get('errors', 0) == 0 or r['requests'] > 100 * r['errors']:
    best = max(best, r['qps'])
print(f'{best:.3f}')
PY
}

run_loadgen() {
  local arm="$1" name="$2"; shift 2
  local out="${RESULTS_DIR}/${arm}/${name}"
  mkdir -p "${RESULTS_DIR}/${arm}"
  start_monitor "${out}.monitor.json"
  local argv
  read -r -a argv <<< "$(tool_argv loadgen \
    --server="${SERVER_ADDR}" --weights_dir="${WEIGHTS_DIR}" \
    --model_name="${MODEL_NAME}" --duration_s="${DURATION}" \
    --warmup_s="${WARMUP}" --num_workers="${NUM_WORKERS}" \
    --channels_per_worker="${CHANNELS_PER_WORKER}" \
    --output="${out}.json" \
    --labels="${BASE_LABELS}" \
    "$@")"
  if [[ "${MODE}" == "" ]]; then
    run "${CLIENT_TASKSET[@]}" "${argv[@]}"
  else
    (cd "${PY_ROOT}" && run "${CLIENT_TASKSET[@]}" "${argv[@]}")
  fi
  stop_monitor
}

run_arm() {
  local arm="$1"
  echo "==================== arm=${arm} ===================="
  BASE_LABELS="arm=${arm},platform=${PLATFORM},embedding_size=${EMBEDDING_SIZE},num_threads=${NUM_THREADS},channels_per_worker=${CHANNELS_PER_WORKER}"
  case "${arm}" in
    py-mp*) BASE_LABELS+=",num_processes=${NUM_PROCESSES}" ;;
    xse-tf-gather-*) BASE_LABELS+=",preprocessor=tf-gather,preprocessor_threads=${PREPROCESSOR_THREADS}" ;;
    xse-tf-*) BASE_LABELS+=",preprocessor=tf-fused,preprocessor_threads=${PREPROCESSOR_THREADS}" ;;
    xse-*) BASE_LABELS+=",preprocessor=jce,preprocessor_threads=${PREPROCESSOR_THREADS}" ;;
  esac
  case "${arm}" in
    *-batch) BASE_LABELS+=",num_batch_threads=${NUM_BATCH_THREADS},max_enqueued_batches=${MAX_ENQUEUED_BATCHES}" ;;
  esac
  start_server "${arm}"

  # Numerical parity against an in-process JAX reference (CPU).
  mkdir -p "${RESULTS_DIR}/${arm}"
  run tool parity_check --server_a="${SERVER_ADDR}" \
    --weights_dir="${WEIGHTS_DIR}" --model_name="${MODEL_NAME}" \
    --batch_sizes="${REQUEST_BATCH_SIZES}" \
    --output="${RESULTS_DIR}/${arm}/parity.parity.json"

  local b c r frac peak qps
  for b in ${REQUEST_BATCH_SIZES//,/ }; do
    for c in ${CONCURRENCIES//,/ }; do
      for r in $(seq 1 "${REPEATS}"); do
        run_loadgen "${arm}" "closed_B${b}_C${c}_r${r}" \
          --request_batch_size="${b}" --concurrency="${c}"
      done
    done
    [[ "${DRY_RUN}" == 1 ]] && continue
    peak="$(peak_qps "${arm}" "${b}")"
    echo "arm=${arm} B=${b}: peak closed-loop ${peak} req/s"
    for frac in ${OPEN_LOOP_FRACTIONS//,/ }; do
      qps="$(python3 -c "print(max(0.5, ${peak} * ${frac}))")"
      for r in $(seq 1 "${REPEATS}"); do
        run_loadgen "${arm}" "open_B${b}_f${frac}_r${r}" \
          --request_batch_size="${b}" --qps="${qps}" \
          --labels="${BASE_LABELS},peak_fraction=${frac}"
      done
    done
  done

  stop_server
  sleep 2
}

for arm in ${ARMS//,/ }; do
  run_arm "${arm}"
done

run tool report --results_dir="${RESULTS_DIR}" \
  --output="${RESULTS_DIR}/REPORT.md"
if [[ "${DRY_RUN}" == 0 ]]; then
  echo
  cat "${RESULTS_DIR}/REPORT.md"
  echo
  echo "Report: ${RESULTS_DIR}/REPORT.md"
fi

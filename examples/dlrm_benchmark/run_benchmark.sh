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
# End-to-end orchestration of the XSE vs. pure-Python JAX serving benchmark.
#
# One invocation exports the model once, then for every arm starts the server
# pinned to --server_cpus, verifies numerical parity, runs the closed-loop and
# open-loop sweeps with the load generator pinned to --client_cpus, stops the
# server, and renders results/REPORT.md.
#
# Python tools run via `python3 -m examples.dlrm_benchmark.*` from an isolated
# import root under <work_dir>/pysrc containing only `examples/`, so that the
# repository's ungenerated `xla_saved_model/` source directory never shadows the
# installed `xla-saved-model` wheel (pass --bin_dir to use prebuilt binaries).
#
# TPU notes:
#   * Prebuilt `tpu-jax-only` images bundle `/usr/lib/libtpu.so` out of the box.
#     Pass `--tpu_library_path` only when running a bare `--xse_bin` TPU server
#     (when auto-detection cannot locate `libtpu.so`) or to override the
#     container's bundled plugin.
#   * `docker run` does not inherit the caller's environment, so every exported
#     `TPU_*` variable (except `TPU_LIBRARY_PATH`) and `LIBTPU_INIT_ARGS` is
#     forwarded into the container. On a Compute Engine VM without `tpu-env`
#     instance metadata (`gcloud compute instances create --machine-type ct6e-*`),
#     single-host topology variables are derived automatically from the machine
#     type when unset.

set -euo pipefail

usage() {
  local code="${1:-1}"
  local out=2
  [[ "${code}" == 0 ]] && out=1
  cat >&"${out}" <<EOF
Usage: $0 --platform <cpu|cuda|tpu> [options]
EOF
  cat >&"${out}" <<'EOF'

Required:
  --platform P            Target accelerator: cpu | cuda | tpu.

XSE server (pass --xse_image or --xse_bin; auto-detects bazel-bin if built):
  --xse_image IMAGE       Prebuilt container image, e.g.
                          us-docker.pkg.dev/xla-serving-engine/xla-serving/tpu-jax-only:nightly
                          us-docker.pkg.dev/xla-serving-engine/xla-serving/gpu-jax-cuda12:nightly
  --xse_bin PATH          xla_serving_main_{cpu,gpu,tpu} binary.
  --tpu_library_path PATH Host libtpu.so path. Required only for a bare
                          --xse_bin TPU server when $TPU_LIBRARY_PATH and the
                          Python libtpu package are absent; if passed with
                          --xse_image, bind-mounts PATH over the image's
                          bundled /usr/lib/libtpu.so.

Fairness / placement:
  --server_cpus LIST      CPU list for the server (taskset / --cpuset-cpus).
  --client_cpus LIST      CPU list for the load generator (taskset).
  --num_threads N         Server request threads for all arms (default 8).
  --num_batch_threads N   Batch execution threads for xse-batch, py-batch, and
                          py-mp-batch (default: --num_threads; py-mp-batch
                          splits them across frontends, at least 1 each).
  --batch_timeout_micros  Dynamic batching timeout in microseconds (default 1000).
  --max_enqueued_batches N
                          Batch queue admission control for *-batch arms
                          (default 64): rejects excess load with UNAVAILABLE.
  --batch_sizes LIST      Exported batch sizes = PAD_UP buckets for every arm
                          (default: powers of two 1..4096).
  --num_processes N       Frontend processes for py-mp and py-mp-batch
                          (default: --num_threads). Each loads a private copy
                          of the embedding tables (N x ~3.2 GB at full scale).
  --channels_per_worker N gRPC connections per load-generator worker across all
                          arms (default 8).
  --ring_mb N             Shared-memory ring per py-mp frontend in MiB (256).

Model:
  --embedding_size N      Embedding dimension (default 128).
  --max_vocab_size N      Cap per-table vocab; 0 = full ~6.2M rows (default 0).
  --skip_export           Reuse <work_dir>/model from a previous run.

Sweep & report:
  --arms LIST             Subset of xse-batch,xse-nobatch,py,py-batch,py-mp,
                          py-mp-batch (default: all six).
  --request_batch_sizes   Examples per request (default 1,16,128).
  --concurrencies LIST    Closed-loop concurrent clients (default 1,4,16,64,256).
  --open_loop_fractions   Open-loop QPS as fraction of peak (default 0.5,0.8;
                          "" disables open-loop runs).
  --duration S            Measured seconds per point (default 30).
  --warmup S              Discarded warm-up seconds per point (default 5).
  --repeats N             Repeats per point, median reported (default 3).
  --num_workers N         Load generator processes (default 8).
  --p99_budget_ms MS      Tail-latency budget for headline table (default 20;
                          <= 0 disables).

Misc:
  --work_dir DIR          Working directory (default /tmp/dlrm_benchmark).
  --port N                Server port (default 8500).
  --bin_dir DIR           Optional directory of prebuilt benchmark binaries.
  --dry_run               Print commands without executing them.
EOF
  exit "${code}"
}

# ---------------------------------------------------------------------------
# Defaults.
# ---------------------------------------------------------------------------
PLATFORM=""
XSE_BIN=""
XSE_IMAGE=""
TPU_LIBRARY_PATH_FLAG=""
TPU_LIBRARY_PATH_HOST=""
SERVER_CPUS=""
CLIENT_CPUS=""
NUM_THREADS=8
NUM_BATCH_THREADS=""
BATCH_TIMEOUT_MICROS=1000
MAX_ENQUEUED_BATCHES=64
BATCH_SIZES=""
NUM_PROCESSES=""
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
P99_BUDGET_MS=20
WORK_DIR="/tmp/dlrm_benchmark"
PORT=8500
BIN_DIR=""
DRY_RUN=0
MODEL_NAME="dlrm"
DOCKER_LIBTPU="/usr/lib/libtpu.so"

need_arg() {
  if [[ $# -lt 2 ]]; then
    echo "Option $1 requires a value" >&2
    usage 1
  fi
}

while [[ $# -gt 0 ]]; do
  if [[ "$1" == --*=* ]]; then
    set -- "${1%%=*}" "${1#*=}" "${@:2}"
  fi
  case "$1" in
    --platform) need_arg "$@"; PLATFORM="$2"; shift 2 ;;
    --xse_bin) need_arg "$@"; XSE_BIN="$2"; shift 2 ;;
    --xse_image) need_arg "$@"; XSE_IMAGE="$2"; shift 2 ;;
    --tpu_library_path) need_arg "$@"; TPU_LIBRARY_PATH_FLAG="$2"; shift 2 ;;
    --server_cpus) need_arg "$@"; SERVER_CPUS="$2"; shift 2 ;;
    --client_cpus) need_arg "$@"; CLIENT_CPUS="$2"; shift 2 ;;
    --num_threads) need_arg "$@"; NUM_THREADS="$2"; shift 2 ;;
    --num_batch_threads) need_arg "$@"; NUM_BATCH_THREADS="$2"; shift 2 ;;
    --batch_timeout_micros) need_arg "$@"; BATCH_TIMEOUT_MICROS="$2"; shift 2 ;;
    --max_enqueued_batches) need_arg "$@"; MAX_ENQUEUED_BATCHES="$2"; shift 2 ;;
    --batch_sizes) need_arg "$@"; BATCH_SIZES="$2"; shift 2 ;;
    --num_processes) need_arg "$@"; NUM_PROCESSES="$2"; shift 2 ;;
    --channels_per_worker) need_arg "$@"; CHANNELS_PER_WORKER="$2"; shift 2 ;;
    --ring_mb) need_arg "$@"; RING_MB="$2"; shift 2 ;;
    --embedding_size) need_arg "$@"; EMBEDDING_SIZE="$2"; shift 2 ;;
    --max_vocab_size) need_arg "$@"; MAX_VOCAB_SIZE="$2"; shift 2 ;;
    --skip_export) SKIP_EXPORT=1; shift ;;
    --arms) need_arg "$@"; ARMS="$2"; shift 2 ;;
    --request_batch_sizes) need_arg "$@"; REQUEST_BATCH_SIZES="$2"; shift 2 ;;
    --concurrencies) need_arg "$@"; CONCURRENCIES="$2"; shift 2 ;;
    --open_loop_fractions) need_arg "$@"; OPEN_LOOP_FRACTIONS="$2"; shift 2 ;;
    --duration) need_arg "$@"; DURATION="$2"; shift 2 ;;
    --warmup) need_arg "$@"; WARMUP="$2"; shift 2 ;;
    --repeats) need_arg "$@"; REPEATS="$2"; shift 2 ;;
    --num_workers) need_arg "$@"; NUM_WORKERS="$2"; shift 2 ;;
    --p99_budget_ms) need_arg "$@"; P99_BUDGET_MS="$2"; shift 2 ;;
    --work_dir) need_arg "$@"; WORK_DIR="$2"; shift 2 ;;
    --port) need_arg "$@"; PORT="$2"; shift 2 ;;
    --bin_dir) need_arg "$@"; BIN_DIR="$2"; shift 2 ;;
    --dry_run) DRY_RUN=1; shift ;;
    -h|--help) usage 0 ;;
    *) echo "Unknown argument: $1" >&2; usage 1 ;;
  esac
done
: "${NUM_BATCH_THREADS:=${NUM_THREADS}}"
: "${NUM_PROCESSES:=${NUM_THREADS}}"

require_pos_int() {
  if [[ ! "$2" =~ ^[1-9][0-9]*$ ]]; then
    echo "$1 must be a positive integer, got '$2'" >&2
    usage 1
  fi
}
require_pos_int --num_threads "${NUM_THREADS}"
require_pos_int --num_batch_threads "${NUM_BATCH_THREADS}"
require_pos_int --num_processes "${NUM_PROCESSES}"
require_pos_int --repeats "${REPEATS}"
require_pos_int --num_workers "${NUM_WORKERS}"
require_pos_int --channels_per_worker "${CHANNELS_PER_WORKER}"
require_pos_int --embedding_size "${EMBEDDING_SIZE}"
require_pos_int --port "${PORT}"

[[ -n "${ARMS//,/}" ]] || { echo "--arms must not be empty" >&2; usage 1; }
[[ -n "${REQUEST_BATCH_SIZES//,/}" ]] || { echo "--request_batch_sizes must not be empty" >&2; usage 1; }
[[ -n "${CONCURRENCIES//,/}" ]] || { echo "--concurrencies must not be empty" >&2; usage 1; }

for arm in ${ARMS//,/ }; do
  case "${arm}" in
    xse-batch|xse-nobatch|py|py-batch|py-mp|py-mp-batch) ;;
    *) echo "Unknown arm ${arm}" >&2; usage ;;
  esac
done

case "${PLATFORM}" in
  cpu|cuda|tpu) ;;
  *) echo "--platform must be cpu, cuda, or tpu" >&2; usage ;;
esac

# ---------------------------------------------------------------------------
# Locate Python benchmark tools and XSE server binary / image.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

if [[ -z "${BIN_DIR}" && -x "${REPO_ROOT}/bazel-bin/examples/dlrm_benchmark/export_benchmark_model" ]]; then
  BIN_DIR="${REPO_ROOT}/bazel-bin/examples/dlrm_benchmark"
fi

TOOL_MODE="python"
if [[ -n "${BIN_DIR}" ]]; then
  BIN_DIR="$(cd "${BIN_DIR}" && pwd)"
  TOOL_MODE="bin"
  if [[ "${DRY_RUN}" == 0 && ! -x "${BIN_DIR}/export_benchmark_model" ]]; then
    echo "Tool binaries not found under --bin_dir=${BIN_DIR}" >&2
    exit 1
  fi
fi

# Runs one of the benchmark tools with the given flags.
# In python mode, runs `python3 -m examples.dlrm_benchmark.<name>` from PY_ROOT
# (containing only a symlink to `examples/`) so `sys.path[0]` does not shadow
# the installed `xla_saved_model` wheel with the repo's source directory.
tool() {
  local name="$1"; shift
  if [[ "${TOOL_MODE}" == "bin" ]]; then
    "${BIN_DIR}/${name}" "$@"
  else
    (cd "${PY_ROOT}" && python3 -m "examples.dlrm_benchmark.${name}" "$@")
  fi
}

TOOL_ARGV=()
set_tool_argv() {
  local name="$1"; shift
  if [[ "${TOOL_MODE}" == "bin" ]]; then
    TOOL_ARGV=("${BIN_DIR}/${name}" "$@")
  else
    TOOL_ARGV=(python3 -m "examples.dlrm_benchmark.${name}" "$@")
  fi
}

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
    set_tool_argv "${@:2}"
    if [[ "${TOOL_MODE}" == "bin" ]]; then
      echo "+ $(quoted "${TOOL_ARGV[@]}")" >&2
    else
      echo "+ (cd $(quoted "${PY_ROOT}") && $(quoted "${TOOL_ARGV[@]}"))" >&2
    fi
  else
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

needs_xse=0
for arm in ${ARMS//,/ }; do
  [[ "${arm}" == xse-* ]] && needs_xse=1
done

if [[ "${needs_xse}" == 1 && -z "${XSE_BIN}" && -z "${XSE_IMAGE}" ]]; then
  if [[ -n "${BIN_DIR}" && -x "${BIN_DIR%/examples/dlrm_benchmark}/runtime/serving/gcp/server/${XSE_TARGET}" ]]; then
    XSE_BIN="${BIN_DIR%/examples/dlrm_benchmark}/runtime/serving/gcp/server/${XSE_TARGET}"
  elif [[ -x "${REPO_ROOT}/bazel-bin/runtime/serving/gcp/server/${XSE_TARGET}" ]]; then
    XSE_BIN="${REPO_ROOT}/bazel-bin/runtime/serving/gcp/server/${XSE_TARGET}"
  fi
fi

if [[ "${needs_xse}" == 1 && -z "${XSE_IMAGE}" && ! -x "${XSE_BIN}" ]]; then
  if [[ "${DRY_RUN}" == 0 ]]; then
    echo "XSE server not specified and no local ${XSE_TARGET} binary found." >&2
    echo "Pass --xse_image us-docker.pkg.dev/xla-serving-engine/xla-serving/<variant>:nightly" >&2
    echo "or --xse_bin /path/to/${XSE_TARGET}. See README.md." >&2
    exit 1
  fi
  XSE_BIN="${XSE_BIN:-<XSE_SERVER_BIN>}"
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
  if ! docker image inspect "${XSE_IMAGE}" >/dev/null 2>&1; then
    run docker pull "${XSE_IMAGE}"
  fi
fi

# ---------------------------------------------------------------------------
# libtpu resolution (bare --xse_bin TPU server, or explicit --tpu_library_path).
# ---------------------------------------------------------------------------
resolve_tpu_library_path() {
  local candidate=""
  TPU_LIBRARY_PATH_HOST="${TPU_LIBRARY_PATH_FLAG}"
  if [[ -z "${TPU_LIBRARY_PATH_HOST}" && -n "${TPU_LIBRARY_PATH:-}" ]]; then
    TPU_LIBRARY_PATH_HOST="${TPU_LIBRARY_PATH}"
  fi
  if [[ -z "${TPU_LIBRARY_PATH_HOST}" ]]; then
    candidate="$(python3 - 2>/dev/null <<'PY'
import os
import libtpu
try:
  print(libtpu.get_library_path())
except AttributeError:
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
  if [[ -n "${TPU_LIBRARY_PATH_HOST}" && -f "${TPU_LIBRARY_PATH_HOST}" ]]; then
    TPU_LIBRARY_PATH_HOST="$(readlink -f "${TPU_LIBRARY_PATH_HOST}")"
  fi
}

# ---------------------------------------------------------------------------
# TPU runtime environment (topology) for the server.
# ---------------------------------------------------------------------------
TPU_FORWARD_ENV=()
TPU_ENV_METADATA="unknown"  # present | missing | unknown
TPU_ENV_SOURCE="none"       # host | metadata | derived(<machine type>) | none
METADATA_URL="http://metadata.google.internal/computeMetadata/v1/instance"

tpu_topology_for_machine_type() {
  local machine_type="$1" accel chips bounds
  case "${machine_type}" in
    ct6e-standard-[148]t) accel="v6e-" ;;
    ct5lp-hightpu-[148]t) accel="v5litepod-" ;;
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

metadata_status() {
  if [[ -n "${DLRM_BENCHMARK_FAKE_METADATA:-}" ]]; then
    echo "${DLRM_BENCHMARK_FAKE_METADATA%%:*}"
    return 0
  fi
  command -v curl >/dev/null || { echo 000; return 0; }
  local code
  code="$(curl -s -m 2 -o /dev/null -w '%{http_code}' -H 'Metadata-Flavor: Google' \
    "${METADATA_URL}/$1" 2>/dev/null)" || true
  echo "${code:-000}"
}

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
          export TPU_SKIP_MDS_QUERY=1
          : "${TPU_HOST_BOUNDS:=1,1,1}"; export TPU_HOST_BOUNDS
          : "${TPU_WORKER_HOSTNAMES:=localhost}"; export TPU_WORKER_HOSTNAMES
          : "${TPU_WORKER_ID:=0}"; export TPU_WORKER_ID
          : "${TPU_ACCELERATOR_TYPE:=${accel}}"; export TPU_ACCELERATOR_TYPE
          : "${TPU_CHIPS_PER_HOST_BOUNDS:=${bounds}}"; export TPU_CHIPS_PER_HOST_BOUNDS
          TPU_ENV_SOURCE="derived(${machine_type})"
          echo "No tpu-env instance metadata found; derived single-host TPU topology for" \
            "${machine_type}: TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=${TPU_ACCELERATOR_TYPE}" \
            "TPU_CHIPS_PER_HOST_BOUNDS=${TPU_CHIPS_PER_HOST_BOUNDS}" \
            "TPU_HOST_BOUNDS=${TPU_HOST_BOUNDS} TPU_WORKER_ID=${TPU_WORKER_ID}" \
            "TPU_WORKER_HOSTNAMES=${TPU_WORKER_HOSTNAMES}" >&2
        else
          cat >&2 <<EOF
This VM has no tpu-env instance metadata (not created via the Cloud TPU API)
and its machine type "${machine_type:-unknown}" is not a known single-host TPU
type (ct6e-standard-{1,4,8}t, ct5lp-hightpu-{1,4,8}t, ct5p-hightpu-4t).

Export the topology variables for this machine before running, e.g. for v6e-1:
  export TPU_SKIP_MDS_QUERY=1 TPU_ACCELERATOR_TYPE=v6e-1 TPU_CHIPS_PER_HOST_BOUNDS=1,1,1
  export TPU_HOST_BOUNDS=1,1,1 TPU_WORKER_HOSTNAMES=localhost TPU_WORKER_ID=0
or create the VM with the Cloud TPU API (gcloud compute tpus tpu-vm create).
EOF
          if [[ "${DRY_RUN}" == 0 ]]; then
            exit 1
          fi
          echo "warning: continuing without TPU topology variables (--dry_run)." >&2
        fi
        ;;
      *) ;;
    esac
  fi
  while IFS= read -r name; do
    [[ "${name}" == "TPU_LIBRARY_PATH" ]] && continue
    TPU_FORWARD_ENV+=("${name}=${!name:-}")
  done < <(compgen -e | grep -E '^(TPU_|LIBTPU_INIT_ARGS$)' | sort || true)
}

if [[ "${PLATFORM}" == "tpu" ]]; then
  configure_tpu_runtime_env
  if [[ -n "${TPU_LIBRARY_PATH_FLAG}" ]]; then
    TPU_LIBRARY_PATH_HOST="${TPU_LIBRARY_PATH_FLAG}"
    if [[ -f "${TPU_LIBRARY_PATH_HOST}" ]]; then
      TPU_LIBRARY_PATH_HOST="$(readlink -f "${TPU_LIBRARY_PATH_HOST}")"
    elif [[ "${DRY_RUN}" == 0 ]]; then
      echo "libtpu.so not found at --tpu_library_path=${TPU_LIBRARY_PATH_HOST}." >&2
      exit 1
    fi
  elif [[ "${needs_xse}" == 1 && -z "${XSE_IMAGE}" ]]; then
    # Bare xla_serving_main_tpu binary requires TPU_LIBRARY_PATH on the host.
    resolve_tpu_library_path
    if [[ -z "${TPU_LIBRARY_PATH_HOST}" ]]; then
      if [[ "${DRY_RUN}" == 0 ]]; then
        cat >&2 <<EOF
No libtpu.so found for bare TPU binary ${XSE_BIN}.

xla_serving_main_tpu loads the TPU PJRT plugin from \$TPU_LIBRARY_PATH at
startup. Install jax[tpu] in the active environment:
  pip install "jax[tpu]" -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
or pass --tpu_library_path /path/to/libtpu.so (or use --xse_image, which
bundles libtpu.so at /usr/lib/libtpu.so).
EOF
        exit 1
      fi
      echo "warning: no libtpu.so found; using <TPU_LIBRARY_PATH> placeholder (--dry_run)." >&2
      TPU_LIBRARY_PATH_HOST="<TPU_LIBRARY_PATH>"
    elif [[ "${DRY_RUN}" == 0 && ! -f "${TPU_LIBRARY_PATH_HOST}" ]]; then
      echo "libtpu.so not found at ${TPU_LIBRARY_PATH_HOST} (from \$TPU_LIBRARY_PATH)." >&2
      exit 1
    fi
  fi
fi

SERVER_ENV=()
if [[ "${PLATFORM}" == "tpu" && -z "${XSE_IMAGE}" && -n "${TPU_LIBRARY_PATH_HOST}" ]]; then
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
WORK_DIR="$(cd "${WORK_DIR}" && pwd)"
MODEL_DIR="${WORK_DIR}/model"
RESULTS_DIR="${WORK_DIR}/results"
LOG_DIR="${WORK_DIR}/logs"
BATCHED_EXPORT="${MODEL_DIR}/xse_batched"
UNBATCHED_EXPORT="${MODEL_DIR}/xse_unbatched"
WEIGHTS_DIR="${MODEL_DIR}/baseline_weights"
mkdir -p "${MODEL_DIR}" "${RESULTS_DIR}" "${LOG_DIR}"
SERVER_ADDR="localhost:${PORT}"
DOCKER_NAME="dlrm-benchmark-xse-$$"

PY_ROOT=""
if [[ "${TOOL_MODE}" == "python" ]]; then
  PY_ROOT="${WORK_DIR}/pysrc"
  mkdir -p "${PY_ROOT}"
  ln -sfn "${REPO_ROOT}/examples" "${PY_ROOT}/examples"
  if [[ "${DRY_RUN}" == 0 ]] && ! (cd "${PY_ROOT}" && \
      python3 -c 'import xla_saved_model.protos.grpc.predict_pb2' 2>/dev/null); then
    cat >&2 <<EOF
The installed xla-saved-model package is missing generated protobuf modules
(import xla_saved_model.protos.grpc.predict_pb2 failed from ${PY_ROOT}).

Install the release packages into the active virtual environment:
  pip install --force-reinstall xla-saved-model xla-serving-data

If building from source instead, ensure bazel is on PATH before running
\`python -m build --wheel\`.
EOF
    exit 1
  fi
fi

echo "tool_mode=${TOOL_MODE} platform=${PLATFORM} work_dir=${WORK_DIR}"
echo "arms=${ARMS} B=${REQUEST_BATCH_SIZES} C=${CONCURRENCIES}" \
  "repeats=${REPEATS} duration=${DURATION}s"

# ---------------------------------------------------------------------------
# Environment snapshot.
# ---------------------------------------------------------------------------
{
  echo "date: $(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "host: $(hostname 2>/dev/null || uname -n)"
  echo "platform: ${PLATFORM}"
  echo "tool_mode: ${TOOL_MODE}"
  echo "nproc: $(nproc)"
  echo "server_cpus: ${SERVER_CPUS:-all}"
  echo "client_cpus: ${CLIENT_CPUS:-all}"
  echo "num_threads: ${NUM_THREADS}"
  echo "num_batch_threads: ${NUM_BATCH_THREADS}"
  echo "jce_max_cpu_slots: ${NUM_THREADS}"
  echo "batch_timeout_micros: ${BATCH_TIMEOUT_MICROS}"
  echo "max_enqueued_batches: ${MAX_ENQUEUED_BATCHES}"
  echo "batch_sizes: ${BATCH_SIZES:-default (powers of two 1..4096)}"
  echo "num_processes: ${NUM_PROCESSES}"
  echo "channels_per_worker: ${CHANNELS_PER_WORKER}"
  echo "num_workers: ${NUM_WORKERS}"
  echo "ring_mb: ${RING_MB}"
  echo "p99_budget_ms: ${P99_BUDGET_MS}"
  echo "xse: ${XSE_IMAGE:-${XSE_BIN:-none}}"
  if [[ -n "${XSE_IMAGE}" ]] && docker image inspect "${XSE_IMAGE}" >/dev/null 2>&1; then
    docker image inspect --format \
      'xse_image_id: {{.Id}}{{"\n"}}xse_image_digests: {{join .RepoDigests ","}}' \
      "${XSE_IMAGE}" 2>/dev/null || true
  fi
  grep -m1 'model name' /proc/cpuinfo || true
  awk '/MemTotal|MemAvailable/ {printf "%s %.1f GiB\n", tolower($1), $2/1048576}' /proc/meminfo 2>/dev/null || true
  command -v nvidia-smi >/dev/null && nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv || true
  python3 -c 'import jax; print("jax:", jax.__version__); print("devices:", jax.devices())' 2>/dev/null || true
  if [[ "${PLATFORM}" == "tpu" ]]; then
    if [[ -n "${XSE_IMAGE}" && -z "${TPU_LIBRARY_PATH_FLAG}" ]]; then
      echo "tpu_library_path: bundled (${DOCKER_LIBTPU} in image)"
    else
      echo "tpu_library_path: ${TPU_LIBRARY_PATH_HOST:-unset}"
    fi
    python3 -c 'import importlib.metadata as m; print("libtpu:", m.version("libtpu"))' 2>/dev/null || true
    echo "tpu_env_metadata: ${TPU_ENV_METADATA}"
    echo "tpu_env_source: ${TPU_ENV_SOURCE}"
    echo "tpu_env: ${TPU_FORWARD_ENV[*]:-none}"
  fi
} > "${RESULTS_DIR}/env.txt" || true

# ---------------------------------------------------------------------------
# Export (once).
# ---------------------------------------------------------------------------
if [[ "${SKIP_EXPORT}" == 0 ]]; then
  rm -rf "${BATCHED_EXPORT}" "${UNBATCHED_EXPORT}" "${WEIGHTS_DIR}"
  export_flags=(
    --batched_export_path="${BATCHED_EXPORT}"
    --unbatched_export_path="${UNBATCHED_EXPORT}"
    --baseline_weights_dir="${WEIGHTS_DIR}"
    --embedding_size="${EMBEDDING_SIZE}"
    --max_vocab_size="${MAX_VOCAB_SIZE}"
    --platforms="${PLATFORM}"
    --batch_timeout_micros="${BATCH_TIMEOUT_MICROS}"
    --num_batch_threads="${NUM_BATCH_THREADS}"
    --max_enqueued_batches="${MAX_ENQUEUED_BATCHES}"
    --jce_max_cpu_slots="${NUM_THREADS}"
  )
  [[ -n "${BATCH_SIZES}" ]] && export_flags+=(--batch_sizes="${BATCH_SIZES}")
  run tool export_benchmark_model "${export_flags[@]}"
fi

# ---------------------------------------------------------------------------
# Server lifecycle.
# ---------------------------------------------------------------------------
SERVER_PID=""
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
    if docker ps -aq -f "name=^${DOCKER_NAME}$" | grep -q .; then
      docker rm -f "${DOCKER_NAME}" >/dev/null 2>&1 || true
    fi
    SERVER_IN_DOCKER=0
  fi
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

server_alive() {
  if [[ "${SERVER_IN_DOCKER}" == 1 ]]; then
    [[ "$(docker inspect -f '{{.State.Running}}' "${DOCKER_NAME}" 2>/dev/null)" == "true" ]]
  else
    [[ -n "${SERVER_PID}" && "${SERVER_PID}" != 0 ]] && kill -0 "${SERVER_PID}" 2>/dev/null
  fi
}

wait_for_port() {
  local arm="$1" log="$2"
  local deadline=$(( $(date +%s) + 1800 ))
  while true; do
    if [[ "${arm}" != py* ]] || grep -q '^READY port=' "${log}" 2>/dev/null; then
      if (exec 3<>"/dev/tcp/127.0.0.1/${PORT}") 2>/dev/null; then
        return 0
      fi
    fi
    if ! server_alive; then
      echo "Server exited before opening port ${PORT}; see ${log}" >&2
      sleep 1
      echo "--- last lines of ${log} ---" >&2
      tail -n 20 "${log}" >&2 2>/dev/null || true
      return 1
    fi
    if grep -q 'Failed to fetch URL after 30 tries' "${log}" 2>/dev/null; then
      echo "TPU server cannot read topology from the GCE metadata server" \
        "(no tpu-env attribute)." >&2
      echo "Export TPU_SKIP_MDS_QUERY=1 and the single-host TPU_* topology" \
        "variables (see README.md)." >&2
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
  if [[ -n "${XSE_IMAGE}" ]]; then
    local docker_args=(--name "${DOCKER_NAME}" --net=host --ulimit memlock=-1
      --mount "type=bind,src=${model_path},dst=/tmp/model,readonly")
    [[ -n "${SERVER_CPUS}" ]] && docker_args+=(--cpuset-cpus "${SERVER_CPUS}")
    case "${PLATFORM}" in
      cuda) docker_args+=(--gpus all -e PJRT_DEVICE=cuda
              -e NVIDIA_VISIBLE_DEVICES=all
              -e NVIDIA_DRIVER_CAPABILITIES=all,compute,utility
              -e LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/lib/x86_64-linux-gnu:/usr/local/cuda/lib64) ;;
      tpu) docker_args+=(--privileged)
           if [[ -n "${TPU_LIBRARY_PATH_FLAG}" ]]; then
             docker_args+=(
               --mount "type=bind,src=${TPU_LIBRARY_PATH_HOST},dst=${DOCKER_LIBTPU},readonly"
               -e "TPU_LIBRARY_PATH=${DOCKER_LIBTPU}"
             )
           fi
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
    local prefix=("${SERVER_TASKSET[@]}" "${SERVER_ENV[@]}")
    echo "+ $(quoted "${prefix[@]}" "${XSE_BIN}" --model_path="${model_path}" "${flags[@]}")" >&2
    [[ "${DRY_RUN}" == 1 ]] && return 0
    "${prefix[@]}" "${XSE_BIN}" --model_path="${model_path}" \
      "${flags[@]}" > "${log}" 2>&1 &
    SERVER_PID=$!
  fi
}

start_python() {
  local log="$1"; shift
  set_tool_argv python_server \
    --weights_dir="${WEIGHTS_DIR}" --port="${PORT}" --device="${PLATFORM}" \
    --model_name="${MODEL_NAME}" --num_threads="${NUM_THREADS}" \
    --alsologtostderr "$@"
  local argv=("${TOOL_ARGV[@]}")
  echo "+ $(quoted "${SERVER_TASKSET[@]}" "${argv[@]}")" >&2
  [[ "${DRY_RUN}" == 1 ]] && return 0
  if [[ "${TOOL_MODE}" == "bin" ]]; then
    "${SERVER_TASKSET[@]}" "${argv[@]}" > "${log}" 2>&1 &
  else
    (cd "${PY_ROOT}" && exec "${SERVER_TASKSET[@]}" "${argv[@]}") > "${log}" 2>&1 &
  fi
  SERVER_PID=$!
}

check_memory_for_processes() {
  local tables="${WEIGHTS_DIR}/tables.npz" bytes avail need
  [[ -f "${tables}" ]] || return 0
  bytes="$(stat -c %s "${tables}" 2>/dev/null || echo 0)"
  avail="$(awk '/MemAvailable/ {print $2 * 1024}' /proc/meminfo 2>/dev/null || echo 0)"
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
  wait_for_port "${arm}" "${log}"
}

start_monitor() {
  local output="$1"
  [[ "${DRY_RUN}" == 1 ]] && return 0
  set_tool_argv monitor --pid="${SERVER_PID}" \
    --output="${output}" --accelerator="${PLATFORM}"
  local argv=("${TOOL_ARGV[@]}")
  if [[ "${TOOL_MODE}" == "bin" ]]; then
    "${CLIENT_TASKSET[@]}" "${argv[@]}" >> "${LOG_DIR}/monitor.log" 2>&1 &
  else
    (cd "${PY_ROOT}" && exec "${CLIENT_TASKSET[@]}" "${argv[@]}") \
      >> "${LOG_DIR}/monitor.log" 2>&1 &
  fi
  MONITOR_PID=$!
}

peak_qps() {
  local arm="$1" b="$2"
  python3 - "${RESULTS_DIR}/${arm}" "${b}" <<'PY'
import collections, glob, json, os, statistics, sys
d, b = sys.argv[1], int(sys.argv[2])
by_c = collections.defaultdict(list)
for p in glob.glob(os.path.join(d, f'closed_B{b}_*.json')):
  if p.endswith('.monitor.json'):
    continue
  with open(p) as f:
    r = json.load(f)
  if r.get('errors', 0) == 0 or r['requests'] > 100 * r['errors']:
    by_c[r.get('concurrency', 0)].append(r['qps'])
best = max((statistics.median(v) for v in by_c.values()), default=0.0)
print(f'{best:.3f}')
PY
}

run_loadgen() {
  local arm="$1" name="$2"; shift 2
  local out="${RESULTS_DIR}/${arm}/${name}"
  mkdir -p "${RESULTS_DIR}/${arm}"
  start_monitor "${out}.monitor.json"
  set_tool_argv loadgen \
    --server="${SERVER_ADDR}" --weights_dir="${WEIGHTS_DIR}" \
    --model_name="${MODEL_NAME}" --duration_s="${DURATION}" \
    --warmup_s="${WARMUP}" --num_workers="${NUM_WORKERS}" \
    --channels_per_worker="${CHANNELS_PER_WORKER}" \
    --output="${out}.json" \
    --labels="${BASE_LABELS}" \
    "$@"
  local argv=("${TOOL_ARGV[@]}")
  if [[ "${TOOL_MODE}" == "bin" ]]; then
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
  esac
  case "${arm}" in
    *-batch) BASE_LABELS+=",num_batch_threads=${NUM_BATCH_THREADS},max_enqueued_batches=${MAX_ENQUEUED_BATCHES}" ;;
  esac
  start_server "${arm}"

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
  --p99_budget_ms="${P99_BUDGET_MS}" \
  --output="${RESULTS_DIR}/REPORT.md"
if [[ "${DRY_RUN}" == 0 ]]; then
  echo
  cat "${RESULTS_DIR}/REPORT.md"
  echo
  echo "Report: ${RESULTS_DIR}/REPORT.md"
fi

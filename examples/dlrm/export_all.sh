#!/bin/bash
# Script to export all DLRM and JCE models for C++ benchmarking.
# Usage: ./export_all.sh [export_dir]
# Default export_dir is /tmp/dlrm_benchmarks

set -e

EXPORT_DIR="${1:-${HOME}/dlrm_benchmarks}"
rm -rf "${EXPORT_DIR}"
mkdir -p "${EXPORT_DIR}/tmp"
export TMPDIR="${EXPORT_DIR}/tmp"


echo "Exporting models to ${EXPORT_DIR}..."

# Build the exporters first
SKYBUILD=1 blaze build -c opt \
  //third_party/neptune/examples/dlrm:run_export \
  //third_party/neptune/examples/dlrm:run_export_jce

for dim in 16 64 128; do
  echo "----------------------------------------"
  echo "Exporting Baseline (TPUEmbeddingForServing) dim ${dim}..."
  echo "----------------------------------------"
  SKYBUILD=1 blaze run -c opt //third_party/neptune/examples/dlrm:run_export -- \
    --export_path="${EXPORT_DIR}/baseline/dim_${dim}" \
    --embedding_size="${dim}"

  echo "----------------------------------------"
  echo "Exporting JCE dim ${dim}..."
  echo "----------------------------------------"
  SKYBUILD=1 blaze run -c opt //third_party/neptune/examples/dlrm:run_export_jce -- \
    --export_path="${EXPORT_DIR}/jce/dim_${dim}" \
    --embedding_size="${dim}"
done

rm -rf "${EXPORT_DIR}/tmp"

echo "========================================"
echo "All models exported successfully to ${EXPORT_DIR}"
echo "========================================"


#!/bin/bash
export LILITH_EXTRA_ARGS="--path $TEST_SRCDIR/google3/third_party/neptune/runtime/core/stablehlo_transforms/mlir"

exec "$TEST_SRCDIR/google3/third_party/llvm/llvm-project/mlir/run_lit.sh" \
  "$@"
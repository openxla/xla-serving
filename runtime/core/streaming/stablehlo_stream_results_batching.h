// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STABLEHLO_STREAM_RESULTS_BATCHING_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STABLEHLO_STREAM_RESULTS_BATCHING_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project

namespace neptune {

// Automatically threads a dynamic step_id (or request_ids) dynamic-sized tensor
// through the MLIR entry function and passes it down to all `neptune_stream`
// custom calls.
// Returns an error if the propagation fails.
absl::Status PropagateStreamingStepId(mlir::ModuleOp module);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_STABLEHLO_STREAM_RESULTS_BATCHING_H_

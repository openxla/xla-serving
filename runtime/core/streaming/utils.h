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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_UTILS_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune {

inline constexpr absl::string_view kStreamedResultBatchedStepIdName =
    "__streamed_result_batched_step_id";

// Injects a dynamic step_id (or request_ids) dynamic-sized tensor
// through the MLIR entry function and passes it down to all `neptune_stream`
// custom calls.
// Returns an error if the propagation fails.
// This function assumes that the shape refinements have already been performed.
absl::Status PropagateStreamingStepId(mlir::ModuleOp module);

// Creates a HostBuffer containing request_ids for streaming.
absl::StatusOr<neptune::HostBuffer> CreateRequestIdsBuffer(int64_t batch_size,
                                                       int32_t request_id);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_STREAMING_UTILS_H_

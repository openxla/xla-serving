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

#ifndef XLA_SERVING_RUNTIME_CORE_STREAMING_UTILS_H_
#define XLA_SERVING_RUNTIME_CORE_STREAMING_UTILS_H_

#include <cstdint>
#include <optional>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving {

inline constexpr absl::string_view kStreamedResultBatchedStepIdName =
    "__streamed_result_batched_step_id";

// Injects a dynamic step_id (or request_ids) dynamic-sized tensor
// through the MLIR entry function and passes it down to all
// `xla_serving_stream` custom calls.
// Returns an error if the propagation fails.
// This function assumes that the shape refinements have already been performed.
// When `is_batched` is true and `batch_size` is provided, uses `*batch_size`
// directly instead of inferring the batch dimension from `@main`'s arguments
// (which may already be R1-flattened at export time).
absl::Status PropagateStreamingStepId(
    mlir::ModuleOp module, bool is_batched = false,
    std::optional<int64_t> batch_size = std::nullopt);

// Creates a HostBuffer containing request_ids for streaming.
absl::StatusOr<xla_serving::HostBuffer> CreateRequestIdsBuffer(int64_t batch_size,
                                                       int32_t request_id);

absl::StatusOr<xla_serving::HostBuffer> CreateScalarRequestIdBuffer(int32_t request_id);

absl::StatusOr<xla_serving::HostBuffer> CreateRequestIdsBuffer(
    absl::Span<const int32_t> request_ids);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_STREAMING_UTILS_H_

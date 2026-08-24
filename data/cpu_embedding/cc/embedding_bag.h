// Copyright 2026 Arm Ltd. and affiliates
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

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_H_

#include <cstddef>
#include <cstdint>

#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp

namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla_serving::data::cpu_embedding {
class EmbeddingTable;
}  // namespace xla_serving::data::cpu_embedding

namespace xla_serving::data::cpu_embedding::internal {

// Auto-dispatch selects SVE only for the measured 128-bit vector length.
inline constexpr int kAutoSveVectorLengthBytes = 16;

// Backend selection for standard FP32 embedding-bag reductions.
enum class EmbeddingBagBackend {
  kAuto,
  kPortable,
  kSve,
};

// Borrowed inputs, the allocator, and allocated output storage must remain
// valid until the synchronous call returns.
struct EmbeddingBagRequest {
  absl::Span<const int64_t> embedding_ids;
  absl::Span<const int32_t> row_pointers;
  const EmbeddingTable* embedding_table = nullptr;
  absl::Span<const float> weights;
  absl::AnyInvocable<absl::Span<float>(size_t)>* output_allocator = nullptr;
};

// Availability is process-wide; vector length belongs to the calling thread.
bool IsEmbeddingBagSveAvailable();
int EmbeddingBagSveVectorLengthBytes();

// Executes one request and optionally reports the selected backend.
absl::Status ComputeEmbeddingActivationsF32(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<float> buffer,
    EmbeddingBagBackend requested, EmbeddingBagBackend* selected = nullptr);

// Executes cost-balanced bag ranges after validating the complete group.
// Re-entry from the supplied persistent pool executes inline.
absl::Status ExecuteEmbeddingBagBatch(
    absl::Span<const EmbeddingBagRequest> requests,
    EmbeddingBagBackend requested, Eigen::ThreadPoolInterface* thread_pool);

}  // namespace xla_serving::data::cpu_embedding::internal

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_H_

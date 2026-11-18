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

// Work threshold (in float32 work elements: output + looked-up rows) below
// which thread-pool task scheduling overhead dominates compute time. For small
// workloads (<= 64K elements, 256 KB, matching private CPU L2 cache size),
// sequential execution avoids thread wake and synchronization latency.
inline constexpr size_t kInlineWorkThreshold = 65536;

// Backend selection for standard FP32 embedding-bag reductions.
enum class EmbeddingBagBackend {
  // Uses SVE only when every condition below holds, and the portable backend
  // otherwise:
  //   - SVE is available (see IsEmbeddingBagSveAvailable()).
  //   - The table's embedding dimension is 128.
  //   - The input is unweighted and the table combine mode is kSum.
  //   - The executing thread's SVE vector length equals
  //     kAutoSveVectorLengthBytes. This is checked per worker thread at
  //     execution time because the vector length is thread-local.
  kAuto,
  // Always uses the portable Eigen implementation.
  kPortable,
  // Always uses SVE. Fails with FailedPrecondition if SVE was not compiled in
  // or is unavailable on the running CPU.
  kSve,
};

// A single table lookup: a borrowed CSR view over one feature's ids plus the
// table to reduce against. Borrowed inputs, the allocator, and allocated
// output storage must remain valid until the synchronous call returns.
struct EmbeddingBagInput {
  absl::Span<const int64_t> embedding_ids;
  absl::Span<const int32_t> row_pointers;
  const EmbeddingTable* embedding_table = nullptr;
  absl::Span<const float> weights;
  absl::AnyInvocable<absl::Span<float>(size_t)>* output_allocator = nullptr;
  absl::Span<const int32_t> embedding_ids_i32 = {};
};

// Availability is process-wide; vector length belongs to the calling thread.
bool IsEmbeddingBagSveAvailable();
int EmbeddingBagSveVectorLengthBytes();

// Executes one input and optionally reports the selected backend.
absl::Status ComputeEmbeddingActivationsF32(
    const EmbeddingBagInput& input, EmbeddingBagBackend requested,
    EmbeddingBagBackend* selected = nullptr);

// Executes cost-balanced bag ranges after validating the complete group.
// Runs inline on the calling thread when the total work size across all
// inputs is at most `inline_work_threshold` elements, when the pool has a
// single thread, or when called from one of the pool's own threads.
absl::Status ExecuteEmbeddingBagBatch(
    absl::Span<const EmbeddingBagInput> inputs, EmbeddingBagBackend requested,
    Eigen::ThreadPoolInterface* thread_pool,
    size_t inline_work_threshold = kInlineWorkThreshold);

}  // namespace xla_serving::data::cpu_embedding::internal

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_H_

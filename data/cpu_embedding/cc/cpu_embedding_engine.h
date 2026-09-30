// Copyright 2026 Google LLC
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

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding {

class CpuEmbeddingEngine final {
 public:
  struct Options {
    int max_cpu_slots = 8;
    // Only the in-place, single-feature Execute<T>() uses this setting.
    size_t block_size = 64;
  };

  // Borrowed inputs and allocator-owned outputs must remain valid until
  // ExecuteBatch() returns.
  struct BatchRequest {
    absl::string_view feature_name;
    absl::Span<const int64_t> embedding_ids;
    absl::Span<const int32_t> row_pointers;
    absl::Span<const float> weights;
    absl::AnyInvocable<absl::Span<float>(size_t)> output_allocator;
  };

  CpuEmbeddingEngine();
  explicit CpuEmbeddingEngine(Options options);
  ~CpuEmbeddingEngine();

  absl::Status RegisterFeatureSpec(FeatureSpec feature_spec);

  absl::Status AddLocalTable(std::string name,
                             std::shared_ptr<const EmbeddingTable> table);

  absl::StatusOr<std::vector<Array>> Execute(
      absl::Span<const FeatureBatchCsrInput> feature_batches) const;

  // Executes standard non-sequence FP32 bags without custom combiners on the
  // shared pool. Validates request inputs before allocation and allocated
  // output spans before writing. Unsupported sequences and custom combiners
  // return Unimplemented without invoking any output allocator.
  absl::Status ExecuteBatch(absl::Span<BatchRequest> requests) const;

  template <typename T>
  absl::StatusOr<absl::Span<T>> Execute(
      absl::string_view feature_name, absl::Span<const int64_t> embedding_ids,
      absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
      absl::AnyInvocable<absl::Span<T>(size_t)> allocator) const;

  void Clear();

  absl::StatusOr<FeatureSpec> GetFeatureSpec(absl::string_view feature_name)
      const ABSL_LOCKS_EXCLUDED(feature_specs_mu_);

  absl::StatusOr<std::shared_ptr<const EmbeddingTable>> GetLocalTable(
      absl::string_view table_name) const ABSL_LOCKS_EXCLUDED(local_tables_mu_);

 private:
  mutable absl::Mutex feature_specs_mu_;
  mutable absl::Mutex local_tables_mu_;

  absl::flat_hash_map<std::string, FeatureSpec> feature_specs_
      ABSL_GUARDED_BY(feature_specs_mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<const EmbeddingTable>>
      local_tables_ ABSL_GUARDED_BY(local_tables_mu_);

  Options options_;
  mutable std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_;
};

}  // namespace xla_serving::data::cpu_embedding

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_

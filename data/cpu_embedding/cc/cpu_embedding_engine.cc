#define EIGEN_USE_THREADS

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

#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/blocking_counter.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/compute_activations.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_bag.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding {
namespace {

// Work threshold (in float32 elements / bytes equivalent) below which
// thread-pool task scheduling overhead dominates compute time. Empirically
// determined via DLRM CPU benchmark sweeps (Dim=16..128, BS=1..4096): for
// workloads <= 64K elements (256 KB, matching private CPU L2 cache size),
// sequential execution avoids mutex, thread wake, and interconnect
// synchronization latency.
constexpr size_t kWorkThreshold = 65536;

}  // namespace

CpuEmbeddingEngine::CpuEmbeddingEngine() : CpuEmbeddingEngine(Options{}) {}

CpuEmbeddingEngine::CpuEmbeddingEngine(Options options) : options_(options) {
  const int max_cpu_slots =
      options_.max_cpu_slots > 0 ? options_.max_cpu_slots : 8;

  thread_pool_ = std::make_unique<Eigen::ThreadPool>(max_cpu_slots);
}

CpuEmbeddingEngine::~CpuEmbeddingEngine() = default;

absl::Status CpuEmbeddingEngine::RegisterFeatureSpec(FeatureSpec feature_spec) {
  absl::MutexLock lock(feature_specs_mu_);
  auto [it, inserted] =
      feature_specs_.insert({feature_spec.name, feature_spec});
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Feature spec already registered: ", feature_spec.name));
  }
  return absl::OkStatus();
}

absl::Status CpuEmbeddingEngine::AddLocalTable(
    std::string name, std::shared_ptr<const EmbeddingTable> table) {
  absl::MutexLock lock(local_tables_mu_);
  auto [it, inserted] =
      local_tables_.emplace(std::move(name), std::move(table));
  if (!inserted) {
    return absl::AlreadyExistsError(
        absl::StrCat("Table already exists: ", it->first));
  }
  return absl::OkStatus();
}

void CpuEmbeddingEngine::Clear() {
  // Match ExecuteBatch's lock order to keep specs and tables consistent.
  absl::MutexLock spec_lock(feature_specs_mu_);
  absl::MutexLock table_lock(local_tables_mu_);
  feature_specs_.clear();
  local_tables_.clear();
}

absl::StatusOr<std::vector<Array>> CpuEmbeddingEngine::Execute(
    absl::Span<const FeatureBatchCsrInput> feature_batches) const {
  absl::Mutex status_mu;
  absl::Status status;
  std::vector<Array> output(feature_batches.size());
  absl::BlockingCounter counter(feature_batches.size());

  // Pre-lookup all feature specs and tables on the main thread to avoid
  // parallel lock contention.
  std::vector<FeatureSpec> resolved_specs;
  resolved_specs.reserve(feature_batches.size());
  {
    absl::MutexLock spec_lock(feature_specs_mu_);
    for (const auto& batch : feature_batches) {
      auto spec_it = feature_specs_.find(batch.Name());
      if (spec_it == feature_specs_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Feature spec not found for: ", batch.Name()));
      }
      resolved_specs.push_back(spec_it->second);
    }
  }

  std::vector<std::shared_ptr<const EmbeddingTable>> resolved_tables;
  resolved_tables.reserve(feature_batches.size());
  {
    absl::MutexLock table_lock(local_tables_mu_);
    for (const auto& spec : resolved_specs) {
      auto table_it = local_tables_.find(spec.table_name);
      if (table_it == local_tables_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Table not found for: ", spec.table_name));
      }
      resolved_tables.push_back(table_it->second);
    }
  }

  for (size_t feature_batch_no = 0; feature_batch_no < feature_batches.size();
       ++feature_batch_no) {
    thread_pool_->Schedule([&, feature_batch_no]() {
      auto process_batch = [&]() -> absl::Status {
        const FeatureBatchCsrInput& feature_batch =
            feature_batches[feature_batch_no];
        const FeatureSpec& feature_spec = resolved_specs[feature_batch_no];
        const EmbeddingTable& embedding_table =
            *resolved_tables[feature_batch_no];

        if (feature_spec.max_sequence_length == 0) {
          ASSIGN_OR_RETURN(
              output[feature_batch_no],
              ComputeEmbeddingActivations(feature_batch, embedding_table));
        } else {
          ASSIGN_OR_RETURN(
              output[feature_batch_no],
              LookupEmbeddingSequence(feature_batch, embedding_table,
                                      feature_spec.max_sequence_length));
        }
        return absl::OkStatus();
      };

      absl::Status s = process_batch();
      if (!s.ok()) {
        absl::MutexLock lock(status_mu);
        status.Update(s);
      }
      counter.DecrementCount();
    });
  }

  counter.Wait();

  if (status.ok()) {
    return output;
  }
  return status;
}

absl::Status CpuEmbeddingEngine::ExecuteBatch(
    absl::Span<BatchRequest> requests) const {
  if (requests.empty()) {
    return absl::InvalidArgumentError(
        "Embedding batch requires at least one request.");
  }

  // Resolve specs and tables under the same lock order as Clear.
  std::vector<std::shared_ptr<const EmbeddingTable>> resolved_tables;
  resolved_tables.reserve(requests.size());
  {
    absl::MutexLock spec_lock(feature_specs_mu_);
    absl::MutexLock table_lock(local_tables_mu_);
    for (size_t index = 0; index < requests.size(); ++index) {
      const BatchRequest& request = requests[index];
      auto spec_it = feature_specs_.find(request.feature_name);
      if (spec_it == feature_specs_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Feature spec not found for: ", request.feature_name));
      }
      if (spec_it->second.max_sequence_length != 0) {
        return absl::UnimplementedError(
            absl::StrCat("Batch request ", index,
                         " requires a non-sequence feature specification."));
      }
      auto table_it = local_tables_.find(spec_it->second.table_name);
      if (table_it == local_tables_.end()) {
        return absl::NotFoundError(
            absl::StrCat("Table not found for: ", spec_it->second.table_name));
      }
      resolved_tables.push_back(table_it->second);
    }
  }

  std::vector<internal::EmbeddingBagRequest> batch_requests;
  batch_requests.reserve(requests.size());
  for (size_t index = 0; index < requests.size(); ++index) {
    batch_requests.push_back({
        .embedding_ids = requests[index].embedding_ids,
        .row_pointers = requests[index].row_pointers,
        .embedding_table = resolved_tables[index].get(),
        .weights = requests[index].weights,
        .output_allocator = &requests[index].output_allocator,
    });
  }

  return internal::ExecuteEmbeddingBagBatch(
      absl::MakeSpan(batch_requests), internal::EmbeddingBagBackend::kAuto,
      thread_pool_.get());
}

// The in-place path retains custom-combiner support and block_size tuning.
// TODO: Consider routing standard FP32 calls through ExecuteBatch after its
// small-workload and block-size behavior is matched.
template <typename T>
absl::StatusOr<absl::Span<T>> CpuEmbeddingEngine::Execute(
    absl::string_view feature_name, absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    absl::AnyInvocable<absl::Span<T>(size_t)> allocator) const {
  ASSIGN_OR_RETURN(const FeatureSpec feature_spec,
                   GetFeatureSpec(feature_name));
  if (feature_spec.max_sequence_length != 0) {
    return absl::UnimplementedError(
        "Sequence assembly is not supported for in-place execution.");
  }
  ASSIGN_OR_RETURN(std::shared_ptr<const EmbeddingTable> embedding_table,
                   GetLocalTable(feature_spec.table_name));
  if (row_pointers.size() < 2) {
    return absl::InvalidArgumentError(
        "Row pointers must have at least 2 elements.");
  }
  if (row_pointers.front() < 0) {
    return absl::InvalidArgumentError("Row pointers must be non-negative.");
  }
  bool is_monotonic = true;
  const size_t size_minus_1 = row_pointers.size() - 1;
  for (size_t i = 0; i < size_minus_1; ++i) {
    is_monotonic &= (row_pointers[i] <= row_pointers[i + 1]);
  }
  if (!is_monotonic) {
    return absl::InvalidArgumentError("Row pointers must be monotonic.");
  }
  if (static_cast<size_t>(row_pointers.back()) > embedding_ids.size()) {
    return absl::InvalidArgumentError(
        "Final element of row pointers exceeds embedding IDs size.");
  }
  if (!weights.empty() &&
      static_cast<size_t>(row_pointers.back()) > weights.size()) {
    return absl::InvalidArgumentError(
        "Weights size is less than the final element of row pointers.");
  }
  const size_t num_samples = row_pointers.size() - 1;
  const size_t dim = embedding_table->EmbeddingDim();
  absl::Span<T> buffer = allocator(dim * num_samples);

  const size_t block_size = options_.block_size;
  const int slots = options_.max_cpu_slots > 0 ? options_.max_cpu_slots : 8;

  // Optimize: Avoid thread pool overhead for small workloads or single-thread
  // configs.
  if (slots <= 1 || num_samples * dim <= kWorkThreshold) {
    RETURN_IF_ERROR(ComputeEmbeddingActivations(
        embedding_ids, row_pointers, weights, *embedding_table, buffer));
    return buffer;
  }

  if (block_size == std::numeric_limits<size_t>::max()) {
    // Mode: Eigen Loop Partitioner (parallelFor)
    Eigen::ThreadPoolDevice device(thread_pool_.get(), slots);

    // Cost model per sample: bytes loaded/stored, computation cycles
    // Load: row_pointers (4B) + average index read (4B) + weights read (dim*4B)
    // Store: outputs (dim*4B)
    Eigen::TensorOpCost cost(8 + dim * 4, dim * 4, dim * 2);

    absl::Mutex status_mu;
    absl::Status status;

    device.parallelFor(
        num_samples, cost, [&](Eigen::Index start, Eigen::Index end) {
          auto block_buffer = buffer.subspan(start * dim, (end - start) * dim);
          absl::Status s = ComputeEmbeddingActivations(
              embedding_ids, row_pointers.subspan(start, end - start + 1),
              weights, *embedding_table, block_buffer);
          if (!s.ok()) {
            absl::MutexLock lock(status_mu);
            status.Update(s);
          }
        });
    RETURN_IF_ERROR(status);
  } else {
    // Mode: Manual Blocking (Static or Custom Dynamic)
    size_t active_block_size = block_size;
    if (active_block_size == 0) {
      active_block_size =
          std::max<size_t>(256, (num_samples + (slots * 2) - 1) / (slots * 2));
    }

    if (num_samples <= active_block_size) {
      RETURN_IF_ERROR(ComputeEmbeddingActivations(
          embedding_ids, row_pointers, weights, *embedding_table, buffer));
    } else {
      const size_t num_blocks =
          (num_samples + active_block_size - 1) / active_block_size;
      absl::BlockingCounter counter(num_blocks);
      absl::Mutex status_mu;
      absl::Status status;

      for (size_t block = 0; block < num_blocks; ++block) {
        size_t start = block * active_block_size;
        size_t end = std::min(start + active_block_size, num_samples);

        thread_pool_->Schedule([&, start, end]() {
          auto block_buffer = buffer.subspan(start * dim, (end - start) * dim);
          absl::Status s = ComputeEmbeddingActivations(
              embedding_ids, row_pointers.subspan(start, end - start + 1),
              weights, *embedding_table, block_buffer);
          if (!s.ok()) {
            absl::MutexLock lock(status_mu);
            status.Update(s);
          }
          counter.DecrementCount();
        });
      }
      counter.Wait();
      RETURN_IF_ERROR(status);
    }
  }
  return buffer;
}

template absl::StatusOr<absl::Span<float>> CpuEmbeddingEngine::Execute(
    absl::string_view feature_name, absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    absl::AnyInvocable<absl::Span<float>(size_t)> allocator) const;

template absl::StatusOr<absl::Span<Eigen::bfloat16>>
CpuEmbeddingEngine::Execute(
    absl::string_view feature_name, absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    absl::AnyInvocable<absl::Span<Eigen::bfloat16>(size_t)> allocator) const;

absl::StatusOr<FeatureSpec> CpuEmbeddingEngine::GetFeatureSpec(
    absl::string_view feature_name) const {
  absl::MutexLock lock(feature_specs_mu_);
  auto it = feature_specs_.find(feature_name);
  if (it == feature_specs_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Feature spec not found for feature: ", feature_name));
  }
  return it->second;
}

absl::StatusOr<std::shared_ptr<const EmbeddingTable>>
CpuEmbeddingEngine::GetLocalTable(absl::string_view table_name) const {
  absl::MutexLock lock(local_tables_mu_);
  auto it = local_tables_.find(table_name);
  if (it == local_tables_.end()) {
    return absl::NotFoundError(
        absl::StrCat("Table not found for table: ", table_name));
  }
  return it->second;
}

}  // namespace xla_serving::data::cpu_embedding

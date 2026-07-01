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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_SCHEDULER_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_SCHEDULER_UTILS_H_

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/batch_stats.h"  // from @neptune

namespace neptune {

// Returns the next allowed batch size, which is the smallest allowed batch size
// greater than or equal to the given batch size. If allowed_batch_sizes,
// returns batch_size as is.
int32_t GetNextAllowedBatchSize(int batch_size,
                                const std::vector<int32_t>& allowed_batch_sizes,
                                bool disable_padding);

// Returns the largest allowed batch size that is smaller than or equal to
// batch_size. Returns batch_size if no such size exists.
int32_t GetPrevAllowedBatchSize(int batch_size,
                                const std::vector<int32_t>& allowed_batch_sizes,
                                bool disable_padding);

// Applies the batch padding policy to the candidate size and returns the target
// size depending on the policy.
int32_t ApplyBatchPaddingPolicy(
    int candidate_size, const std::vector<int32_t>& allowed_batch_sizes,
    bool disable_padding, absl::string_view batch_padding_policy,
    ModelBatchStats* model_batch_stats);

// Trims the batch to the next allowed batch size when possible and when
// configured by batch_padding_policy.
//
// When trimming, this function puts the trimmed tasks go into the
// out_trimmed_tasks vector in the same order as they were in the batch.
template <typename TaskType>
void MaybeBatchDown(Batch<TaskType>& batch,
               const std::vector<int32_t>& allowed_batch_sizes,
               bool disable_padding, absl::string_view batch_padding_policy,
               ModelBatchStats* model_batch_stats,
               std::vector<std::unique_ptr<TaskType>>& out_trimmed_tasks) {
  if (batch.empty()) {
    return;
  }
  int32_t batch_size = batch.size();
  int32_t target_batch_size = ApplyBatchPaddingPolicy(
      batch_size, allowed_batch_sizes, disable_padding, batch_padding_policy,
      model_batch_stats);
  if (target_batch_size < batch_size) {
    batch.TryTrimToNewSize(target_batch_size, out_trimmed_tasks);
  }
}

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_SCHEDULER_UTILS_H_

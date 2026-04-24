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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_STATS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_STATS_H_

#include <atomic>
#include <cstdint>
#include <optional>
#include <string>
#include <tuple>
#include <vector>

#include "absl/container/node_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "tsl/platform/thread_annotations.h"  // from @tsl

namespace neptune {

// Default values for when there is no recorded statistic in ModelBatchStats.
constexpr int64_t kNumBatchThreadsUnknown = -1;
constexpr int64_t kBatchTimeoutMicrosUnknown = -1;

// Tracks the average cost of registered samples.
//
// Thread-safe.
class CostTracker {
 public:
  // Registers a cost sample.
  void Register(absl::Duration cost) {
    DCHECK_GT(cost, absl::ZeroDuration());

    absl::MutexLock l(mu_);
    sample_count_++;
    sample_sum_ += cost;
  };

  // Returns the average cost of all registered samples, giving each sample
  // the same weight.
  //
  // Returns std::nullopt if no samples have been registered.
  //
  // TODO: Switch this to an exponentially-decaying average. It's
  // likely enough to set the half-life to the last 100-1000 samples.
  std::optional<absl::Duration> mean() const {
    int64_t count;
    absl::Duration sum;

    {
      // We only hold the lock to read the values and release it before later
      // performing a relatively slow division operation.
      absl::MutexLock l(mu_);
      count = sample_count_;
      sum = sample_sum_;
    }

    if (count == 0) return std::nullopt;

    return sum / count;
  };

 private:
  friend class CostTrackerTestPeer;

  // Resets the cost tracker. For testing only.
  void TestOnlyReset() {
    absl::MutexLock l(mu_);
    sample_count_ = 0;
    sample_sum_ = absl::ZeroDuration();
  }

  mutable absl::Mutex mu_;

  int64_t sample_count_ TF_GUARDED_BY(mu_) = 0;
  absl::Duration sample_sum_ TF_GUARDED_BY(mu_);
};

class CostTrackerTestPeer {
 public:
  static void Reset(CostTracker* tracker) {
    tracker->TestOnlyReset();
  }
};

// Tracks statistics for a particular model and batch size.
//
// Thread-safe.
class BatchSizeStats {
 public:
  CostTracker& tpu_cost() { return tpu_cost_; };

 private:
  CostTracker tpu_cost_;
};

// Tracks statistics for a particular model.
//
// Here, "model" means a specific version of a model (we assume that version is
// encoded in the model_version).
//
// Thread-safe.
class ModelBatchStats {
 public:
  // Returns a reference to the BatchSizeStats instance for the given batch
  // size.
  //
  // The returned reference persist for as long as 'this' is alive.
  BatchSizeStats& batch_size(int32_t batch_size) {
    absl::MutexLock l(mu_);
    return batch_size_stats_by_batch_size_[batch_size];
  }

  // Registers that the model server has processed a batch of size `size`
  // non-padding tasks for this model, updating the current cumulative
  // processed size.
  void RegisterProcessedSize(int64_t size) {
    cumulative_processed_size_.fetch_add(size, std::memory_order_relaxed);
  }

  // Returns the cumulative size processed by this model (the total
  // count of individual unit-sized queries processed by the model).
  int64_t cumulative_processed_size() const {
    return cumulative_processed_size_.load(std::memory_order_relaxed);
  }

  // Returns the list of batch sizes for which this model has statistics.
  //
  // The returned list is not guaranteed to be sorted.
  std::vector<int32_t> BatchSizes() const {
    std::vector<int32_t> result;
    absl::MutexLock l(mu_);
    result.reserve(batch_size_stats_by_batch_size_.size());
    for (const auto& [key, value] : batch_size_stats_by_batch_size_) {
      result.push_back(key);
    }
    return result;
  }

  void SetNumBatchThreads(int64_t num_batch_threads) {
    num_batch_threads_.store(num_batch_threads, std::memory_order_relaxed);
  }

  int64_t num_batch_threads() const {
    return num_batch_threads_.load(std::memory_order_relaxed);
  }

  void SetBatchTimeoutMicros(int64_t batch_timeout_micros) {
    batch_timeout_micros_.store(batch_timeout_micros,
                                std::memory_order_relaxed);
  }

  int64_t batch_timeout_micros() const {
    return batch_timeout_micros_.load(std::memory_order_relaxed);
  }

 private:
  mutable absl::Mutex mu_;

  // The storage of all BatchSizeStats instances.
  //
  // The mutex only protects adding/finding element in the map. Access to
  // elements themselves (after they were created) is not protected here. No
  // element deletion is possible because we return references to items in this
  // map and don't track their lifetime. We are using the node hash map so that
  // elements, once created, are fixed in memory.
  absl::node_hash_map<int32_t, BatchSizeStats> batch_size_stats_by_batch_size_
      TF_GUARDED_BY(mu_);

  // The total count of individual unit-sized queries processed by this model.
  // Can be used to generate an internal load metric per model. See
  // RegisterQuerySize for more details.
  std::atomic<int64_t> cumulative_processed_size_ = 0;

  // The number of batch threads assigned to this model.
  std::atomic<int64_t> num_batch_threads_ = kNumBatchThreadsUnknown;

  // The timeout in microseconds for this model (after which the current batch
  // is sent to be processed by the TPU).
  std::atomic<int64_t> batch_timeout_micros_ = kBatchTimeoutMicrosUnknown;
};

// Tracks batch statistics for all models.
//
// Thread-safe.
class BatchStatsRegistry {
 public:
  // Returns a reference to ModelBatchStats for the provided model_name and
  // version.
  //
  // Upon invocation with a not-yet-seen arguments, creates an empty
  // ModelBatchStats instance.
  //
  // The returned reference persist for as long as 'this' is alive.
  ModelBatchStats& model(const std::string& model_name, int64_t model_version) {
    std::tuple key(model_name, model_version);
    absl::MutexLock l(mu_);
    return model_batch_stats_by_model_name_and_version_[key];
  }

  // Returns a list of all model names and versions.
  //
  // This is the set of model names/versions tracked by this BatchStats
  // instance. Note that the returned list is not guaranteed to be sorted.
  std::vector<std::tuple<std::string, int64_t>> ModelNamesAndVersions() const {
    std::vector<std::tuple<std::string, int64_t>> result;
    absl::MutexLock l(mu_);
    result.reserve(model_batch_stats_by_model_name_and_version_.size());
    for (const auto& [key, value] :
         model_batch_stats_by_model_name_and_version_) {
      result.push_back(key);
    }
    return result;
  }

 private:
  mutable absl::Mutex mu_;

  // The storage of all ModelBatchStats instances.
  //
  // The mutex only protects adding/finding element in the map. Access to
  // elements themselves (after they were created) is not protected here. No
  // element deletion is possible because we return references to items in this
  // map and don't track their lifetime. We are using the node hash map for
  // element pointer stability.
  absl::node_hash_map<std::tuple<std::string, int64_t>, ModelBatchStats>
      model_batch_stats_by_model_name_and_version_ TF_GUARDED_BY(mu_);
};

// Returns the global instance of BatchStats, to be used for all production
// purposes (one should only instantiate individual classes from this file to
// test them).
inline BatchStatsRegistry& GlobalBatchStatsRegistry() {
  static BatchStatsRegistry* instance = new BatchStatsRegistry();
  return *instance;
}

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_BATCH_STATS_H_

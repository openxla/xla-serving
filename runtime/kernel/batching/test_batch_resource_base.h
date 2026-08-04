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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_TEST_BATCH_RESOURCE_BASE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_TEST_BATCH_RESOURCE_BASE_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "runtime/core/default_allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/kernel/batching/batch_resource_base.h"  // from @neptune
#include "runtime/kernel/batching/batch_runner_op.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

class TestBatchResourceBase : public BatchResourceBase {
 public:
  TestBatchResourceBase(
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatcherT::QueueOptions& batcher_queue_options,
      std::vector<int32_t> allowed_batch_sizes,
      std::vector<int32_t> low_priority_allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool)
      : BatchResourceBase(std::move(batcher), batcher_queue_options,
                          allowed_batch_sizes, thread_pool) {
    allowed_batch_sizes_ = absl::flat_hash_set<int64_t>(
        allowed_batch_sizes.begin(), allowed_batch_sizes.end());
    low_priority_allowed_batch_sizes_ =
        absl::flat_hash_set<int64_t>(low_priority_allowed_batch_sizes.begin(),
                                     low_priority_allowed_batch_sizes.end());
  }

  std::string DebugString() const override { return "TestBatchResourceBase"; }

  absl::Time propagated_task_deadline() const {
    absl::MutexLock l(mu_);
    return propagated_task_deadline_;
  }

  absl::StatusOr<const BatchResourceBase::BatchTask&> GetLatestDeadlineTask(
      const BatchT& batch) const {
    return BatchResourceBase::GetLatestDeadlineTask(batch);
  }

 protected:
  // Simple function that adds 1 to each element in the input buffer.
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& propagated_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    {
      absl::MutexLock l(mu_);
      propagated_task_deadline_ =
          propagated_task.context->run_context.cancellation_context.deadline;
    }
    for (const auto& input : inputs) {
      if (!allowed_batch_sizes_.empty() ||
          !low_priority_allowed_batch_sizes_.empty()) {
        CHECK(allowed_batch_sizes_.contains(input.shape().dims()[0]) ||
              low_priority_allowed_batch_sizes_.contains(
                  input.shape().dims()[0]));
      }
      combined_outputs->push_back(input);
      for (int i = 0; i < input.shape().num_elements(); ++i) {
        combined_outputs->back().flat<int32_t>()(i) += 1;
      }
    }
    done(absl::OkStatus());
  }
  absl::flat_hash_set<int64_t> allowed_batch_sizes_;
  absl::flat_hash_set<int64_t> low_priority_allowed_batch_sizes_;
  mutable absl::Mutex mu_;
  mutable absl::Time propagated_task_deadline_ ABSL_GUARDED_BY(mu_);
};

// Test when a batch function returns an error
class TestFailedBatchResource : public BatchResourceBase {
 public:
  TestFailedBatchResource(
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatcherT::QueueOptions& batcher_queue_options,
      std::vector<int32_t> allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool)
      : BatchResourceBase(std::move(batcher), batcher_queue_options,
                          std::move(allowed_batch_sizes), thread_pool) {}

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& propagated_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    done(absl::InternalError("Test failed batch resource"));
  }
};

class TestJsvBatchResource : public JsvBatchResource {
 public:
  TestJsvBatchResource(
      std::vector<std::unique_ptr<Kernel>> batched_graph,
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatchResourceBase::BatcherT::QueueOptions& queue_options,
      const std::vector<int32_t>& allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool)
      : JsvBatchResource(std::move(batched_graph), batcher, queue_options,
                         allowed_batch_sizes, thread_pool) {
    allowed_batch_sizes_.insert(allowed_batch_sizes.begin(),
                                allowed_batch_sizes.end());
  }

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    for (const auto& input : inputs) {
      CHECK(allowed_batch_sizes_.contains(input.shape().dims()[0]));
    }
    JsvBatchResource::ProcessFuncBatchImpl(last_task, std::move(batch_request_ids),
                                           inputs, combined_outputs, done);
  }

 private:
  absl::flat_hash_set<int32_t> allowed_batch_sizes_;
};

class TestRaggedBatchResourceBase : public BatchResourceBase {
 public:
  TestRaggedBatchResourceBase(
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatcherT::QueueOptions& batcher_queue_options,
      std::vector<int32_t> allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool,
      std::vector<RaggedInputSpec> ragged_inputs = {{.flat_values_index = 1,
                                                     .row_splits_index = 0}})
      : BatchResourceBase(std::move(batcher), batcher_queue_options,
                          allowed_batch_sizes, thread_pool,
                          std::move(ragged_inputs)),
        allocator_(batcher_queue_options.allocator
                       ? batcher_queue_options.allocator
                       : DefaultAllocator::GetInstance()) {
    allowed_batch_sizes_ = absl::flat_hash_set<int64_t>(
        allowed_batch_sizes.begin(), allowed_batch_sizes.end());
  }

  std::string DebugString() const override {
    return "TestRaggedBatchResourceBase";
  }

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& propagated_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    CHECK_GE(inputs.size(), 2);
    // Assumes input 0 is row_splits and input 1 is flat_values.
    int64_t batch_size = inputs[0].shape().dims()[0] - 1;
    if (!allowed_batch_sizes_.empty()) {
      CHECK(allowed_batch_sizes_.contains(batch_size));
    }
    if (propagated_task.forced_warmup_batch_size > 0) {
      CHECK_EQ(batch_size, propagated_task.forced_warmup_batch_size);
      CHECK_EQ(inputs[0].shape().dims()[0], batch_size + 1);
      CHECK_EQ(inputs[0].flat<int32_t>()(0), 0);
      for (int i = 1; i < inputs[0].shape().dims()[0]; ++i) {
        CHECK_GE(inputs[0].flat<int32_t>()(i), inputs[0].flat<int32_t>()(i - 1));
      }
      done(absl::OkStatus());
      return;
    }
    // For non-warmup requests, compute the sum of flat_values for each row.
    HostBuffer output(xla::ifrt::DType(xla::ifrt::DType::kS32),
                      xla::ifrt::Shape({batch_size, 1}), allocator_);
    for (int64_t r = 0; r < batch_size; ++r) {
      int32_t sum = 0;
      int32_t start_idx = inputs[0].flat<int32_t>()(r);
      int32_t end_idx = inputs[0].flat<int32_t>()(r + 1);
      for (int32_t idx = start_idx; idx < end_idx; ++idx) {
        sum += inputs[1].flat<int32_t>()(idx);
      }
      output.flat<int32_t>()(r) = sum;
    }
    combined_outputs->push_back(std::move(output));
    done(absl::OkStatus());
  }

 private:
  absl::flat_hash_set<int64_t> allowed_batch_sizes_;
  Allocator* allocator_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_TEST_BATCH_RESOURCE_BASE_H_

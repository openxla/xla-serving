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
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/kernel/batching/batch_resource_base.h"  // from @neptune
#include "runtime/kernel/batching/batch_runner_op.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
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
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    for (const auto& input : inputs) {
      CHECK(allowed_batch_sizes_.contains(input.shape().dims()[0]));
    }
    JsvBatchResource::ProcessFuncBatchImpl(last_task, inputs, combined_outputs,
                                           done);
  }

 private:
  absl::flat_hash_set<int32_t> allowed_batch_sizes_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_TEST_BATCH_RESOURCE_BASE_H_

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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_QUEUE_BASE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_QUEUE_BASE_H_

#include <stddef.h>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
namespace neptune {
namespace internal {

// An abstract base class for task queues used by SharedBatchScheduler.
// Accepts tasks and accumulates them into batches, and dispenses those batches
// to be processed via a "pull" interface. The queue's behavior is governed by
// maximum batch size, timeout and maximum queue length parameters; see their
// documentation in SharedBatchScheduler.
//
// Concrete subclasses act as specific strategy implementations:
// - FifoQueue: Implements strict eager batching.
// - PriorityAwareQueue: Implements a 4-level priority system.
template <typename TaskType>
class QueueBase {
 public:
  using ProcessBatchCallback =
      std::function<void(std::unique_ptr<Batch<TaskType>>)>;

  using SchedulableBatchCallback = std::function<void()>;

  // Orderable key representing the priority of a batch. Higher priority batches
  // will be prioritized for execution first (when using rank_queues=true).
  // - A smaller key value is higher priority than a larger one.
  // - This is a pair formed from <priority, batch_timestamp>. The exact values
  //   used are an implementation detail of PeekBatchPriority().
  using BatchPriorityKey = std::pair<int, int64_t>;

  QueueBase(
      const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
      tsl::Env* env, ProcessBatchCallback process_batch_callback,
      SchedulableBatchCallback schedulable_batch_callback);

  // Illegal to destruct unless the queue is empty. It is the responsibility of
  // derived classes to assert this in their destructors, as doing so here would
  // result in a pure virtual method call (since IsEmptyInternal() relies on a
  // pure virtual method).
  virtual ~QueueBase() = default;

  // Submits a task to the queue, with the same semantics as
  // BatchScheduler::Schedule().
  absl::Status Schedule(std::unique_ptr<TaskType>* task);

  // Returns the number of enqueued tasks, with the same semantics as
  // BatchScheduler::NumEnqueuedTasks().
  size_t NumEnqueuedTasks() const;

  // Returns the queue capacity, with the same semantics as
  // BatchScheduler::SchedulingCapacity().
  size_t SchedulingCapacity() const;

  // Returns the maximum allowed size of tasks submitted to the queue.
  size_t max_task_size() const { return options_.input_batch_size_limit; }

  // Returns the maximum allowed size of tasks to be executed.
  // Returned value would be less than or equal to the maximum allowed input
  // size that's provided by caller of batch scheduler.
  size_t max_execution_batch_size() const { return max_execution_batch_size_; }

  // Called by a thread that is ready to process a batch, to request one from
  // this queue. Either returns a batch that is ready to be processed, or
  // nullptr if the queue declines to schedule a batch at this time. If it
  // returns a batch, the batch is guaranteed to be closed.
  typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr ScheduleBatch();

  // Without mutating the queue, checks if ScheduleBatch() will return a valid
  // batch and if so will return the priority of that batch.
  std::optional<BatchPriorityKey> PeekBatchPriority() const;

  // Processes a batch that has been returned earlier by ScheduleBatch().
  void ProcessBatch(std::unique_ptr<Batch<TaskType>> batch);

  // Determines whether the queue is empty, i.e. has no tasks waiting or being
  // processed.
  bool IsEmpty() const;

  // Marks the queue closed, and waits until it is empty.
  void CloseAndWaitUntilEmpty();

  bool closed() const ABSL_NO_THREAD_SAFETY_ANALYSIS { return closed_.load(); }

 protected:
  // Handles subclass-specific scheduling.
  // 'notify_schedulable' should be set to true if a batch became schedulable.
  virtual absl::Status ScheduleTaskInternal(std::unique_ptr<TaskType>* task,
                                            bool* notify_schedulable)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  virtual size_t NumEnqueuedTasksInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  virtual size_t SchedulingCapacityInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  virtual std::unique_ptr<Batch<TaskType>> ScheduleBatchInternal()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  virtual std::optional<BatchPriorityKey> PeekBatchPriorityInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  virtual bool IsEmptyTaskQueueInternal() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) = 0;

  bool IsEmptyInternal() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Computes the max_execution_batch_size of the queue based on queue options.
  static size_t GetMaxExecutionBatchSize(
      const typename SharedBatchScheduler<TaskType>::QueueOptions& options) {
    // If `enable_large_batch_splitting`, returns `max_execution_batch_size`
    // configured by user options directly; returns `input_batch_size_limit`
    // otherwise.
    //
    // Note `input_batch_size_limit` is used for backward compatibility ->
    // users may not specify `max_execution_batch_size` explicitly.
    if (options.enable_large_batch_splitting) {
      return options.max_execution_batch_size;
    } else {
      return options.input_batch_size_limit;
    }
  }

  const typename SharedBatchScheduler<TaskType>::QueueOptions options_;

  // The environment to use.
  tsl::Env* env_;

  // The maximum batch size to be executed by `Queue::ProcessBatch`.
  // See the comment of QueueOptions and helper function
  // `GetMaxExecutionBatchSize` for more details on what it means.
  const size_t max_execution_batch_size_;

  // A callback invoked to processes a batch of work units. Always invoked
  // from a batch thread.
  ProcessBatchCallback process_batch_callback_;

  // A callback invoked to notify the scheduler that a new batch has become
  // schedulable.
  SchedulableBatchCallback schedulable_batch_callback_;

  mutable absl::Mutex mu_;

  // Whether this queue can accept new tasks. This variable is monotonic: it
  // starts as false, and then at some point gets set to true and remains true
  // for the duration of this object's life.
  std::atomic<bool> closed_ ABSL_GUARDED_BY(mu_){false};

  // The number of batches currently being processed by batch threads.
  // Incremented in ScheduleBatch() and decremented in ProcessBatch().
  int num_batches_being_processed_ ABSL_GUARDED_BY(mu_) = 0;

  QueueBase(const QueueBase&) = delete;
  QueueBase& operator=(const QueueBase&) = delete;
};

template <typename TaskType>
QueueBase<TaskType>::QueueBase(
    const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
    ::tsl::Env* env, ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : options_(options),
      env_(env),
      max_execution_batch_size_(GetMaxExecutionBatchSize(options_)),
      process_batch_callback_(process_batch_callback),
      schedulable_batch_callback_(schedulable_batch_callback) {}

template <typename TaskType>
absl::Status QueueBase<TaskType>::Schedule(std::unique_ptr<TaskType>* task) {
  const bool large_batch_splitting = options_.enable_large_batch_splitting;
  tsl::profiler::TraceMe trace_me([task, large_batch_splitting] {
    return tsl::profiler::TraceMeEncode(
        large_batch_splitting ? "OrbaxServing::ScheduleWithEagerSplit"
                              : "OrbaxServing::ScheduleWithoutSplit",
        {{"batching_input_task_size", (*task)->size()}});
  });

  bool notify_of_schedulable_batch = false;
  {
    absl::MutexLock l(mu_);
    DCHECK(!closed_);

    TF_RETURN_IF_ERROR(
        ScheduleTaskInternal(task, &notify_of_schedulable_batch));
  }

  if (notify_of_schedulable_batch) {
    schedulable_batch_callback_();
  }

  return absl::OkStatus();
}

template <typename TaskType>
size_t QueueBase<TaskType>::NumEnqueuedTasks() const {
  absl::MutexLock l(mu_);
  return NumEnqueuedTasksInternal();
}

template <typename TaskType>
size_t QueueBase<TaskType>::SchedulingCapacity() const {
  absl::MutexLock l(mu_);
  return SchedulingCapacityInternal();
}

template <typename TaskType>
typename SharedBatchScheduler<TaskType>::BatchTaskUniquePtr
QueueBase<TaskType>::ScheduleBatch() {
  // The batch to schedule, which we may populate below. (If left as nullptr,
  // that means we are electing not to schedule a batch at this time.)
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;

  {
    absl::MutexLock l(mu_);
    batch_to_schedule = ScheduleBatchInternal();

    if (batch_to_schedule != nullptr) {
      ++num_batches_being_processed_;
    }
  }
  return batch_to_schedule;
}

template <typename TaskType>
void QueueBase<TaskType>::ProcessBatch(std::unique_ptr<Batch<TaskType>> batch) {
  tsl::profiler::TraceMeConsumer trace_me(
      [&] {
        return tsl::profiler::TraceMeEncode(
            "OrbaxServing::ProcessBatch",
            {{"batch_size_before_padding", batch->size()},
             {"_r", 2} /*root_event*/});
      },
      tsl::profiler::ContextType::kSharedBatchScheduler,
      batch->traceme_context_id());

  process_batch_callback_(std::move(batch));

  {
    absl::MutexLock l(mu_);
    --num_batches_being_processed_;
  }
}

template <typename TaskType>
bool QueueBase<TaskType>::IsEmpty() const {
  absl::MutexLock l(mu_);
  return IsEmptyInternal();
}

template <typename TaskType>
void QueueBase<TaskType>::CloseAndWaitUntilEmpty() {
  absl::MutexLock l(mu_);
  closed_ = true;
  auto is_empty = [this]() ABSL_EXCLUSIVE_LOCKS_REQUIRED(mu_) {
    return IsEmptyInternal();
  };
  mu_.Await(absl::Condition(&is_empty));
}

template <typename TaskType>
bool QueueBase<TaskType>::IsEmptyInternal() const {
  return num_batches_being_processed_ == 0 && IsEmptyTaskQueueInternal();
}

template <typename TaskType>
std::optional<typename QueueBase<TaskType>::BatchPriorityKey>
QueueBase<TaskType>::PeekBatchPriority() const {
  {
    absl::MutexLock l(mu_);
    return PeekBatchPriorityInternal();
  }
}

}  // namespace internal
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_QUEUE_BASE_H_

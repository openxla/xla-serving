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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_FIFO_QUEUE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_FIFO_QUEUE_H_

#include <stddef.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler_utils.h"  // from @neptune
#include "runtime/core/batching/queue_base.h"  // from @neptune
#include "runtime/core/batching/queue_options.h"  // from @neptune
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "tsl/profiler/lib/connected_traceme.h"  // from @tsl
#include "tsl/profiler/lib/context_types.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace xla_serving {
namespace internal {

template <typename TaskType>
class FifoQueue : public QueueBase<TaskType> {
 public:
  using ProcessBatchCallback =
      typename QueueBase<TaskType>::ProcessBatchCallback;
  using SchedulableBatchCallback =
      typename QueueBase<TaskType>::SchedulableBatchCallback;
  using SplitInputTaskIntoSubtasksCallback = std::function<absl::Status(
      std::unique_ptr<TaskType>* input_task, int open_batch_remaining_slot,
      int max_execution_batch_size,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>;
  using BatchPriorityKey = typename QueueBase<TaskType>::BatchPriorityKey;

  FifoQueue(
      const QueueOptions<TaskType>& options,
      tsl::Env* env, ProcessBatchCallback process_batch_callback,
      SchedulableBatchCallback schedulable_batch_callback);

  // Illegal to destruct unless the queue is empty.
  ~FifoQueue();

 protected:
  // Enqueues `task` as it is or split it inline (eagerly) to form batches to be
  // processed by `QueueBase<TaskType>::ProcessBatch`.
  absl::Status ScheduleTaskInternal(std::unique_ptr<TaskType>* task,
                                    bool* notify_schedulable) override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  size_t NumEnqueuedTasksInternal() const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  size_t SchedulingCapacityInternal() const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::unique_ptr<Batch<TaskType>> ScheduleBatchInternal() override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::optional<BatchPriorityKey> PeekBatchPriorityInternal() const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  bool IsEmptyTaskQueueInternal() const override
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

 private:
  // Closes the open batch residing at the back of std::deque, and inserts a
  // fresh open batch behind it.
  void StartNewBatch() ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Split `input task` into `output_tasks` according to 'task_sizes'.
  absl::Status SplitInputBatchIntoSubtasks(
      std::unique_ptr<TaskType>* input_task,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Determines whether the open batch residing at the back of
  // 'batches_' is currently schedulable.
  bool IsOpenBatchSchedulable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Returns an error if queue doesn't have capacity for this task.
  //
  // `task` must outlive this method.
  absl::Status ValidateBatchTaskQueueCapacity(TaskType* task) const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // The task size of the last batch in the queue.
  size_t tail_batch_task_size() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Returns the number of enqueued batches.
  int64_t num_enqueued_batches() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Gets the appropriate batches.
  std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches()
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Gets the appropriate batches (const version).
  const std::deque<std::unique_ptr<Batch<TaskType>>>& GetBatches() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // The enqueued batches for inputs. Each element corresponds to a task to be
  // dequeued and processed by `QueueBase<TaskType>::ProcessBatch`.
  std::deque<std::unique_ptr<Batch<TaskType>>> batches_
      ABSL_GUARDED_BY(this->mu_);

  // The counter of the TraceMe context ids.
  uint64_t traceme_context_id_counter_ ABSL_GUARDED_BY(this->mu_) = 0;

  // The time at which the first task was added to the open (back-most) batch in
  // 'batches_'. Valid iff that batch contains at least one task.
  //
  // Note that when using a batch padding policy other than PAD_UP, this field
  // might contain an approximate value.
  uint64_t open_batch_start_time_micros_ ABSL_GUARDED_BY(this->mu_);

  // Whether this queue contains a batch that is eligible to be scheduled.
  // Used to keep track of when to call 'schedulable_batch_callback_'.
  bool schedulable_batch_ ABSL_GUARDED_BY(this->mu_) = false;
};

template <typename TaskType>
FifoQueue<TaskType>::FifoQueue(
    const QueueOptions<TaskType>& options,
    tsl::Env* env, ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : QueueBase<TaskType>(options, env, process_batch_callback,
                          schedulable_batch_callback) {
  // Set the higher 32 bits of traceme_context_id_counter_ to be the creation
  // time of the queue. This prevents the batches in different queues to have
  // the same traceme_context_id_counter_.
  traceme_context_id_counter_ = (absl::GetCurrentTimeNanos() & 0xFFFFFFFF)
                                << 32;
  GetBatches().emplace_back(new Batch<TaskType>);
}

template <typename TaskType>
FifoQueue<TaskType>::~FifoQueue() {
  absl::MutexLock l(this->mu_);
  DCHECK(this->IsEmptyInternal());
  GetBatches().back()->Close();
}

template <typename TaskType>
absl::Status FifoQueue<TaskType>::ScheduleTaskInternal(
    std::unique_ptr<TaskType>* task, bool* notify_schedulable) {
  // TODO:
  // Add test coverage when when concurrent incoming batches arrives and
  // use up all queue capacity.
  TF_RETURN_IF_ERROR(ValidateBatchTaskQueueCapacity((*task).get()));

  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  const int64_t open_batch_remaining_slot =
      this->max_execution_batch_size() - batches.back()->size();

  const int64_t input_task_size = (*task)->size();

  std::vector<std::unique_ptr<TaskType>> output_tasks;

  if (input_task_size <= open_batch_remaining_slot ||
      !this->options_.enable_large_batch_splitting) {
    // This is the fast path when input doesn't need to be split.
    output_tasks.push_back(std::move(*task));
  } else {
    TF_RETURN_IF_ERROR(SplitInputBatchIntoSubtasks(task, &output_tasks));
  }

  for (int i = 0; i < output_tasks.size(); ++i) {
    if (batches.back()->size() + output_tasks[i]->size() >
        this->max_execution_batch_size()) {
      StartNewBatch();
    }
    if (batches.back()->empty()) {
      open_batch_start_time_micros_ = this->env_->NowMicros();
    }
    tsl::profiler::TraceMeProducer trace_me(
        [&output_tasks, i] {
          return tsl::profiler::TraceMeEncode(
              "OrbaxServing::ScheduleOutputTask",
              {{"size", output_tasks[i]->size()}});
        },
        tsl::profiler::ContextType::kSharedBatchScheduler,
        batches.back()->traceme_context_id());
    batches.back()->AddTask(std::move(output_tasks[i]),
                            this->env_->NowMicros());
  }

  // Check if the batch queue has a schedulable batch and mark it schedulable
  // if it not already marked.
  if (!schedulable_batch_) {
    if (GetBatches().size() > 1 || IsOpenBatchSchedulable()) {
      schedulable_batch_ = true;
      *notify_schedulable = true;
    }
  }

  return absl::OkStatus();
}

template <typename TaskType>
size_t FifoQueue<TaskType>::NumEnqueuedTasksInternal() const {
  size_t num_enqueued_tasks = 0;
  for (const auto& batch : GetBatches()) {
    num_enqueued_tasks += batch->num_tasks();
  }
  return num_enqueued_tasks;
}

template <typename TaskType>
size_t FifoQueue<TaskType>::SchedulingCapacityInternal() const {
  const int64_t num_new_batches_schedulable =
      static_cast<int64_t>(this->options_.max_enqueued_batches) -
      this->num_enqueued_batches();
  const int64_t execution_batch_size_limit = this->max_execution_batch_size();
  const int64_t open_batch_capacity =
      execution_batch_size_limit - this->tail_batch_task_size();
  // Note the returned value is guaranteed to be not negative, since
  // enqueue operation could only happen if queue has enough capacity.
  return (num_new_batches_schedulable * execution_batch_size_limit) +
         open_batch_capacity;
}

template <typename TaskType>
absl::Status FifoQueue<TaskType>::ValidateBatchTaskQueueCapacity(
    TaskType* task) const {
  // Check if the task size is larger than the batch size limit, regardless of
  // the batch capacity.
  if (task->size() > this->options_.input_batch_size_limit) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Task size %d is larger than maximum input batch size %d", task->size(),
        this->options_.input_batch_size_limit));
  }

  if (this->options_.enable_large_batch_splitting) {
    if (task->size() > SchedulingCapacityInternal()) {
      return absl::UnavailableError(absl::StrCat(
          "The batch scheduling queue to which this task was submitted is "
          "full; task size is ",
          task->size(), " but scheduling capacity is only ",
          SchedulingCapacityInternal(),
          " (num_enqueued_batches=", num_enqueued_batches(),
          ", max_enqueued_batches=", this->options_.max_enqueued_batches,
          ", open_batch_size=", tail_batch_task_size(),
          ", max_execution_batch_size=", this->max_execution_batch_size(),
          ")"));
    }
    return absl::OkStatus();
  }

  // NOTE, the capacity checking below is loose and is retained
  // for backward compatibility that was broken due to the merge of no-split
  // and eager split.
  // There are existing clients/models that rely on the loose check
  // and can get errors after the merge. Retaining the old behavior
  // allows such models to continue to work.
  //
  // We need to revisit/remove this check after we fix model configs.
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  if (batches.back()->size() + task->size() >
      this->options_.input_batch_size_limit) {
    if (batches.size() >= this->options_.max_enqueued_batches) {
      return absl::UnavailableError(absl::StrCat(
          "The batch scheduling queue to which this task was submitted is "
          "full; currently ",
          batches.size(), " batches enqueued and max_enqueued_batches is ",
          this->options_.max_enqueued_batches));
    }
  }
  return absl::OkStatus();
}

template <typename TaskType>
std::unique_ptr<Batch<TaskType>> FifoQueue<TaskType>::ScheduleBatchInternal() {
  // The batch to schedule, which we may populate below. (If left as nullptr,
  // that means we are electing not to schedule a batch at this time.)
  std::unique_ptr<Batch<TaskType>> batch_to_schedule;

  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  // Consider closing the open batch at this time, to schedule it.
  if (batches.size() == 1 && IsOpenBatchSchedulable()) {
    // Support BatchPaddingPolicy::kBatchDown and
    // BatchPaddingPolicy::kMinimizeTpuCostPerRequest. We do this before
    // starting a new batch because starting a new batch will close the old
    // batch, making it read-only.
    Batch<TaskType>& old_batch = *batches[0];
    if (!old_batch.empty()) {
      uint64_t old_batch_time = old_batch.EarliestTaskStartTime().value();
      std::vector<std::unique_ptr<TaskType>> trimmed_tasks;
      MaybeBatchDown(
          /* batch= */ old_batch,
          /* allowed_batch_sizes= */ this->options_.allowed_batch_sizes,
          /* disable_padding= */ this->options_.disable_padding,
          /* batch_padding_policy= */ this->options_.batch_padding_policy,
          /* model_batch_stats= */ this->options_.model_batch_stats,
          /* out_trimmed_tasks= */ trimmed_tasks);

      StartNewBatch();

      // Move the trimmed tasks, if any, into the new batch.
      Batch<TaskType>& new_batch = *batches[1];
      for (std::unique_ptr<TaskType>& task : trimmed_tasks) {
        new_batch.AddTask(std::move(task), old_batch_time);
      }
      if (!new_batch.empty()) {
        // TODO - b/325954758: Reconsider the starting time of a trimmed
        // batch.
        //
        // Ideally, we'd set open_batch_start_time_micros_ to time we received
        // the first task in the open batch, but we don't have this
        // information here. For now, we're trying as alternative solution
        // that doesn't require adding time to each task: assume that requests
        // arrived at a steady rate and therefore use a point between the old
        // value of open_batch_start_time_micros_ and NOW.
        //
        // Let's say that originally, the batch had 10 requests, and we want
        // to schedule a batch of size 8 and leave 2 requests in the open
        // batch (new_batch). Then, variable `position` is 0.8, which means we
        // have to set open_batch_start_time_micros_ to be at a position of
        // 80% between open_batch_start_time_micros_ and now.
        double position = static_cast<double>(old_batch.size()) /
                          (old_batch.size() + new_batch.size());
        open_batch_start_time_micros_ +=
            (this->env_->NowMicros() - open_batch_start_time_micros_) *
            position;
      }
    }
  }
  if (batches.size() >= 2) {
    // There is at least one closed batch that is ready to be scheduled.
    batch_to_schedule = std::move(batches.front());
    batches.pop_front();
  }

  if (batch_to_schedule == nullptr) {
    // There is no batch that can be scheduled,
    // mark the condition false and return the nullptr.
    schedulable_batch_ = false;
    return batch_to_schedule;
  }

  return batch_to_schedule;
}

template <typename TaskType>
bool FifoQueue<TaskType>::IsEmptyTaskQueueInternal() const {
  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  return batches.size() == 1 && batches.back()->empty();
}

template <typename TaskType>
void FifoQueue<TaskType>::StartNewBatch() {
  std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();
  batches.back()->Close();
  batches.emplace_back(new Batch<TaskType>(++traceme_context_id_counter_));
}

template <typename TaskType>
absl::Status FifoQueue<TaskType>::SplitInputBatchIntoSubtasks(
    std::unique_ptr<TaskType>* input_task,
    std::vector<std::unique_ptr<TaskType>>* output_tasks) {
  const int open_batch_remaining_slot =
      this->max_execution_batch_size() - this->tail_batch_task_size();
  return this->options_.split_input_task_func(
      std::move(input_task), open_batch_remaining_slot,
      this->max_execution_batch_size(), std::move(output_tasks));
}

template <typename TaskType>
bool FifoQueue<TaskType>::IsOpenBatchSchedulable() const {
  return PeekBatchPriorityInternal().has_value();
}

template <typename TaskType>
std::optional<typename FifoQueue<TaskType>::BatchPriorityKey>
FifoQueue<TaskType>::PeekBatchPriorityInternal() const {
  const int kHighPriority = 1;

  const std::deque<std::unique_ptr<Batch<TaskType>>>& batches = GetBatches();

  if (batches.size() >= 2) {
    Batch<TaskType>* batch = batches.front().get();
    return std::make_pair(kHighPriority,
                          batch->EarliestTaskStartTime().value());
  }

  Batch<TaskType>* open_batch = batches.back().get();

  size_t effective_batch_size = open_batch->size();
  if (effective_batch_size == 0) {
    return std::nullopt;
  }

  uint64_t effective_start_time_micros = open_batch_start_time_micros_;
  int64_t effective_batch_timeout_micros = this->options_.batch_timeout_micros;

  bool schedulable =
      this->closed_ ||
      effective_batch_size >= this->max_execution_batch_size() ||
      this->env_->NowMicros() >=
          effective_start_time_micros + effective_batch_timeout_micros;

  if (!schedulable) {
    return std::nullopt;
  }

  return std::make_pair(kHighPriority, effective_start_time_micros);
}

template <typename TaskType>
size_t FifoQueue<TaskType>::tail_batch_task_size() const {
  return GetBatches().back()->size();
}

template <typename TaskType>
int64_t FifoQueue<TaskType>::num_enqueued_batches() const {
  return GetBatches().size();
}

template <typename TaskType>
std::deque<std::unique_ptr<Batch<TaskType>>>&
FifoQueue<TaskType>::GetBatches() {
  return batches_;
}

template <typename TaskType>
const std::deque<std::unique_ptr<Batch<TaskType>>>&
FifoQueue<TaskType>::GetBatches() const {
  return batches_;
}

}  // namespace internal
}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_FIFO_QUEUE_H_

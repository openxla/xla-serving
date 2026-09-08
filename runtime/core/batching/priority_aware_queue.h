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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_PRIORITY_AWARE_QUEUE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_PRIORITY_AWARE_QUEUE_H_

#include <stddef.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <optional>
#include <set>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler_utils.h"  // from @neptune
#include "runtime/core/batching/queue_base.h"  // from @neptune
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla

namespace neptune {
namespace internal {

// A priority-aware task queue for SharedBatchScheduler.
// Manages requests using a single priority queue ordered by criticality
// (highest first) and arrival time (FIFO within the same criticality).
// Forms batches lazily at execution time and supports task eviction on a full
// queue.
template <typename TaskType>
class PriorityAwareQueue : public QueueBase<TaskType> {
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

  PriorityAwareQueue(
      const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
      tsl::Env* env, ProcessBatchCallback process_batch_callback,
      SchedulableBatchCallback schedulable_batch_callback);

  // Illegal to destruct unless the queue is empty.
  ~PriorityAwareQueue() override;

 protected:
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
  struct QueueEntry {
    mutable std::unique_ptr<TaskType> task;
    // The timestamp when the task arrived and was enqueued. Represents the
    // start of the task's queuing life cycle. Used for FIFO ordering within the
    // same criticality and for batch timeout calculations.
    uint64_t start_time_micros;
    tsl::criticality::Criticality criticality;

    // Highest criticality first; FIFO within same criticality.
    bool operator<(const QueueEntry& other) const {
      if (criticality != other.criticality) {
        return criticality > other.criticality;
      }
      return start_time_micros < other.start_time_micros;
    }
  };

  bool CanSplitTask(const TaskType& task) const;

  // If queue has capacity, adds task to queue and returns OK.
  // If queue doesn't have capacity:
  // - If task has higher criticality than lowest criticality task in queue,
  //   evicts lowest criticality tasks until there is capacity, adds task to
  //   queue and returns OK.
  // - Otherwise, returns UnavailableError.
  // The entire task is either accepted or rejected; partial acceptance is not
  // supported.
  absl::Status AddTask(std::unique_ptr<TaskType>* task,
                       uint64_t start_time_micros)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  // Returns tasks whose sizes sum up to `size`; highest priority tasks are
  // considered first. Within a priority, tasks that came first are
  // considered first.
  std::vector<QueueEntry> RemoveTask(int size)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  void AddEntryInternal(QueueEntry entry)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  QueueEntry RemoveEntryInternal(
      typename std::multiset<QueueEntry>::iterator it)
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::optional<uint64_t> EarliestTaskStartTime() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::optional<uint64_t> EarliestHighPriorityTaskStartTime() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::optional<tsl::criticality::Criticality> HighestCriticality() const
      ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  tsl::criticality::Criticality GetCriticality(const TaskType& task) const;

  bool IsSchedulable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  bool IsOpenBatchSchedulable() const ABSL_EXCLUSIVE_LOCKS_REQUIRED(this->mu_);

  std::multiset<QueueEntry> tasks_ ABSL_GUARDED_BY(this->mu_);
  std::multiset<uint64_t> start_times_ ABSL_GUARDED_BY(this->mu_);
  size_t current_queue_size_ ABSL_GUARDED_BY(this->mu_) = 0;
  const size_t max_queue_depth_ = 0;
  const bool enable_task_resplit_ = false;
  bool schedulable_batch_ ABSL_GUARDED_BY(this->mu_) = false;
  // The counter of the TraceMe context ids.
  // The higher 32 bits of traceme_context_id_counter_ are set to the creation
  // time of the queue. This prevents batches in different queues from having
  // the same traceme_context_id.
  uint64_t traceme_context_id_counter_ ABSL_GUARDED_BY(this->mu_) =
      (absl::GetCurrentTimeNanos() & 0xFFFFFFFF) << 32;
};

template <typename TaskType>
PriorityAwareQueue<TaskType>::PriorityAwareQueue(
    const typename SharedBatchScheduler<TaskType>::QueueOptions& options,
    tsl::Env* env, ProcessBatchCallback process_batch_callback,
    SchedulableBatchCallback schedulable_batch_callback)
    : QueueBase<TaskType>(options, env, process_batch_callback,
                          schedulable_batch_callback),
      max_queue_depth_(
          options.priority_aware_batching_options.max_queue_depth > 0
              ? options.priority_aware_batching_options.max_queue_depth
              : options.max_enqueued_batches *
                    this->max_execution_batch_size()),
      enable_task_resplit_(
          options.priority_aware_batching_options.enable_task_resplit) {}

template <typename TaskType>
PriorityAwareQueue<TaskType>::~PriorityAwareQueue() {
  absl::MutexLock l(this->mu_);
  DCHECK(this->IsEmptyInternal());
}

template <typename TaskType>
absl::Status PriorityAwareQueue<TaskType>::ScheduleTaskInternal(
    std::unique_ptr<TaskType>* task, bool* notify_schedulable) {
  if ((*task)->size() > this->options_.input_batch_size_limit) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Task size %d is larger than maximum input batch size %d",
        (*task)->size(), this->options_.input_batch_size_limit));
  }

  TF_RETURN_IF_ERROR(AddTask(task, this->env_->NowMicros()));

  if (!schedulable_batch_ && IsOpenBatchSchedulable()) {
    schedulable_batch_ = true;
    *notify_schedulable = true;
  }
  return absl::OkStatus();
}

template <typename TaskType>
size_t PriorityAwareQueue<TaskType>::NumEnqueuedTasksInternal() const {
  return tasks_.size();
}

template <typename TaskType>
size_t PriorityAwareQueue<TaskType>::SchedulingCapacityInternal() const {
  if (current_queue_size_ >= max_queue_depth_) return 0;
  return max_queue_depth_ - current_queue_size_;
}

// Returns a batch of tasks from the queue if the batch is ready to be executed.
// Otherwise, returns nullptr.
// BatchPaddingPolicy is applied to determine the optimal batch size.
template <typename TaskType>
std::unique_ptr<Batch<TaskType>>
PriorityAwareQueue<TaskType>::ScheduleBatchInternal() {
  if (!IsSchedulable()) {
    schedulable_batch_ = false;
    return nullptr;
  }

  size_t candidate_size =
      std::min(current_queue_size_, this->max_execution_batch_size());
  int32_t tasks_to_schedule = ApplyBatchPaddingPolicy(
      candidate_size, this->options_.allowed_batch_sizes,
      this->options_.disable_padding, this->options_.batch_padding_policy,
      this->options_.model_batch_stats);
  std::vector<QueueEntry> tasks = RemoveTask(tasks_to_schedule);

  // If batching down yielded no tasks (e.g. the head task is unsplittable and
  // larger than tasks_to_schedule), fall back to initial candidate_size to
  // avoid livelock.
  if (tasks.empty() && tasks_to_schedule < candidate_size) {
    tasks = RemoveTask(candidate_size);
  }

  // If no tasks could be dequeued, return nullptr to avoid scheduling an empty
  // batch.
  if (tasks.empty()) {
    schedulable_batch_ = false;
    return nullptr;
  }

  auto batch = std::make_unique<Batch<TaskType>>(++traceme_context_id_counter_);
  for (auto& t : tasks) {
    batch->AddTask(std::move(t.task), t.start_time_micros);
  }
  batch->Close();

  // Update schedulable_batch_ based on whether remaining tasks in the queue can
  // form another batch.
  schedulable_batch_ = IsOpenBatchSchedulable();

  return batch;
}

template <typename TaskType>
std::optional<typename QueueBase<TaskType>::BatchPriorityKey>
PriorityAwareQueue<TaskType>::PeekBatchPriorityInternal() const {
  if (!IsSchedulable()) {
    return std::nullopt;
  }
  std::optional<tsl::criticality::Criticality> highest_criticality =
      HighestCriticality();
  if (!highest_criticality.has_value()) {
    return std::nullopt;
  }
  const int max_criticality =
      static_cast<int>(tsl::criticality::Criticality::kCriticalPlus);
  // Priority is max_criticality - highest_criticality, so kCriticalPlus (3) ->
  // 0, kCritical (2) -> 1, kSheddablePlus (1) -> 2, kSheddable (0) -> 3.
  // Lower value means higher priority.
  return std::make_pair(
      max_criticality - static_cast<int>(highest_criticality.value()),
      EarliestHighPriorityTaskStartTime().value());
}

template <typename TaskType>
bool PriorityAwareQueue<TaskType>::IsEmptyTaskQueueInternal() const {
  return tasks_.empty();
}

template <typename TaskType>
bool PriorityAwareQueue<TaskType>::CanSplitTask(const TaskType& task) const {
  if constexpr (std::is_base_of_v<BatchTask, TaskType>) {
    // If a BatchTask is a subtask, we can split it if task resplitting and
    // large batch splitting are enabled.
    if (task.is_subtask()) {
      return enable_task_resplit_ &&
             this->options_.enable_large_batch_splitting;
    }
  }
  return this->options_.enable_large_batch_splitting;
}

template <typename TaskType>
absl::Status PriorityAwareQueue<TaskType>::AddTask(
    std::unique_ptr<TaskType>* task, uint64_t start_time_micros) {
  QueueEntry new_task_entry;
  new_task_entry.criticality = GetCriticality(**task);
  new_task_entry.start_time_micros = start_time_micros;

  while (current_queue_size_ + (*task)->size() > max_queue_depth_ &&
         !tasks_.empty()) {
    // Evict if necessary
    auto it = std::prev(tasks_.end());
    const QueueEntry& lowest_priority_entry = *it;

    // Would new task be sorted into the queue before the lowest priority
    // entry?
    if (new_task_entry < lowest_priority_entry) {
      // Evict lower-priority tasks to make room for a higher-priority task,
      // even if the higher-priority task may end up being rejected. This is
      // necessary to prevent priority inversion, where a low-priority task
      // could occupy queue space and prevent a high-priority task from being
      // admitted.
      QueueEntry evicted_entry = RemoveEntryInternal(it);
      if constexpr (std::is_base_of_v<BatchTask, TaskType>) {
        // Terminate the evicted task. FinishTask restores the evicted task's
        // context so the error is reported in its original context rather than
        // under the incoming higher-priority task's context.
        evicted_entry.task->FinishTask(
            absl::UnavailableError("Task evicted due to priority queue full."));
      }
    } else {
      return absl::UnavailableError(
          "The priority queue to which this task was submitted is full; "
          "and the task priority is too low to evict other tasks.");
    }
  }

  if (current_queue_size_ + (*task)->size() > max_queue_depth_) {
    return absl::UnavailableError(absl::StrFormat(
        "The priority queue to which this task was submitted is full; "
        "task size %zu is larger than the max queue depth %zu.",
        (*task)->size(), max_queue_depth_));
  }

  new_task_entry.task = std::move(*task);
  AddEntryInternal(std::move(new_task_entry));
  return absl::OkStatus();
}

template <typename TaskType>
std::vector<typename PriorityAwareQueue<TaskType>::QueueEntry>
PriorityAwareQueue<TaskType>::RemoveTask(int size) {
  std::vector<QueueEntry> tasks_to_schedule;
  int remaining_size = size;

  while (remaining_size > 0 && !tasks_.empty()) {
    auto it = tasks_.begin();

    // If task fits just add it to tasks_to_schedule.
    if (it->task->size() <= remaining_size) {
      QueueEntry highest_priority_entry = RemoveEntryInternal(it);
      remaining_size -= highest_priority_entry.task->size();
      tasks_to_schedule.push_back(std::move(highest_priority_entry));
      continue;
    }

    // If task does not fit, check if we can split it.
    if (CanSplitTask(*(it->task))) {
      QueueEntry highest_priority_entry = RemoveEntryInternal(it);
      std::unique_ptr<TaskType> highest_priority_task_ptr =
          std::move(highest_priority_entry.task);
      std::vector<std::unique_ptr<TaskType>> output_tasks;
      absl::Status status = this->options_.split_input_task_func(
          &highest_priority_task_ptr, remaining_size,
          this->max_execution_batch_size(), &output_tasks);

      if (!status.ok()) {
        LOG(ERROR) << "Failed to split task: " << status;
        highest_priority_entry.task = std::move(highest_priority_task_ptr);
        AddEntryInternal(std::move(highest_priority_entry));
        break;
      }

      // First subtask fits.
      remaining_size -= output_tasks[0]->size();
      QueueEntry first_split_entry;
      first_split_entry.task = std::move(output_tasks[0]);
      first_split_entry.start_time_micros =
          highest_priority_entry.start_time_micros;
      first_split_entry.criticality = highest_priority_entry.criticality;
      tasks_to_schedule.push_back(std::move(first_split_entry));

      for (size_t i = 1; i < output_tasks.size(); ++i) {
        QueueEntry new_entry;
        new_entry.task = std::move(output_tasks[i]);
        new_entry.start_time_micros = highest_priority_entry.start_time_micros;
        new_entry.criticality = highest_priority_entry.criticality;
        AddEntryInternal(std::move(new_entry));
      }
    }

    // The task does not fit and we cannot split it. Leave it for the next
    // batch.
    break;
  }
  return tasks_to_schedule;
}

template <typename TaskType>
void PriorityAwareQueue<TaskType>::AddEntryInternal(QueueEntry entry) {
  current_queue_size_ += entry.task->size();
  start_times_.insert(entry.start_time_micros);
  tasks_.insert(std::move(entry));
}

template <typename TaskType>
typename PriorityAwareQueue<TaskType>::QueueEntry
PriorityAwareQueue<TaskType>::RemoveEntryInternal(
    typename std::multiset<QueueEntry>::iterator it) {
  auto node = tasks_.extract(it);
  QueueEntry& entry = node.value();
  current_queue_size_ -= entry.task->size();
  auto start_time_it = start_times_.find(entry.start_time_micros);
  if (start_time_it != start_times_.end()) {
    start_times_.erase(start_time_it);
  }
  return std::move(entry);
}

template <typename TaskType>
std::optional<uint64_t> PriorityAwareQueue<TaskType>::EarliestTaskStartTime()
    const {
  if (start_times_.empty()) {
    return std::nullopt;
  }
  return *start_times_.begin();
}

template <typename TaskType>
std::optional<uint64_t>
PriorityAwareQueue<TaskType>::EarliestHighPriorityTaskStartTime() const {
  if (tasks_.empty()) {
    return std::nullopt;
  }
  return tasks_.begin()->start_time_micros;
}

template <typename TaskType>
std::optional<tsl::criticality::Criticality>
PriorityAwareQueue<TaskType>::HighestCriticality() const {
  if (tasks_.empty()) {
    return std::nullopt;
  }
  return tasks_.begin()->criticality;
}

template <typename TaskType>
tsl::criticality::Criticality PriorityAwareQueue<TaskType>::GetCriticality(
    const TaskType& task) const {
  if constexpr (std::is_base_of_v<BatchTask, TaskType>) {
    return task.criticality();
  }
  return tsl::criticality::Criticality::kCritical;
}

template <typename TaskType>
bool PriorityAwareQueue<TaskType>::IsSchedulable() const {
  if (tasks_.empty()) return false;
  return this->closed_ ||
         current_queue_size_ >= this->max_execution_batch_size() ||
         this->env_->NowMicros() >= EarliestTaskStartTime().value() +
                                        this->options_.batch_timeout_micros;
}

template <typename TaskType>
bool PriorityAwareQueue<TaskType>::IsOpenBatchSchedulable() const {
  return PeekBatchPriorityInternal().has_value();
}

}  // namespace internal
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_BATCHING_PRIORITY_AWARE_QUEUE_H_

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

#ifndef XLA_SERVING_RUNTIME_CORE_BATCHING_QUEUE_OPTIONS_H_
#define XLA_SERVING_RUNTIME_CORE_BATCHING_QUEUE_OPTIONS_H_

#include <stddef.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/batch_stats.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving

namespace xla_serving {

// Scheduling and configuration options for a task queue in a batch scheduler
// (e.g. SharedBatchScheduler). Each queue has its own set of scheduling
// options governing input batch size limits, execution batch sizes, timeouts,
// queue depth, large batch splitting, padding, and priority policies.
template <typename TaskType>
struct QueueOptions {
  // The size limit of an input batch to the queue.
  //
  // If `enable_large_batch_splitting` is True, 'input_batch_size_limit'
  // should be greater or equal than `max_execution_batch_size`; otherwise
  // `input_batch_size_limit` should be equal to `max_execution_batch_size`.
  size_t input_batch_size_limit = 1000;

  // If a task has been enqueued for this amount of time (in microseconds),
  // and a thread is available, the scheduler will immediately form a batch
  // from enqueued tasks and assign the batch to the thread for processing,
  // even if the batch's size is below 'input_batch_size_limit'.
  //
  // This parameter offers a way to bound queue latency, so that a task isn't
  // stuck in the queue indefinitely waiting for enough tasks to arrive to
  // make a full batch. (The latency bound is given in the SharedBatchScheduler
  // class documentation.)
  //
  // The goal is to smooth out batch sizes under low request rates, and thus
  // avoid latency spikes.
  int64_t batch_timeout_micros = 0;

  // The maximum allowable number of enqueued (accepted by Schedule() but
  // not yet being processed on a batch thread) tasks in terms of batches.
  // If this limit is reached, Schedule() will return an UNAVAILABLE error.
  // See the SharedBatchScheduler class documentation for guidelines on how to
  // tune this parameter.
  //
  // Must be positive, or else invalid argument error will be returned at
  // queue creation time.
  size_t max_enqueued_batches = 10;

  // If true, queue implementation would split one input batch task into
  // subtasks (as specified by `split_input_task_func` below) and fit subtasks
  // into different batches.
  //
  // For usage of `split_input_task_func`, please see its comment.
  bool enable_large_batch_splitting = false;

  // `input_task`: a unit of task to be split.
  // `first_output_task_size`: task size of first output.
  // `max_execution_batch_size`: Maximum size of each batch.
  // `output_tasks`: A list of output tasks after split.
  //
  // REQUIRED:
  // 1) All `output_tasks` should be non-empty tasks.
  // 2) Sizes of `output_tasks` add up to size of `input_task`.
  //
  // NOTE:
  // Instantiations of `TaskType` may vary, so it's up to caller to define
  // how (e.g., which members to access) to split input tasks.
  std::function<absl::Status(
      std::unique_ptr<TaskType>* input_task, int first_output_task_size,
      int input_batch_size_limit,
      std::vector<std::unique_ptr<TaskType>>* output_tasks)>
      split_input_task_func;

  // The maximum size of each enqueued batch to be processed.
  //
  // The scheduler may form batches of any size between 1 and this number
  // (inclusive). If there is a need to quantize the batch sizes, i.e. only
  // submit batches whose size is in a small set of allowed sizes, that can be
  // done by adding padding in the process-batch callback.
  size_t max_execution_batch_size = 1000;

  // If non-empty, contains configured batch sizes.
  std::vector<int32_t> allowed_batch_sizes;

  // If true, the padding will not be appended.
  bool disable_padding = false;

  // If true, variable-length inputs across non-batch dimensions (dim >= 1) are
  // padded to the maximum shape within the batch before concatenation.
  bool pad_variable_length_inputs = false;

  // The padding policy to use.
  //
  // See the documentation for kPadUpPolicy for details.
  std::string batch_padding_policy = std::string(kPadUpPolicy);

  // A pointer to a ModelBatchStats instance for this model. To be used for
  // cost-based padding policy selection.
  //
  // If null, some other padding policy will be used if a cost-based one is
  // requested.
  ModelBatchStats* model_batch_stats = nullptr;

  // If true, queue implementation would split high priority and low priority
  // inputs into two sub queues.
  bool enable_priority_queue = false;

  // A separate set of queue options for different priority inputs.
  // Use iff `enable_priority_queue` is true.
  struct PriorityQueueOptions {
    // See QueueOptions.max_execution_batch_size
    size_t max_execution_batch_size = 0;
    // See QueueOptions.batch_timeout_micros
    int64_t batch_timeout_micros = 0;
    // See QueueOptions.input_batch_size_limit
    size_t input_batch_size_limit = 0;
    // See QueueOptions.max_enqueued_batches
    size_t max_enqueued_batches = 0;
    // See QueueOptions.allowed_batch_sizes
    std::vector<int32_t> allowed_batch_sizes;
  };
  // A subset of queue options for high priority input. These options are
  // currently not being used in favor of the equivalent options at the
  // QueueOptions level.
  PriorityQueueOptions high_priority_queue_options;
  // A subset of queue options for low priority input.
  PriorityQueueOptions low_priority_queue_options;

  // A policy that determines the mixed priority batching behavior. It is
  // effective only when enable_priority_queue is true.
  MixedPriorityBatchingPolicy mixed_priority_batching_policy =
      MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;

  // Controls the internal queue batching strategy.
  PriorityBatchingPolicy priority_batching_policy =
      PriorityBatchingPolicy::kStrictFifo;

  // Options for priority aware batching.
  // Used iff PriorityBatchingPolicy is set to PRIORITY_AWARE.
  PriorityAwareBatchingOptions priority_aware_batching_options;

  // The allocator for allocating memory for HostBuffer. The default allocator
  // is DefaultAllocator. It can be overridden with a custom allocator.
  Allocator* allocator = DefaultAllocator::GetInstance();
};

// Computes the max_execution_batch_size of a queue based on its options.
template <typename TaskType>
size_t GetMaxExecutionBatchSize(const QueueOptions<TaskType>& options) {
  return options.enable_large_batch_splitting ? options.max_execution_batch_size
                                              : options.input_batch_size_limit;
}

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_BATCHING_QUEUE_OPTIONS_H_

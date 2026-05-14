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

#include "runtime/kernel/batching/batch_resource_base.h"  // from @neptune

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <iterator>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/cleanup/cleanup.h"  // from @abseil-cpp
#include "absl/container/fixed_array.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/functional/bind_front.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/blocking_counter.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler_utils.h"  // from @neptune
#include "runtime/core/batching/batch_stats.h"  // from @neptune
#include "runtime/core/batching/input_split_metadata.h"  // from @neptune
#include "runtime/core/batching/threadsafe_status.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/incremental_barrier.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/batching/batch_options_utils.h"  // from @neptune
#include "runtime/kernel/batching/warmup.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/lib/monitoring/sampler.h"  // from @xla
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/env_time.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
// TODO: add Copybara notation for dependencies on
// "third_party/tensorflow/core/common_runtime" as they are used for cost
// measurement and it depends on usage of registration pattern with internal
// libraries. OSS users do not need this.
#include "tensorflow/core/common_runtime/cost_constants.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/cost_measurement.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/cost_util.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/request_cost.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/request_cost_accessor.h"  // from @org_tensorflow
#include "tsl/platform/context.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {
namespace {

// Move this to a util.
std::vector<double> GetPaddingSizeBucketLimits();

static auto* executed_batch_size_counter = tsl::monitoring::Counter<3>::New(
    "/orbax/serving/batching/executed_batch_size",
    "Tracks the batch size on processing by model_name.", "model_name",
    "model_version", "batch_size");

static auto* padding_size_sampler = tsl::monitoring::Sampler<3>::New(
    {"/orbax/serving/batching/padding_sizes",
     "Tracks the padding size distribution on batches by model_name (if "
     "available).",
     "model_name", "model_version", "execution_batch_size"},
    tsl::monitoring::Buckets::Explicit(GetPaddingSizeBucketLimits()));

static auto* batch_costs_sampler = tsl::monitoring::Sampler<4>::New(
    {"/orbax/serving/batching/costs",
     "Tracks the batch costs (in microseconds) by model name and processed "
     "size.",
     "model_name", "model_version", "processed_size", "cost_type"},
    // It's 27 buckets with the last bucket being 2^26 to DBL_MAX;
    // so the limits are [1, 2, 4, 8, ..., 64 * 1024 * 1024 (~64s), DBL_MAX].
    tsl::monitoring::Buckets::Exponential(1, 2, 27));

static auto* request_counter = tsl::monitoring::Counter<8>::New(
    "/orbax/serving/batching/request_count",
    "Counts input requests by batching configuration: model_name, "
    "model_version, use_global_scheduler, rank_queues, "
    "mixed_priority_batching_policy, enable_priority_queue, "
    "batch_padding_policy and request criticality.",
    "model_name", "model_version", "use_global_scheduler", "rank_queues",
    "mixed_priority_batching_policy", "enable_priority_queue",
    "batch_padding_policy", "criticality");

void RecordExecutedBatchSize(absl::string_view model_name,
                             int64_t model_version, int32_t batch_size) {
  VLOG(2) << "RecordExecutedBatchSize: model_name: " << model_name
          << ", model_version: " << model_version
          << ", batch_size: " << batch_size;
  executed_batch_size_counter
      ->GetCell(model_name, absl::StrCat(model_version),
                absl::StrCat(batch_size))
      ->IncrementBy(1);
}

std::vector<double> GetPaddingSizeBucketLimits() {
  static absl::NoDestructor<std::vector<double>> bucket_limits([]() {
    std::vector<double> bucket_limits;
    // Bucket containing 0 has bounds [-2/3, 2/3).
    // Remaining buckets are centered at powers of 2 and have bounds:
    // [(2/3) * 2^i, (4/3) * 2^i) for i = 1, ..., 13.
    // Largest bucket has range: [(2/3) *  2^14, DBL_MAX]
    bucket_limits.push_back(-2.0 / 3.0);
    // populate rest of bounds
    double bound = 2.0 / 3.0;
    double growth_factor = 2;
    for (int i = 0; i < 16; i++) {
      bucket_limits.push_back(bound);
      bound *= growth_factor;
    }
    return bucket_limits;
  }());
  return *bucket_limits;
}

void RecordPaddingSize(int32_t padding_size, absl::string_view model_name,
                       int64_t model_version, int32_t execution_batch_size) {
  VLOG(2) << "RecordPaddingSize: padding_size: " << padding_size
          << ", model_name: " << model_name
          << ", model_version: " << model_version
          << ", execution_batch_size: " << execution_batch_size;
  padding_size_sampler
      ->GetCell(model_name, absl::StrCat(model_version),
                absl::StrCat(execution_batch_size))
      ->Add(static_cast<double>(padding_size));
}

void RecordBatchCosts(const std::string& model_name, int64_t model_version,
                      const int64_t processed_size,
                      const absl::string_view cost_type,
                      const absl::Duration total_cost) {
  VLOG(2) << "RecordBatchCosts: model_name: " << model_name
          << ", model_version: " << model_version
          << ", processed_size: " << processed_size
          << ", cost_type: " << cost_type << ", total_cost: " << total_cost;
  batch_costs_sampler
      ->GetCell(model_name, absl::StrCat(model_version),
                std::to_string(processed_size), cost_type)
      ->Add(absl::ToDoubleMicroseconds(total_cost));
}

void RecordInputBatchConfig(
    absl::string_view model_name, int64_t model_version,
    bool use_global_scheduler, bool rank_queues,
    MixedPriorityBatchingPolicy mixed_priority_batching_policy,
    bool enable_priority_queue, absl::string_view batch_padding_policy,
    tsl::criticality::Criticality criticality) {
  request_counter
      ->GetCell(std::string(model_name), absl::StrCat(model_version),
                use_global_scheduler ? "true" : "false",
                rank_queues ? "true" : "false",
                std::string(MixedPriorityBatchingPolicyToString(
                    mixed_priority_batching_policy)),
                enable_priority_queue ? "true" : "false",
                std::string(batch_padding_policy),
                std::string(CriticalityToString(criticality)))
      ->IncrementBy(1);
}

// Returns the sum of the task sizes. The caller must guarantee that the
// unique_ptrs in the argument vectors are not null.
int GetTotalTaskSize(
    absl::Span<const std::unique_ptr<BatchResourceBase::BatchTask>> tasks) {
  int tasks_size = 0;
  for (const auto& task : tasks) {
    tasks_size += task->size();
  }
  return tasks_size;
}

}  // namespace

std::unique_ptr<BatchResourceBase::BatchTask>
BatchResourceBase::BatchTask::CreateSplitTask(
    int split_index, absl::AnyInvocable<void(absl::Status)> done_callback) {
  std::unique_ptr<BatchTask> task = CreateDerivedTask();

  task->guid = this->guid;
  task->propagated_context = tsl::Context(tsl::ContextKind::kThread);
  task->inputs.reserve(this->inputs.size());
  task->captured_inputs = this->captured_inputs;
  task->context = this->context;
  task->num_outputs = this->num_outputs;
  task->done_callback = std::move(done_callback);
  task->split_index = split_index;
  task->output = this->output;
  task->status = this->status;
  task->is_partial = true;
  task->start_time = this->start_time;
  task->request_cost = this->request_cost;
  task->forced_warmup_batch_size = this->forced_warmup_batch_size;

  return task;
}

absl::Span<const Value> BatchResourceBase::GetInputsToBatch(
    Kernel::ExecuteContext* context) const {
  return context->inputs;
}

using BufferMatrix = std::vector<std::vector<HostBuffer>>;

absl::Status BatchResourceBase::RegisterInputsForAllBatchSizes(
    int64_t guid, Kernel::ExecuteContext* context, int num_outputs,
    const std::string& batcher_queue_name,
    const CreateBatchTaskFn& create_batch_task_fn,
    absl::AnyInvocable<void(absl::Status)> done_callback) {
  // Update the creator to inject the `shared_status` to the created task.
  // `shared_status` is shared across all the tasks (dummy warmup batches and
  // original batch) in this invocation.
  auto shared_task_status = std::make_shared<ThreadSafeStatus>();
  const CreateBatchTaskFn new_create_batch_task_fn =
      [&]() -> absl::StatusOr<std::unique_ptr<BatchTask>> {
    TF_ASSIGN_OR_RETURN(std::unique_ptr<BatchTask> batch_task,
                        create_batch_task_fn());
    batch_task->status = shared_task_status;
    return batch_task;
  };
  auto pending_warmup_counter =
      std::make_shared<absl::BlockingCounter>(allowed_batch_sizes_.size());
  // Enqueue dummy warmup batches.
  for (const auto batch_size : allowed_batch_sizes_) {
    TF_RETURN_IF_ERROR(RegisterInput(
        guid, context, num_outputs, batcher_queue_name,
        new_create_batch_task_fn,
        /*done_callback=*/
        [pending_warmup_counter](absl::Status status) {
          pending_warmup_counter->DecrementCount();
        },
        /*forced_warmup_batch_size=*/batch_size));
  }
  // Enqueue the original batch.
  return RegisterInput(
      guid, context, num_outputs, batcher_queue_name, new_create_batch_task_fn,
      /*done_callback=*/
      [pending_warmup_counter,
       done_callback = std::move(done_callback)](absl::Status status) mutable {
        pending_warmup_counter->Wait();
        done_callback(status);
      });
}

absl::Status BatchResourceBase::RegisterInput(
    int64_t guid, Kernel::ExecuteContext* context, int num_outputs,
    const std::string& batcher_queue_name,
    const CreateBatchTaskFn& create_batch_task_fn,
    absl::AnyInvocable<void(absl::Status)> done_callback,
    int forced_warmup_batch_size) {
  TF_ASSIGN_OR_RETURN(std::unique_ptr<BatchTask> batch_components,
                      create_batch_task_fn());
  batch_components->start_time = tsl::EnvTime::NowNanos();
  batch_components->guid = guid;
  batch_components->propagated_context =
      tsl::Context(tsl::ContextKind::kThread);

  absl::Span<const Value> inputs_for_batch = context->inputs;
  batch_components->inputs.reserve(inputs_for_batch.size());
  for (int i = 0; i < inputs_for_batch.size(); ++i) {
    const Value& value = inputs_for_batch[i];
    if (!value.IsType<HostBuffer>()) {
      return absl::InvalidArgumentError(
          "inputs_for_batch is not a HostBuffer.");
    }
    const HostBuffer& host_buffer = value.Get<HostBuffer>();
    if (host_buffer.shape().dims().empty()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Batching input buffers must have at least one "
                       "dimension. Input at index ",
                       i, " has no dimensions."));
    }
    if (inputs_for_batch.size() >= 2 &&
        host_buffer.shape().dims()[0] !=
            inputs_for_batch[0].Get<HostBuffer>().shape().dims()[0]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Batching input bufferss supplied in a given op invocation must "
          "have equal 0th-dimension size. The first input has 0th-dimension "
          "size ",
          inputs_for_batch[0].Get<HostBuffer>().shape().dims()[0],
          ", but the input at index ", i, " has 0th-dimension size ",
          host_buffer.shape().dims()[0]));
    }
    batch_components->inputs.push_back(host_buffer);
  }

  RecordInputBatchConfig(
      context->metadata.model_name, context->metadata.model_version,
      batcher_->use_global_scheduler(), batcher_->rank_queues(),
      batcher_queue_options_.mixed_priority_batching_policy,
      batcher_queue_options_.enable_priority_queue,
      batcher_queue_options_.batch_padding_policy,
      batch_components->criticality());

  batch_components->context = context;
  batch_components->num_outputs = num_outputs;
  batch_components->split_index = 0;
  batch_components->output = std::make_shared<BufferMatrix>();
  if (!batch_components->status) {
    // A shared status has already been injected if `RegisterWarmupInputs`
    // was called. If not, create the `ThreadSafeStatus` and tie the setting
    // of the kernel context's status to this shared status.
    batch_components->status = std::make_shared<ThreadSafeStatus>();
    batch_components->done_callback = std::move(done_callback);
  } else {
    // Otherwise `RegisterWarmupInputs` was called and already setup the
    // `done_callback` and `status` correctly for this `BatchTask`.
    batch_components->done_callback = std::move(done_callback);
  }
  batch_components->forced_warmup_batch_size = forced_warmup_batch_size;

  std::unique_ptr<tensorflow::RequestCostAccessor> request_cost_accessor =
      tensorflow::CreateRequestCostAccessor();
  if (request_cost_accessor) {
    batch_components->request_cost = request_cost_accessor->GetRequestCost();
  }

  BatcherQueueT* batcher_queue;
  TF_RETURN_IF_ERROR(LookupOrCreateBatcherQueue(
      /* queue_name= */ batcher_queue_name, /* queue= */ &batcher_queue,
      std::string(context->metadata.model_name),
      context->metadata.model_version));
  // Sequentialize warmup batches scheduling.
  // TODO: b/405168505: Maybe move the sequentialization logic to the
  // `WarmupStateRegistry` for centralized control, safer and clearer
  // synchronization, and code sharing.
  // TODO: b/405168505: Checking model name existence is copied form TF, and is
  // kept for compatibility. Maybe remove it.
  if (!context->metadata.model_name.empty()) {
    // TODO: Update the serving path to use the JSV warmup
    // registry in case of Orbax models.
    WarmupStateRegistry::Key key(std::string(context->metadata.model_name),
                                 context->metadata.model_version);
    absl::MutexLock lock(outstanding_batch_mu_);
    if (GetGlobalWarmupStateRegistry().Lookup(key) != nullptr) {
      outstanding_batch_mu_.Await({+[](size_t* outstanding_batch_total_size) {
                                     return *outstanding_batch_total_size == 0;
                                   },
                                   &outstanding_batch_total_size_});
    }
    // TODO: b/405168505: Inheriting the TF behavior, but maybe only touch
    // `outstanding_batch_total_size_` during warmup phase.
    outstanding_batch_total_size_ += batch_components->size();
  }

  return batcher_queue->Schedule(&batch_components);
}

/*static*/ BatchResourceBase::BatcherT::QueueOptions
BatchResourceBase::GetBatcherQueueOptions(
    int32_t num_batch_threads, int32_t max_batch_size,
    int32_t batch_timeout_micros, int32_t max_enqueued_batches,
    const std::vector<int32_t>& allowed_batch_sizes,
    bool enable_large_batch_splitting, bool disable_padding,
    absl::string_view batch_padding_policy, int32_t low_priority_max_batch_size,
    int32_t low_priority_batch_timeout_micros,
    int32_t low_priority_max_enqueued_batches,
    const std::vector<int32_t>& low_priority_allowed_batch_sizes,
    MixedPriorityBatchingPolicy mixed_priority_batching_policy,
    tsl::thread::ThreadPool* thread_pool, Allocator* allocator) {
  BatcherT::QueueOptions batcher_queue_options;
  batcher_queue_options.input_batch_size_limit = max_batch_size;
  batcher_queue_options.max_enqueued_batches = max_enqueued_batches;
  batcher_queue_options.batch_timeout_micros = batch_timeout_micros;
  if (low_priority_max_batch_size > 0) {
    batcher_queue_options.enable_priority_queue = true;
  }
  batcher_queue_options.high_priority_queue_options.input_batch_size_limit =
      max_batch_size;
  batcher_queue_options.high_priority_queue_options.max_enqueued_batches =
      max_enqueued_batches;
  batcher_queue_options.high_priority_queue_options.batch_timeout_micros =
      batch_timeout_micros;
  batcher_queue_options.low_priority_queue_options.input_batch_size_limit =
      low_priority_max_batch_size;
  batcher_queue_options.low_priority_queue_options.max_enqueued_batches =
      low_priority_max_enqueued_batches;
  batcher_queue_options.low_priority_queue_options.batch_timeout_micros =
      low_priority_batch_timeout_micros;
  if (low_priority_allowed_batch_sizes.empty()) {
    batcher_queue_options.low_priority_queue_options.max_execution_batch_size =
        low_priority_max_batch_size;
  } else {
    batcher_queue_options.low_priority_queue_options.max_execution_batch_size =
        *low_priority_allowed_batch_sizes.rbegin();
  }
  batcher_queue_options.low_priority_queue_options.allowed_batch_sizes =
      low_priority_allowed_batch_sizes;
  batcher_queue_options.mixed_priority_batching_policy =
      mixed_priority_batching_policy;
  batcher_queue_options.enable_large_batch_splitting =
      enable_large_batch_splitting;
  batcher_queue_options.allocator = allocator;
  if (enable_large_batch_splitting) {
    batcher_queue_options.split_input_task_func =
        [thread_pool = thread_pool, allocator = allocator](
            std::unique_ptr<BatchTask>* input_task,
            int open_batch_remaining_slot, int max_batch_size,
            std::vector<std::unique_ptr<BatchTask>>* output_tasks)
        -> absl::Status {
      return SplitInputTask(input_task, open_batch_remaining_slot,
                            max_batch_size, thread_pool, output_tasks,
                            allocator);
    };
  }
  if (allowed_batch_sizes.empty()) {
    batcher_queue_options.max_execution_batch_size = max_batch_size;
    batcher_queue_options.high_priority_queue_options.max_execution_batch_size =
        max_batch_size;
  } else {
    batcher_queue_options.max_execution_batch_size =
        *allowed_batch_sizes.rbegin();
    batcher_queue_options.high_priority_queue_options.max_execution_batch_size =
        *allowed_batch_sizes.rbegin();
    batcher_queue_options.allowed_batch_sizes = allowed_batch_sizes;
  }

  batcher_queue_options.disable_padding = disable_padding;
  batcher_queue_options.batch_padding_policy = batch_padding_policy;

  return batcher_queue_options;
}

/*static*/ absl::Status BatchResourceBase::ValidateBatch(const BatchT& batch) {
  for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
    const BatchResourceBase::BatchTask& task = batch.task(task_idx);

    if (task.inputs.size() != batch.task(0).inputs.size()) {
      return absl::InvalidArgumentError(
          "Batching inputs must have equal number of edges");
    }
  }

  return absl::OkStatus();
}

bool BatchResourceBase::IsLowPriorityBatch(const BatchT& batch) const {
  if (!batcher_queue_options_.enable_priority_queue) return false;
  if (batch.empty()) return false;

  // TODO: Once the criticality and priority become configurable,
  // this should rely on the batch parameters instead of the hard coded value.
  return batch.task(0).criticality() ==
             tsl::criticality::Criticality::kSheddablePlus ||
         batch.task(0).criticality() ==
             tsl::criticality::Criticality::kSheddable;
}

// Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
// or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
// returns 'batch_size'.
int BatchResourceBase::RoundToLowestAllowedBatchSize(
    int batch_size, bool is_low_priority_batch) const {
  const std::vector<int32_t>& allowed_batch_sizes =
      is_low_priority_batch ? batcher_queue_options_.low_priority_queue_options
                                  .allowed_batch_sizes
                            : allowed_batch_sizes_;
  return GetNextAllowedBatchSize(batch_size, allowed_batch_sizes,
                                 batcher_queue_options_.disable_padding);
}

absl::Status BatchResourceBase::ConcatInputBuffers(
    const BatchT& batch,
    absl::Span<const std::unique_ptr<BatchTask>> unbatched_tasks,
    Kernel::ExecuteContext* context,
    std::vector<HostBuffer>* concatenated_buffers) const {
  if (batch.num_tasks() == 0) {
    return absl::InvalidArgumentError("Empty batch.");
  }

  const bool just_for_warmup = batch.task(0).forced_warmup_batch_size > 0;
  int all_tasks_size = batch.size() + GetTotalTaskSize(unbatched_tasks);
  const int padded_batch_size =
      just_for_warmup ? batch.task(0).forced_warmup_batch_size
                      : RoundToLowestAllowedBatchSize(
                            all_tasks_size, IsLowPriorityBatch(batch));
  const int padding_amount =
      just_for_warmup ? padded_batch_size : padded_batch_size - all_tasks_size;
  tsl::profiler::TraceMe trace_me(
      [padded_batch_size, padding_amount,
       disable_padding = batcher_queue_options_.disable_padding]() {
        return tsl::profiler::TraceMeEncode(
            "ConcatInputBuffers",
            {{"batch_size_after_padding", padded_batch_size},
             {"padding_amount", padding_amount},
             {"disable_padding", disable_padding}});
      });

  // All tasks should have the same number of input edges.
  const int num_inputs = batch.task(0).inputs.size();
  concatenated_buffers->reserve(num_inputs);
  // Process each input one at a time (the typical case has just one). When
  // `just_for_warmup` is true, the real data is not added. Otherwise, the real
  // data is added to the front of each `concatenated_buffer`.
  for (int i = 0; i < num_inputs; ++i) {
    // Concatenate the tasks ith input buffers into a big output buffer.
    std::vector<HostBuffer> to_concatenate;
    if (just_for_warmup) {
      to_concatenate.reserve(padding_amount);
    } else {
      to_concatenate.reserve(batch.num_tasks() + unbatched_tasks.size() +
                             padding_amount);
      for (int task_idx = 0; task_idx < batch.num_tasks(); ++task_idx) {
        to_concatenate.push_back(batch.task(task_idx).inputs.at(i));
      }
      for (int task_idx = 0; task_idx < unbatched_tasks.size(); ++task_idx) {
        to_concatenate.push_back(unbatched_tasks[task_idx]->inputs.at(i));
      }
    }
    // Add padding as needed if padding is allowed. Use the first row of the
    // first task's buffer as the data for padding.
    if (padding_amount != 0) {
      const HostBuffer& padding_source = batch.task(0).inputs.at(i);
      HostBuffer padding;
      if (padding_source.shape().dims()[0] == 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Cannot use an empty buffer with zero rows as padding when "
            "batching. (Input ",
            i, " got shape ", padding_source.shape(), ".)"));
      }
      if (padding_source.shape().dims()[0] == 1) {
        padding = padding_source;
      } else {
        TF_ASSIGN_OR_RETURN(padding,
                            padding_source.SliceAlongFirstDimension(
                                0, 1, batcher_queue_options_.allocator));
      }
      for (int i = 0; i < padding_amount; ++i) {
        to_concatenate.push_back(padding);
      }
    }

    RecordExecutedBatchSize(context->metadata.model_name,
                            context->metadata.model_version, padded_batch_size);

    RecordPaddingSize(padding_amount, context->metadata.model_name,
                      context->metadata.model_version, padded_batch_size);

    TF_ASSIGN_OR_RETURN(
        HostBuffer concatenated_buffer,
        Concat(thread_pool_, to_concatenate, batcher_queue_options_.allocator));
    concatenated_buffers->push_back(concatenated_buffer);
  }
  return absl::OkStatus();
}

absl::Status BatchResourceBase::SplitOutputBuffers(
    absl::Span<const HostBuffer> combined_outputs, BatchT* batch,
    std::vector<std::unique_ptr<BatchTask>>& unbatched_tasks) const {
  DCHECK_GE(batch->num_tasks(), 1);
  if (batch->num_tasks() < 1) {
    return absl::InternalError(absl::StrCat(
        "Batch size expected to be positive; was ", batch->num_tasks()));
  }

  std::vector<int64_t> task_sizes_plus_optional_padding;
  task_sizes_plus_optional_padding.reserve(batch->num_tasks() +
                                           unbatched_tasks.size());
  for (int i = 0; i < batch->num_tasks(); ++i) {
    task_sizes_plus_optional_padding.push_back(batch->task(i).size());
  }
  for (int i = 0; i < unbatched_tasks.size(); ++i) {
    task_sizes_plus_optional_padding.push_back(unbatched_tasks[i]->size());
  }
  int unbatched_tasks_size = GetTotalTaskSize(unbatched_tasks);
  int all_tasks_size = batch->size() + unbatched_tasks_size;
  const int padding_size =
      batcher_queue_options_.disable_padding
          ? 0
          : RoundToLowestAllowedBatchSize(all_tasks_size,
                                          IsLowPriorityBatch(*batch)) -
                all_tasks_size;
  if (padding_size > 0) {
    task_sizes_plus_optional_padding.push_back(padding_size);
  }

  // Split each element of `combined_outputs` according to task sizes
  // within the batch, and use this to populate context outputs.
  for (int i = 0, iter_limit = combined_outputs.size(); i < iter_limit; ++i) {
    const HostBuffer& output_buffer = combined_outputs[i];
    if (output_buffer.shape().dims().empty()) {
      return absl::FailedPreconditionError(
          "Batched output buffer has 0 dimensions");
    }
    int64_t zeroth_dim_output_buffer_size = output_buffer.shape().dims()[0];
    if (zeroth_dim_output_buffer_size !=
        static_cast<int64_t>(all_tasks_size + padding_size)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Batched output buffer's 0th dimension does not equal the sum of "
          "the 0th dimension sizes of the input buffers. "
          "0th dimension size: ",
          zeroth_dim_output_buffer_size, "; batch size: ", batch->size(),
          "; unbatched tasks size: ", unbatched_tasks_size,
          "; padding size: ", padding_size));
    }

    TF_ASSIGN_OR_RETURN(std::vector<HostBuffer> split_buffers,
                        output_buffer.SplitAlongFirstDimension(
                            task_sizes_plus_optional_padding,
                            batcher_queue_options_.allocator));

    DCHECK_EQ(split_buffers.size(), task_sizes_plus_optional_padding.size());
    if (split_buffers.size() != task_sizes_plus_optional_padding.size()) {
      return absl::InternalError(
          absl::StrCat("Buffer split operation did not work as expected; got ",
                       split_buffers.size(), " splits; expected ",
                       task_sizes_plus_optional_padding.size()));
    }

    // Ignore a possible final split_buffers entry containing the padding.
    for (int j = 0; j < batch->num_tasks(); ++j) {
      BatchTask& task = *(batch->mutable_task(j));
      if (task.is_partial) {
        std::vector<HostBuffer>& buffer_vector =
            (*task.output)[task.split_index];
        buffer_vector[i] = std::move(split_buffers[j]);
      } else {
        task.context->outputs.push_back(Value(std::move(split_buffers[j])));
      }
    }
    for (int j = 0; j < unbatched_tasks.size(); ++j) {
      // The unbatched tasks are not split, so no need to handle the partial
      // case separately.
      unbatched_tasks[j]->context->outputs.push_back(
          Value(std::move(split_buffers[batch->num_tasks() + j])));
    }
  }

  return absl::OkStatus();
}

void BatchResourceBase::CleanUpFunctionHelper(
    BatchTask& task, const absl::Status& status) const {
  tsl::WithContext wc(task.propagated_context);
  if (!status.ok()) {
    if (!absl::StrContains(status.message(),
                           "Function was cancelled before it was started")) {
      task.status->Update(status);
    } else {
      // Do not propagate this error; Prefer a more helpful error message.
      LOG(ERROR) << "ERROR!!!! " << status.message();
    }
  }
  task.done_callback(task.status->status());
}

absl::StatusOr<const BatchResourceBase::BatchTask&>
BatchResourceBase::GetLatestDeadlineTask(const BatchT& batch) const {
  const BatchResourceBase::BatchTask* latest_deadline_task = nullptr;
  for (int i = 0; i < batch.num_tasks(); ++i) {
    if (latest_deadline_task == nullptr ||
        batch.task(i).context->run_context.cancellation_context.deadline >
            latest_deadline_task->context->run_context.cancellation_context
                .deadline) {
      latest_deadline_task = &batch.task(i);
    }
    if (latest_deadline_task->context->run_context.cancellation_context
            .deadline == absl::InfiniteFuture()) {
      break;
    }
  }
  if (latest_deadline_task == nullptr) {
    return absl::FailedPreconditionError(
        "nullptr while getting latest deadline task, batch may be empty");
  }
  return *latest_deadline_task;
}

void BatchResourceBase::ProcessFuncBatch(
    std::unique_ptr<BatchT> batch,
    std::vector<std::unique_ptr<BatchTask>> unbatched_tasks) const {
  if (batch->empty()) {
    return;
  }

  // We use the 'propagated_context' from one of the threads which setup one
  // of the tasks. This will propagate any common context over all the threads
  // which are running this Session, of which this BatchOp is a part.
  tsl::WithContext wc(batch->task(batch->num_tasks() - 1).propagated_context);

  const tensorflow::CostMeasurement::Context batching_context{
      /*is_per_query=*/false};
  std::vector<std::unique_ptr<tensorflow::CostMeasurement>>
      batch_cost_measurements =
          tensorflow::CreateCostMeasurements(batching_context);

  // Get the task with the latest deadline. This guarantees that the batch
  // request is only cancelled if all tasks have exceeded their deadlines.
  auto propagated_task_status = GetLatestDeadlineTask(*batch);
  const BatchTask& propagated_task = [&]() -> const BatchTask& {
    if (!propagated_task_status.ok()) {
      // This should never happen due to the empty batch check at the beginning
      // of this function. In case it does, fall back to using the last task.
      LOG(ERROR) << "Error while getting latest deadline task for batch: "
                 << propagated_task_status.status().message();
      return batch->task(batch->num_tasks() - 1);
    } else {
      return *propagated_task_status;
    }
  }();
  Kernel::ExecuteContext* propagated_context = propagated_task.context;
  // Regardless of the outcome, we need to propagate the status to the
  // individual tasks and signal that they are done. We use MakeCleanup() to
  // ensure that this happens no matter how we exit the method below.
  absl::Status status;
  bool cleanup_done = false;
  int64_t processed_size = batch->size();
  auto cleanup_fn = [&](const absl::Status& status) {
    if (cleanup_done) {
      return;
    }
    SplitBatchCostsAndRecordMetrics(
        /* model_name= */ std::string(propagated_context->metadata.model_name),
        /* model_version= */ propagated_context->metadata.model_version,
        batch_cost_measurements, processed_size, *batch);
    // Clear the measurements before unblocking the batch task, as measurements
    // are associated with the task's thread context.
    batch_cost_measurements.clear();
    for (int i = 0; i < batch->num_tasks(); ++i) {
      CleanUpFunctionHelper(*batch->mutable_task(i), status);
    }
    for (int i = 0; i < unbatched_tasks.size(); ++i) {
      CleanUpFunctionHelper(*unbatched_tasks[i], status);
    }
    cleanup_done = true;
  };

  absl::Cleanup finally = ([&cleanup_fn, &status] { cleanup_fn(status); });

  status = ValidateBatch(*batch);
  if (!status.ok()) {
    return;
  }

  std::vector<HostBuffer> concatenated_buffers;
  status = ConcatInputBuffers(*batch, unbatched_tasks, propagated_context,
                              &concatenated_buffers);
  processed_size =
      RoundToLowestAllowedBatchSize(batch->size(), IsLowPriorityBatch(*batch));
  if (!status.ok()) {
    return;
  }

  std::vector<HostBuffer> combined_outputs;
  std::vector<HostBuffer> args(concatenated_buffers.begin(),
                               concatenated_buffers.end());

  // Releases the cleanup method here, because the callback of the function
  // library runtime will handle it now.
  std::move(finally).Cancel();

  ProcessFuncBatchImpl(propagated_task, args, &combined_outputs,
                       [&](const absl::Status& run_status) {
                         absl::Status final_status;
                         absl::Cleanup run_finally = ([&]() {
                           // We do the cleanup here as an optimization, so that
                           // it runs in the underlying TF inter-op threadpool.
                           // Running it in the threadpool, let's the ensuing
                           // ops be scheduled faster, because the executor will
                           // add them to the front of the thread pool's task
                           // queue rather than the end.
                           cleanup_fn(final_status);
                         });
                         final_status = run_status;
                         if (!final_status.ok()) {
                           return;
                         }
                         if (propagated_task.forced_warmup_batch_size == 0) {
                           final_status = SplitOutputBuffers(
                               combined_outputs, batch.get(), unbatched_tasks);
                         }
                       });
}

void BatchResourceBase::ProcessBatchCallBack(
    std::unique_ptr<Batch<BatchTask>> batch,
    std::vector<std::unique_ptr<BatchTask>> unbatched_tasks) {
  if (!batch->task(0).context->metadata.model_name.empty()) {
    absl::MutexLock lock(outstanding_batch_mu_);
    outstanding_batch_total_size_ -= batch->size();
  }
  ProcessFuncBatch(std::move(batch), std::move(unbatched_tasks));
}

absl::Status BatchResourceBase::LookupOrCreateBatcherQueue(
    const std::string& queue_name, BatcherQueueT** queue,
    const std::string& model_name, int64_t model_version) {
  absl::MutexLock l(batcher_queues_mu_);

  auto it = batcher_queues_.find(queue_name);
  if (it != batcher_queues_.end()) {
    *queue = it->second.get();
    return absl::OkStatus();
  }

  std::unique_ptr<BatcherQueueT> new_queue;
  if (batcher_) {
    batcher_queue_options_.model_batch_stats =
        &GlobalBatchStatsRegistry().model(
            /*model_name=*/model_name, /*model_version=*/model_version);
    TF_RETURN_IF_ERROR(batcher_->AddQueue(
        batcher_queue_options_,
        absl::bind_front(&BatchResourceBase::ProcessBatchCallBack, this),
        &new_queue));
  } else {
    return absl::InternalError("No batcher defined.");
  }
  *queue = new_queue.get();
  batcher_queues_[queue_name] = std::move(new_queue);
  return absl::OkStatus();
}

/*static*/ absl::Status BatchResourceBase::SplitInputTask(
    std::unique_ptr<BatchTask>* input_task_ptr, int open_batch_remaining_slot,
    int max_batch_size, tsl::thread::ThreadPool* thread_pool,
    std::vector<std::unique_ptr<BatchTask>>* output_tasks,
    Allocator* allocator) {
  BatchTask& input_task = *(*input_task_ptr);
  const int64_t input_task_size = input_task.size();
  DCHECK_GT(input_task_size, 0);

  std::shared_ptr<ThreadSafeStatus> shared_status = input_task.status;

  // `split_task_done_callback` runs only after all split tasks are
  // complete.
  absl::AnyInvocable<void()> split_task_done_callback =
      [done_callback = std::move(input_task.done_callback),
       output = input_task.output, execute_context = input_task.context,
       num_output = input_task.num_outputs, status = shared_status,
       thread_pool = thread_pool, allocator = allocator]() mutable {
        if (!status->status().ok()) {
          done_callback(status->status());
          return;
        }
        for (int i = 0; i < num_output; ++i) {
          // Concat would memcpy each input buffer to one output buffer.
          // TODO: In this context, Concat can be further optimized
          // to get rid of some (probably all) memcpy when input buffers are
          // slices of another copy.
          std::vector<HostBuffer> to_concatenate;
          to_concatenate.reserve(output->size());
          for (int j = 0; j < output->size(); ++j) {
            to_concatenate.push_back(std::move((*output)[j][i]));
          }
          absl::StatusOr<HostBuffer> output_buffer =
              Concat(thread_pool, to_concatenate, allocator);
          if (!output_buffer.ok()) {
            status->Update(output_buffer.status());
          }
          // Since the `num_outputs` is known, we should consider
          // preallocating the `outputs`
          execute_context->outputs.push_back(Value(std::move(*output_buffer)));
        }
        done_callback(status->status());
      };
  IncrementalBarrier barrier(std::move(split_task_done_callback));

  const internal::InputSplitMetadata input_split_metadata(
      input_task_size, open_batch_remaining_slot, max_batch_size);

  const absl::FixedArray<int>& task_sizes = input_split_metadata.task_sizes();

  const int num_batches = task_sizes.size();
  std::vector<int64_t> output_task_sizes;
  output_task_sizes.resize(num_batches);
  for (int i = 0; i < num_batches; i++) {
    output_task_sizes[i] = task_sizes[i];
  }

  input_task.output->resize(num_batches);
  for (int i = 0; i < num_batches; ++i) {
    (*input_task.output)[i].resize(input_task.num_outputs);
  }

  output_tasks->reserve(num_batches);
  for (int i = 0; i < num_batches; i++) {
    auto callback = barrier.Inc();
    output_tasks->push_back(input_task.CreateSplitTask(
        i, [output_addr = input_task.output, callback = std::move(callback)](
               absl::Status status) mutable { callback(); }));
  }

  const int num_input_buffers = input_task.inputs.size();

  // Splits each input buffer according to `output_task_sizes`, and
  // initializes input of `output_tasks` with split results.
  for (int i = 0; i < num_input_buffers; ++i) {
    const HostBuffer& input_buffer = input_task.inputs[i];

    // TODO: `HostBuffer::SplitAlongFirstDimension` doesn't have
    // parallel support. We should implement a split util that can support
    // parallel split like `Concat`.
    // Optional: Figure out the optimal implementation of Split, by using
    // `HostBuffer::SplitAlongFirstDimension` and eliminating unnecessary memcpy
    // as much as possible.
    absl::StatusOr<std::vector<HostBuffer>> split_buffers =
        input_buffer.SplitAlongFirstDimension(output_task_sizes, allocator);

    if (!split_buffers.ok()) {
      return absl::InternalError(absl::StrCat(
          "When splitting input, HostBuffer split operation failed: ",
          split_buffers.status().message()));
    }
    if (split_buffers->size() != output_task_sizes.size()) {
      return absl::InternalError(absl::StrCat(
          "When splitting input, HostBuffer split operation did not work as "
          "expected; got ",
          split_buffers->size(), " splits; expected ",
          output_task_sizes.size()));
    }
    for (int j = 0; j < output_tasks->size(); ++j) {
      BatchTask& output_task = *((*output_tasks)[j]);
      auto moved_buffer_iter = std::next(split_buffers->begin(), j);
      std::move(moved_buffer_iter, moved_buffer_iter + 1,
                std::back_inserter(output_task.inputs));
    }
  }
  return absl::OkStatus();
}

void BatchResourceBase::SplitBatchCostsAndRecordMetrics(
    const std::string& model_name, int64_t model_version,
    absl::Span<const std::unique_ptr<tensorflow::CostMeasurement>>
        batch_cost_measurements,
    const int64_t processed_size, BatchT& batch) {
  absl::flat_hash_map<std::string, absl::Duration> batch_costs;
  // 1. Split the batch costs to each task.
  for (const auto& batch_cost_measurement : batch_cost_measurements) {
    if (batch_cost_measurement->GetTotalCost() <= absl::ZeroDuration()) {
      continue;
    }
    if (batch.size() == 0) {  // NOLINT: empty() checks the batch contains 0
                              // tasks. size() gets the sum of task sizes.
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the batch size is 0.";
      return;
    }
    if (processed_size == 0) {
      LOG_EVERY_N_SEC(ERROR, 60)
          << "Non-zero cost collected but the processed size is 0.";
      return;
    }
    const absl::string_view cost_type = batch_cost_measurement->GetCostType();
    const absl::Duration total_cost = batch_cost_measurement->GetTotalCost();
    batch_costs[cost_type] = total_cost;

    // Smeared batch cost: cost for processing this batch.
    RecordBatchCosts(model_name, model_version, processed_size,
                     absl::StrCat(cost_type, tensorflow::kWithSmearSuffix),
                     total_cost);
    // Non-smeared batch cost: cost for processing inputs in this batch, i.e.
    // cost for processing paddings is excluded.
    RecordBatchCosts(model_name, model_version, processed_size,
                     absl::StrCat(cost_type, tensorflow::kNoSmearSuffix),
                     total_cost / processed_size * batch.size());

    // Register batch stats for in-process use.
    if (cost_type == tensorflow::kTpuCostName) {
      ModelBatchStats& model_stats = GlobalBatchStatsRegistry().model(
          /* model_name= */ model_name, /* model_version= */ model_version);
      model_stats.batch_size(processed_size).tpu_cost().Register(total_cost);
      // batch.size() is the size of the original batch before padding.
      model_stats.RegisterProcessedSize(batch.size());
    }

    for (int i = 0; i < batch.num_tasks(); i++) {
      tensorflow::RequestCost* request_cost = batch.task(i).request_cost;
      // Skip recording the cost if the request_cost is null.
      if (!request_cost) continue;

      // Smeared cost: cost of paddings are assigned to each task.
      const auto cost_with_smear =
          total_cost / batch.size() * batch.task(i).size();

      // Non-smeared cost: cost of paddings are not assigned to any tasks.
      const auto cost_no_smear =
          total_cost / processed_size * batch.task(i).size();

      request_cost->RecordCost(
          {{absl::StrCat(cost_type, tensorflow::kWithSmearSuffix),
            cost_with_smear},
           {absl::StrCat(cost_type, tensorflow::kNoSmearSuffix),
            cost_no_smear}});
    }
  }

  // 2. Records the batch metrics in each task.
  const int64_t padding_size = processed_size - batch.size();
  for (int i = 0; i < batch.num_tasks(); i++) {
    tensorflow::RequestCost* request_cost = batch.task(i).request_cost;
    // Skip recording the metrics if the request_cost is null.
    if (!request_cost) continue;

    request_cost->RecordBatchMetrics(tensorflow::RequestCost::BatchMetrics{
        processed_size, static_cast<int64_t>(batch.task(i).size()),
        padding_size, batch_costs});
  }
}

}  // namespace neptune

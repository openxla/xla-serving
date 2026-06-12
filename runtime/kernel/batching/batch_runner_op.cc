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

#include "runtime/kernel/batching/batch_runner_op.h"  // from @neptune

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/streaming/utils.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/core/value_utils.h"  // from @neptune
#include "runtime/interpreter/linear_graph_execute.h"  // from @neptune
#include "runtime/kernel/batching/batch_resource_base.h"  // from @neptune
#include "runtime/kernel/batching/warmup.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/random.h"  // from @tsl
#include "tsl/platform/refcount.h"  // from @tsl
#include "tsl/profiler/lib/connected_traceme.h"  // from @tsl
#include "tsl/profiler/lib/context_types.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl
#include "tsl/profiler/lib/traceme_encode.h"  // from @tsl

namespace neptune {

void JsvBatchResource::ProcessFuncBatchImpl(
    const BatchResourceBase::BatchTask& propagated_task,
    absl::Span<const HostBuffer> inputs,
    std::vector<HostBuffer>* combined_outputs,
    std::function<void(const absl::Status&)> done) const {
  Kernel::ExecuteContext context;

  context.metadata = propagated_task.context->metadata;
  tsl::profiler::TraceMeConsumer trace_me(
      [&]() {
        return tsl::profiler::TraceMeEncode(
            "OrbaxServing::ProcessFuncBatchImpl",
            {{"model_id", absl::StrCat(context.metadata.model_name, ":",
                                       context.metadata.model_version)}});
      },
      tsl::profiler::ContextType::kJaxServingExecutor,
      context.metadata.trace_id);
  // `PackAsValues` takes `std::vector`, so we need to make a copy of the
  // inputs. Here copying HostBuffer is cheap because it only copy the reference
  // of the underlying buffer.
  // TODO: Consider make `PackAsValues` also take `absl::Span`.
  std::vector<HostBuffer> inputs_copy(inputs.begin(), inputs.end());
  std::vector<Value> inputs_copy_values =
      internal::PackAsValues(std::move(inputs_copy));
  // Here we assume that all input in `inputs` have batch dimension, which means
  // that we currently don't support captured inputs. If we want to support
  // captured inputs, it will need to be a separate field than `inputs`.
  context.inputs = inputs_copy_values;
  absl::StatusOr<std::vector<Value>> outputs = absl::InvalidArgumentError(
      "No batched kernel found. batched_graph_ is empty.");

  absl::Cleanup cleanup = [&]() { done(outputs.status()); };
  // Returns early if there is no batched kernel. The execution status will be
  // error because of the above error status.
  if (batched_graph_.empty()) return;

  outputs = ExecuteLinearGraph(
      propagated_task.context->run_context, context.inputs, batched_graph_,
      context.metadata.model_name, context.metadata.model_version);
  if (!outputs.ok()) {
    return;
  }
  combined_outputs->reserve(outputs->size());
  for (const auto& output : *outputs) {
    combined_outputs->push_back(output.Get<HostBuffer>());
  }
}

absl::StatusOr<absl::Span<const Value>> JsvBatchResource::GetInputsToBatch(
    Kernel::ExecuteContext* context) const {
  absl::Span<const Value> positional_inputs = context->inputs;
  if (positional_inputs.empty()) {
    return absl::InvalidArgumentError("Batching inputs cannot be empty.");
  }

  bool has_streaming = false;
  for (const auto& kernel : batched_graph_) {
    if (kernel->has_streaming()) {
      has_streaming = true;
      break;
    }
  }

  if (!has_streaming) {
    return positional_inputs;
  }

  int32_t request_id = context->run_context.request_id;

  absl::Span<const int64_t> dims =
      positional_inputs[0].Get<HostBuffer>().shape().dims();
  if (dims.empty()) {
    return absl::InvalidArgumentError(
        "Batching input buffers must have at least one dimension.");
  }

  // Add additional input for request_ids with the same batch size.
  int64_t batch_size = dims[0];
  TF_ASSIGN_OR_RETURN(auto request_ids_buffer,
                      neptune::CreateRequestIdsBuffer(batch_size, request_id));

  context->extended_inputs.assign(positional_inputs.begin(),
                                  positional_inputs.end());
  context->extended_inputs.push_back(Value(std::move(request_ids_buffer)));

  return absl::MakeSpan(context->extended_inputs);
}

std::string BatchRunnerOp::Name() const { return "BatchRunnerOp"; }

absl::StatusOr<std::unique_ptr<Kernel>> BatchRunnerOp::Create(
    const KernelBuilderContext& context) {
  // TODO: Implement this once we understand how the batching
  // config will be passed to here.
  return absl::UnimplementedError(
      "Not implemented yet. Use `CreateBatchRunnerOp` instead.");
}

tsl::Future<> BatchRunnerOp::RunAsyncInternal(ExecuteContext& context) {
  tsl::profiler::TraceMeProducer trace_me(
      [&]() {
        return tsl::profiler::TraceMeEncode(
            "OrbaxServing::BatchRunnerOp",
            {{"model_id", absl::StrCat(context.metadata.model_name, ":",
                                       context.metadata.model_version)}});
      },
      tsl::profiler::ContextType::kJaxServingExecutor);
  context.metadata.trace_id = trace_me.GetContextId();
  auto [promise, future] = tsl::MakePromise<>();
  auto create_batch_task_fn = []() {
    auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
    return batch_task;
  };
  absl::AnyInvocable<void(absl::Status)> done_callback =
      [promise = std::move(promise)](absl::Status status) mutable {
        promise.Set(status);
      };

  uint64_t guid = tsl::random::New64();
  // We pass the `batcher_queue_name` because the underlying scheduler use this
  // name to fine a batching queue. Since we are using one scheduler per batch
  // kernel, this may not be necessary.
  // TODO: Revisit this to see if we should just have single batch
  // queue per batch scheduler.
  absl::Status status;
  if (ShouldWarmupAllBatchSizes(context.metadata.model_name,
                                context.metadata.model_version)) {
    status = batch_resource_->RegisterInputsForAllBatchSizes(
        guid, &context, GetNumOutputs(), /*batcher_queue_name=*/"batch_queue",
        create_batch_task_fn, std::move(done_callback));
  } else {
    status = batch_resource_->RegisterInput(
        guid, &context, GetNumOutputs(), /*batcher_queue_name=*/"batch_queue",
        create_batch_task_fn, std::move(done_callback));
  }
  if (!status.ok()) {
    future = tsl::Future<>(status);
  }
  return future;
}

absl::Status BatchRunnerOp::MarkFullyInitialized() {
  for (auto& kernel : batch_resource_->batched_graph()) {
    TF_RETURN_IF_ERROR(kernel->MarkFullyInitialized());
  }
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<Kernel>> CreateBatchRunnerOp(
    std::vector<std::unique_ptr<Kernel>> kernels,
    const BatchRunnerOpOptions& options, tsl::thread::ThreadPool* thread_pool) {
  if (kernels.empty()) {
    return absl::InvalidArgumentError("At least one kernel is required.");
  }
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options scheduler_options;

  scheduler_options.num_batch_threads = options.num_batch_threads;

  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_RETURN_IF_ERROR(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      scheduler_options, &batcher));
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      JsvBatchResource::GetBatcherQueueOptions(
          scheduler_options.num_batch_threads, options.max_batch_size,
          options.batch_timeout_micros, options.max_enqueued_batches,
          options.allowed_batch_sizes, options.enable_large_batch_splitting,
          options.disable_padding, options.batch_padding_policy,
          options.low_priority_max_batch_size,
          options.low_priority_batch_timeout_micros,
          options.low_priority_max_enqueued_batches,
          options.low_priority_allowed_batch_sizes,
          options.mixed_priority_batching_policy, thread_pool,
          options.allocator);

  tsl::core::RefCountPtr<JsvBatchResource> batch_resource(
      new JsvBatchResource(std::move(kernels), batcher, queue_options,
                           options.allowed_batch_sizes, thread_pool));
  return std::make_unique<BatchRunnerOp>(std::move(batch_resource));
}

}  // namespace neptune

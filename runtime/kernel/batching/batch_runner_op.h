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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_RUNNER_OP_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_RUNNER_OP_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @neptune
#include "runtime/core/default_allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/kernel/batching/batch_options_utils.h"  // from @neptune
#include "runtime/kernel/batching/batch_resource_base.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/refcount.h"  // from @tsl
namespace neptune {

struct BatchRunnerOpOptions {
  int num_batch_threads = 0;
  int max_enqueued_batches = 0;
  int max_batch_size = 0;
  int batch_timeout_micros = 0;
  std::vector<int32_t> allowed_batch_sizes;
  bool enable_large_batch_splitting = true;
  bool disable_padding = false;
  std::string batch_padding_policy = std::string(kPadUpPolicy);
  int low_priority_max_batch_size = 0;
  int low_priority_batch_timeout_micros = 0;
  int low_priority_max_enqueued_batches = 0;
  std::vector<int32_t> low_priority_allowed_batch_sizes;
  MixedPriorityBatchingPolicy mixed_priority_batching_policy =
      MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  Allocator* allocator = DefaultAllocator::GetInstance();
};

// A batch implementation in JSV that take a list of kernels and run them in a
// batch mode.
class JsvBatchResource : public BatchResourceBase {
 public:
  JsvBatchResource(
      std::vector<std::unique_ptr<Kernel>> batched_graph,
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatchResourceBase::BatcherT::QueueOptions& batcher_queue_options,
      std::vector<int32_t> allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool)
      : BatchResourceBase(std::move(batcher), batcher_queue_options,
                          allowed_batch_sizes, thread_pool),
        batched_graph_(std::move(batched_graph)) {}

  const std::vector<std::unique_ptr<Kernel>>& batched_graph() const {
    return batched_graph_;
  }

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& propagated_task,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override;

 private:
  std::vector<std::unique_ptr<Kernel>> batched_graph_;
};

// A kernel that runs a a sequence of kernels in a batch mode.
class BatchRunnerOp : public CancellableAsyncKernel {
 public:
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context);

  ~BatchRunnerOp() override = default;

  std::string Name() const override;

  int GetNumInputs() const override {
    return batch_resource_->batched_graph().front()->GetNumInputs();
  }

  int GetNumOutputs() const override {
    return batch_resource_->batched_graph().back()->GetNumOutputs();
  }

  // TODO: Make it to private and use `CreateBatchRunnerOp`.
  explicit BatchRunnerOp(
      tsl::core::RefCountPtr<JsvBatchResource> batch_resource)
      : batch_resource_(std::move(batch_resource)) {}

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  absl::Status MarkFullyInitialized() override;

 private:
  tsl::Future<> RunAsyncInternal(ExecuteContext& context) override;

  tsl::core::RefCountPtr<JsvBatchResource> batch_resource_;
};

// The `thread_pool` will be used for non-blocking tasks such as concat and
// split, so here we reuse the worker pool. Note that the threads executing the
// batched kernels will not be in this pool.
absl::StatusOr<std::unique_ptr<Kernel>> CreateBatchRunnerOp(
    std::vector<std::unique_ptr<Kernel>> kernels,
    const BatchRunnerOpOptions& options, tsl::thread::ThreadPool* thread_pool);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_RUNNER_OP_H_

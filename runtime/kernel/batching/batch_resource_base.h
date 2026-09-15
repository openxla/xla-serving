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

#ifndef XLA_SERVING_RUNTIME_KERNEL_BATCHING_BATCH_RESOURCE_BASE_H_
#define XLA_SERVING_RUNTIME_KERNEL_BATCHING_BATCH_RESOURCE_BASE_H_

#include <cstddef>
#include <cstdint>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/flags/declare.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/queue_options.h"  // from @xla_serving
#include "runtime/core/batching/resource_base.h"  // from @xla_serving
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/threadsafe_status.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tensorflow/core/common_runtime/cost_measurement.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/request_cost.h"  // from @org_tensorflow
#include "tsl/platform/context.h"  // from @tsl
#include "tsl/platform/thread_annotations.h"  // from @tsl

ABSL_DECLARE_FLAG(bool, jsv_support_batching_for_ragged_inputs);

namespace xla_serving {

struct RaggedInputSpec {
  // Index of the flat_values tensor in the flat signature.
  int flat_values_index;
  // Index of the row_splits tensor in the flat signature.
  int row_splits_index;
};

// Base class for resource that encapsulating the state and logic for batching
// host buffers.
// Users are expected to call `RegisterInput` to enqueue a batch task. It
// requires all inputs to be `HostBuffer` and all are capable of being batched
// together with other `HostBuffer`s. It doesn't support non-batched inputs.
class BatchResourceBase : public ResourceBase {
 public:
  // Given a BatchTask (from one op invocation) with 'num_outputs'== M and
  // split into N sub tasks, BufferMatrix is a N X M matrix.
  // Namely, BufferMatrix[i][j] indicates the i-th split buffer of j-th output;
  // concatenating buffers along the 2nd dimension gives a output buffer.
  typedef std::vector<std::vector<HostBuffer>> BufferMatrix;

  // One task to be batched, corresponds to a `slice` of input from one batch-op
  // invocation.
  //
  // Given input from one batch-op invocation or a subtask that is split from
  // the input from an invocation, a `slice` is:
  // 1) Split each buffer in `BatchTask::inputs` along the 0th dimension.
  // 2) 'split_index' is calculated along the 0-th dimension.
  // A subtask may itself be re-split, allowing arbitrary nesting.
  //
  // Note input from one batch-op invocation is valid and considered a
  // specialized `slice`.
  struct BatchTask : public xla_serving::BatchTask {
    BatchTask() : criticality_val(tsl::criticality::GetCriticality()) {};

    // A unique ID to identify this invocation of Batch.
    int64_t guid;

    tsl::Context propagated_context;

    std::vector<HostBuffer> inputs;
    std::vector<HostBuffer> captured_inputs;
    Kernel::ExecuteContext* context;
    int num_outputs;

    // The index of this split, along the 0-th dimension of input from op
    // invocation.
    int split_index = 0;

    // Two-dimensional buffer matrix where output of all splits of task is
    // collected. The splits can be further split; BatchTask supports
    // arbitrary levels of splits.
    // Ownership shared by:
    // 1) each split of task (to fill one row in this matrix) and
    // 2) callback that runs to merge output of individual splits for an op
    // invocation, after all splits complete.
    std::shared_ptr<BufferMatrix> output;

    // 'status' records error (could be from any split) if at least one split
    // returns error, OK otherwise.
    // Ownership is shared by individual splits and callback.
    std::shared_ptr<ThreadSafeStatus> status;

    bool is_partial = false;

    uint64_t start_time;

    // This is only used for ragged inputs when the flag to support ragged
    // inputs is enabled.
    size_t task_size = 0;
    size_t size() const override {
      if (task_size == 0 && !inputs.empty()) {
        return inputs[0].shape().dims()[0];
      }
      return task_size;
    }

    // Create a split task from this one. The caller needs to setup the inputs
    // of the new task
    std::unique_ptr<BatchTask> CreateSplitTask(
        int split_index, absl::AnyInvocable<void(absl::Status)> done_callback);

    // RequestCost is for collecting the cost and must outlive the batching
    // processing.
    //
    // For example, to collect cost in rpc processing, `request_cost` is owned
    // by rpc handler and points to the RequestCost of an rpc which provides
    // the inputs to this BatchTask.
    //
    // After the batch processing, the request cost will be incremented with
    // this task's processing costs.
    tensorflow::RequestCost* request_cost = nullptr;

    // Returns the criticality associated with the task.
    tsl::criticality::Criticality criticality() const override {
      return criticality_val;
    };

    // Returns true if this task is a split subtask.
    bool is_subtask() const override { return is_partial; }

    // If nonzero, make a batch of this size entirely out of padding. This
    // batch is processed, but is not propagated to the kernel outputs.
    int forced_warmup_batch_size = 0;

    // Note the done callback MUST NOT (although unlikely in practice) attempt
    // to acquire the queue's lock — e.g., by calling methods like Schedule() on
    // the same queue — as this may lead to deadlock.
    void set_done_callback(absl::AnyInvocable<void(absl::Status)> callback) {
      done_callback = std::move(callback);
    }

   protected:
    virtual std::unique_ptr<BatchTask> CreateDerivedTask() {
      return std::make_unique<BatchTask>();
    }

    void FinishTaskImpl(const absl::Status& status) override {
      tsl::WithContext wc(this->propagated_context);
      if (!status.ok() && this->status) {
        this->status->Update(status);
      }
      if (this->done_callback) {
        this->done_callback(this->status->status());
        // Clear the callback to avoid double deletion.
        this->done_callback = nullptr;
      }
    }

   private:
    absl::AnyInvocable<void(absl::Status)> done_callback;

    // Criticality associated with the task.
    ::tsl::criticality::Criticality criticality_val;
  };

  using BatcherT = SharedBatchScheduler<BatchResourceBase::BatchTask>;
  using QueueOptions = ::xla_serving::QueueOptions<BatchResourceBase::BatchTask>;

  using BatcherQueueT = BatchScheduler<BatchResourceBase::BatchTask>;
  using BatchT = Batch<BatchResourceBase::BatchTask>;

  BatchResourceBase(std::shared_ptr<BatcherT> batcher,
                    const QueueOptions& batcher_queue_options,
                    std::vector<int32_t> allowed_batch_sizes,
                    tsl::thread::ThreadPool* thread_pool,
                    std::vector<RaggedInputSpec> ragged_inputs = {})
      : batcher_(std::move(batcher)),
        batcher_queue_options_(batcher_queue_options),
        allowed_batch_sizes_(std::move(allowed_batch_sizes)),
        allowed_batch_sizes_str_(absl::StrJoin(allowed_batch_sizes_, ",")),
        ragged_inputs_(std::move(ragged_inputs)),
        thread_pool_(thread_pool) {}

  using CreateBatchTaskFn =
      std::function<absl::StatusOr<std::unique_ptr<BatchTask>>()>;

  // Ingests data from one invocation of the batch op. The data is enqueued to
  // be combined with others into a batch, asynchronously.
  // `CreateBatchTaskFn` should be used to instantiate fields added to a
  // child class of `BatchTask` by the caller.
  absl::Status RegisterInput(
      int64_t guid, Kernel::ExecuteContext* context, int num_outputs,
      const std::string& batcher_queue_name,
      const CreateBatchTaskFn& create_batch_task_fn,
      absl::AnyInvocable<void(absl::Status)> done_callback,
      int forced_warmup_batch_size = 0);

  // Like `RegisterInput()`, but extra "dummy" batches are processed for each
  // batch size. Only the original request's outputs are propagated to the
  // caller.
  // NOTE: `done_callback` is only called for the original request, not
  // for the dummy warmup batches.
  absl::Status RegisterInputsForAllBatchSizes(
      int64_t guid, Kernel::ExecuteContext* context, int num_outputs,
      const std::string& batcher_queue_name,
      const CreateBatchTaskFn& create_batch_task_fn,
      absl::AnyInvocable<void(absl::Status)> done_callback);

  static QueueOptions GetBatcherQueueOptions(
      int32_t num_batch_threads, int32_t max_batch_size,
      int32_t batch_timeout_micros, int32_t max_enqueued_batches,
      const std::vector<int32_t>& allowed_batch_sizes,
      bool enable_large_batch_splitting, bool disable_padding,
      absl::string_view batch_padding_policy,
      int32_t low_priority_max_batch_size,
      int32_t low_priority_batch_timeout_micros,
      int32_t low_priority_max_enqueued_batches,
      const std::vector<int32_t>& low_priority_allowed_batch_sizes,
      MixedPriorityBatchingPolicy mixed_priority_batching_policy,
      PriorityBatchingPolicy priority_batching_policy,
      PriorityAwareBatchingOptions priority_aware_batching_options,
      tsl::thread::ThreadPool* thread_pool, Allocator* allocator,
      std::vector<RaggedInputSpec> ragged_inputs = {});

  // Splits 'input' of 'input_task_ptr' along the 0th dimension, into a list of
  // 'output_tasks'.
  // Task sizes are determined by
  // 1) open_batch_remaining_slot
  // 2) max_batch_size
  // 3) size-of-input-task
  // in a way that
  // 1) Task sizes add up to `size-of-input-task`.
  // 2) Task sizes from left to right are like
  //    [open_batch_remaining_slot, max_batch_size, max_batch_size, ...,
  //    `size-of-input-task` - `sum-of-previous-elements`].
  //
  // REQUIRES:
  // Caller should make sure size-of-input-task is greater than
  // open_batch_remaining_slot.
  static absl::Status SplitInputTask(
      std::unique_ptr<BatchTask>* input_task_ptr, int open_batch_remaining_slot,
      int max_batch_size, tsl::thread::ThreadPool* thread_pool,
      std::vector<std::unique_ptr<BatchTask>>* output_tasks,
      Allocator* allocator, std::vector<RaggedInputSpec> ragged_inputs = {});

  // Splits the batch costs to each task.
  //
  // Inputs:
  // 1) batch_cost_measurements, which provides the total cost of each type;
  // 2) processed_size, it's the batch size plus the padding amount;
  // 3) batch, provides the batch size and input sizes.
  //
  // Outputs:
  // The request_cost in each batch task will be updated.
  // - This function will use two approaches to split the batch cost (if it's
  //   non-zero), thus two costs will be output.
  //   1) smeared cost: batch cost is split proportionally to each task's size,
  //      and paddings do not share any cost;
  //   2) non-smeared cost: batch cost is split proportionally to each task or
  //      padding's size. Here padding's cost is not assigned to any tasks.
  // - This function will also record the metrics of this batch in each task,
  //   including:
  //   1) the batch size;
  //   2) the input size from this task;
  //   3) the padding amount.
  static void SplitBatchCostsAndRecordMetrics(
      const std::string& model_name, int64_t model_version,
      absl::Span<const std::unique_ptr<tensorflow::CostMeasurement>>
          batch_cost_measurements,
      int64_t processed_size, BatchT& batch);

  // Records the batch delay between the batch schedule time and the task start
  // time.
  static void RecordBatchDelayMetrics(const BatchResourceBase::BatchT& batch,
                                      absl::string_view model_name,
                                      int64_t model_version,
                                      int64_t processed_batch_size,
                                      absl::Time batch_schedule_time,
                                      absl::Duration batch_timeout);

  std::string DebugString() const override { return "BatchResourceBase"; }

 protected:
  // Implementation of calling the process batch function.
  virtual void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& propagated_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const = 0;

 private:
  friend class TestJsvBatchResource;
  friend class TestBatchResourceBase;

  virtual absl::StatusOr<absl::Span<const Value>> GetInputsToBatch(
      Kernel::ExecuteContext* context) const;

  // Validates that it's legal to combine the tasks in 'batch' into a batch.
  // Assumes the batch is non-empty.
  static absl::Status ValidateBatch(const BatchT& batch);

  // Returns a boolean indicating whether a batch is formed from low priority
  // tasks only or not.
  bool IsLowPriorityBatch(const BatchT& batch) const;

  // Returns the smallest entry in 'allowed_batch_sizes_' that is greater than
  // or equal to 'batch_size'. If 'allowed_batch_sizes_' is empty, simply
  // returns 'batch_size'.
  int RoundToLowestAllowedBatchSize(int batch_size,
                                    bool is_low_priority_batch = false) const;

  // Helper function to propagate the status to the task's context and call the
  // done callback on the task.
  void CleanUpFunctionHelper(BatchTask& task, const absl::Status& status) const;

  // Concatenates the input buffers of the tasks from the batch and the
  // unbatched task vector. When padding is enabled in the batcher queue, they
  // are padded with garbage value up to the nearest allowed batch size.
  absl::Status ConcatInputBuffers(
      const BatchT& batch,
      absl::Span<const std::unique_ptr<BatchTask>> unbatched_tasks,
      int padded_batch_size, int padding_amount,
      Kernel::ExecuteContext* context,
      std::vector<HostBuffer>* concatenated_buffers) const;

  absl::Status SplitOutputBuffers(
      absl::Span<const HostBuffer> combined_outputs, BatchT* batch,
      std::vector<std::unique_ptr<BatchTask>>& unbatched_tasks) const;

  absl::StatusOr<const BatchTask&> GetLatestDeadlineTask(
      const BatchT& batch) const;

  void ProcessFuncBatch(
      std::unique_ptr<BatchT> batch,
      std::vector<std::unique_ptr<BatchTask>> unbatched_tasks = {}) const;

  // Callback function that wraps the Process*Batch functions above. The caller
  // of the callback must guarantee that the unique pointers passed as argument
  // are not null.
  void ProcessBatchCallBack(
      std::unique_ptr<Batch<BatchTask>> batch,
      std::vector<std::unique_ptr<BatchTask>> unbatched_tasks);

  // Looks up the batcher queue for 'queue_name'. If it didn't previously exist,
  // creates it.
  //
  // The model_name and model_version are for the current model.
  absl::Status LookupOrCreateBatcherQueue(const std::string& queue_name,
                                          BatcherQueueT** queue,
                                          const std::string& model_name,
                                          int64_t model_version);

  // SessionMetadata session_metadata_;

  absl::Mutex outstanding_batch_mu_;
  size_t outstanding_batch_total_size_ TF_GUARDED_BY(outstanding_batch_mu_) = 0;
  // A batch scheduler, and options for creating queues.
  std::shared_ptr<BatcherT> batcher_;
  QueueOptions batcher_queue_options_;

  // A collection of batcher queues, keyed on queue name.
  mutable absl::Mutex batcher_queues_mu_;
  std::map<std::string, std::unique_ptr<BatcherQueueT>> batcher_queues_
      TF_GUARDED_BY(batcher_queues_mu_);

  std::vector<int32_t> allowed_batch_sizes_;
  // A concatenated string of <allowed_batch_sizes_>, separated by ",". This is
  // used to record batching parameter.
  std::string allowed_batch_sizes_str_;

  std::vector<RaggedInputSpec> ragged_inputs_;

  tsl::thread::ThreadPool* thread_pool_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_BATCHING_BATCH_RESOURCE_BASE_H_

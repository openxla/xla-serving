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

#include "runtime/core/batching/priority_aware_queue.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/fixed_array.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/fake_clock_env.h"  // from @xla_serving
#include "runtime/core/batching/input_split_metadata.h"  // from @xla_serving
#include "runtime/core/batching/queue_options.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla

namespace xla_serving {
namespace internal {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size,
                    tsl::criticality::Criticality criticality =
                        tsl::criticality::Criticality::kCritical,
                    int id = 0,
                    std::shared_ptr<absl::Status> status_out = nullptr,
                    bool is_subtask = false)
      : size_(size),
        criticality_(criticality),
        id_(id),
        status_out_(std::move(status_out)),
        is_subtask_(is_subtask) {}
  ~FakeTask() override = default;

  size_t size() const override { return size_; }

  tsl::criticality::Criticality criticality() const override {
    return criticality_;
  }

  int id() const { return id_; }

  bool is_subtask() const override { return is_subtask_; }
  void set_is_subtask(bool is_subtask) { is_subtask_ = is_subtask; }

 protected:
  void FinishTaskImpl(const absl::Status& status) override {
    if (status_out_ != nullptr) {
      *status_out_ = status;
    }
  }

 private:
  const size_t size_;
  const tsl::criticality::Criticality criticality_;
  const int id_;
  std::shared_ptr<absl::Status> status_out_;
  bool is_subtask_ = false;
};

using QueueOptions = ::xla_serving::QueueOptions<FakeTask>;

TEST(PriorityAwareQueueTest, BasicSchedule) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 20;

  bool schedulable_called = false;
  auto schedulable_callback = [&]() { schedulable_called = true; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  EXPECT_EQ(queue.SchedulingCapacity(), 20);
  EXPECT_TRUE(queue.IsEmpty());
  EXPECT_FALSE(queue.closed());

  std::unique_ptr<FakeTask> task1 = std::make_unique<FakeTask>(5);
  EXPECT_OK(queue.Schedule(&task1));

  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue.SchedulingCapacity(), 15);
  EXPECT_FALSE(schedulable_called);
  EXPECT_FALSE(queue.PeekBatchPriority().has_value());

  std::unique_ptr<FakeTask> task2 = std::make_unique<FakeTask>(5);
  EXPECT_OK(queue.Schedule(&task2));
  EXPECT_TRUE(schedulable_called);
  EXPECT_EQ(queue.NumEnqueuedTasks(), 2);
  EXPECT_EQ(queue.SchedulingCapacity(), 10);
  EXPECT_TRUE(queue.PeekBatchPriority().has_value());

  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->size(), 10);

  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, CriticalityOrdering) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 40;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Add tasks in reverse criticality order
  std::unique_ptr<FakeTask> task_sheddable = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kSheddable, /*id=*/1);
  std::unique_ptr<FakeTask> task_sheddable_plus = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kSheddablePlus, /*id=*/2);
  std::unique_ptr<FakeTask> task_critical = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kCritical, /*id=*/3);
  std::unique_ptr<FakeTask> task_critical_plus = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kCriticalPlus, /*id=*/4);
  std::unique_ptr<FakeTask> task_critical_2 = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kCritical, /*id=*/5);

  EXPECT_OK(queue.Schedule(&task_sheddable));
  env.AdvanceByMicroseconds(10);
  EXPECT_OK(queue.Schedule(&task_sheddable_plus));
  env.AdvanceByMicroseconds(10);
  EXPECT_OK(queue.Schedule(&task_critical));
  env.AdvanceByMicroseconds(10);
  EXPECT_OK(queue.Schedule(&task_critical_plus));
  env.AdvanceByMicroseconds(10);
  EXPECT_OK(queue.Schedule(&task_critical_2));

  EXPECT_EQ(queue.NumEnqueuedTasks(), 5);

  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_tasks(), 5);

  // Expected order: CriticalPlus (id 4), Critical (id 3), Critical (id 5),
  // SheddablePlus (id 2), Sheddable (id 1)
  EXPECT_EQ(batch->task(0).id(), 4);
  EXPECT_EQ(batch->task(1).id(), 3);
  EXPECT_EQ(batch->task(2).id(), 5);
  EXPECT_EQ(batch->task(3).id(), 2);
  EXPECT_EQ(batch->task(4).id(), 1);

  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, TaskEvictionOnFullQueue) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 10;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Fill queue with 2 Sheddable tasks of size 5
  auto status1 = std::make_shared<absl::Status>(absl::OkStatus());
  auto status2 = std::make_shared<absl::Status>(absl::OkStatus());
  auto task1 = std::make_unique<FakeTask>(
      5, tsl::criticality::Criticality::kSheddable, /*id=*/1, status1);
  auto task2 = std::make_unique<FakeTask>(
      5, tsl::criticality::Criticality::kSheddable, /*id=*/2, status2);

  EXPECT_OK(queue.Schedule(&task1));
  EXPECT_OK(queue.Schedule(&task2));
  EXPECT_EQ(queue.SchedulingCapacity(), 0);

  // Adding another Sheddable task should fail because it cannot evict same
  // or higher criticality tasks.
  auto task3 = std::make_unique<FakeTask>(
      5, tsl::criticality::Criticality::kSheddable, /*id=*/3);
  EXPECT_THAT(queue.Schedule(&task3),
              StatusIs(absl::StatusCode::kUnavailable,
                       HasSubstr("priority is too low to evict other tasks")));

  // Adding a Critical task should evict task2 (lowest priority, newest)
  auto task_4 = std::make_unique<FakeTask>(
      5, tsl::criticality::Criticality::kCritical, /*id=*/4);
  EXPECT_OK(queue.Schedule(&task_4));

  EXPECT_THAT(*status2,
              StatusIs(absl::StatusCode::kUnavailable,
                       HasSubstr("Task evicted due to priority queue full")));
  EXPECT_OK(*status1);

  // Verify batch contains task_4 and task1.
  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_tasks(), 2);
  EXPECT_EQ(batch->task(0).id(), 4);
  EXPECT_EQ(batch->task(1).id(), 1);

  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, TimeoutSchedulesBatch) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 20;

  bool schedulable_called = false;
  auto schedulable_callback = [&]() { schedulable_called = true; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(5);
  EXPECT_OK(queue.Schedule(&task));

  EXPECT_FALSE(schedulable_called);
  EXPECT_EQ(queue.ScheduleBatch(), nullptr);

  // Advance clock over timeout limit
  env.AdvanceByMicroseconds(1001);

  EXPECT_TRUE(queue.PeekBatchPriority().has_value());
  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->size(), 5);
  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, PeekBatchPriorityCalculation) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 20;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // CriticalPlus task (criticality = 3) -> max_priority - 3 = 0
  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(
      10, tsl::criticality::Criticality::kCriticalPlus);
  EXPECT_OK(queue.Schedule(&task));

  auto priority_key = queue.PeekBatchPriority();
  ASSERT_TRUE(priority_key.has_value());
  EXPECT_EQ(priority_key->first, 0);

  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, LargeTaskSplitting) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 30;
  options.batch_timeout_micros = 1000;
  options.enable_large_batch_splitting = true;
  options.priority_aware_batching_options.max_queue_depth = 30;
  options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int remaining_slots,
         int max_execution_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
        int task_size = (*input_task)->size();
        InputSplitMetadata metadata(task_size, remaining_slots,
                                    max_execution_size);
        absl::FixedArray<int> sizes = metadata.task_sizes();
        for (int size : sizes) {
          output_tasks->push_back(std::make_unique<FakeTask>(
              size, (*input_task)->criticality(), (*input_task)->id(),
              std::make_shared<absl::Status>(absl::OkStatus()),
              /*is_subtask=*/true));
        }
        return absl::OkStatus();
      };

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Input task of size 15 (splits into 10 and 5 upon ScheduleBatch)
  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(
      15, tsl::criticality::Criticality::kCritical, /*id=*/0);
  EXPECT_OK(queue.Schedule(&task));

  auto batch1 = queue.ScheduleBatch();
  ASSERT_NE(batch1, nullptr);
  EXPECT_EQ(batch1->size(), 10);
  EXPECT_EQ(batch1->task(0).id(), 0);
  queue.ProcessBatch(std::move(batch1));

  // Remainder task of size 5 should be left in queue
  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue.SchedulingCapacity(), 25);

  env.AdvanceByMicroseconds(1001);
  auto batch2 = queue.ScheduleBatch();
  ASSERT_NE(batch2, nullptr);
  EXPECT_EQ(batch2->size(), 5);
  EXPECT_EQ(batch2->task(0).id(), 0);
  queue.ProcessBatch(std::move(batch2));

  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, LargeTaskResplittingEnabled) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 30;
  options.batch_timeout_micros = 1000;
  options.enable_large_batch_splitting = true;
  options.priority_aware_batching_options.max_queue_depth = 30;
  options.priority_aware_batching_options.enable_task_resplit = true;
  options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int remaining_slots,
         int max_execution_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
        int task_size = (*input_task)->size();
        InputSplitMetadata metadata(task_size, remaining_slots,
                                    max_execution_size);
        absl::FixedArray<int> sizes = metadata.task_sizes();
        for (int size : sizes) {
          output_tasks->push_back(std::make_unique<FakeTask>(
              size, (*input_task)->criticality(), (*input_task)->id(),
              std::make_shared<absl::Status>(absl::OkStatus()),
              /*is_subtask=*/true));
        }
        return absl::OkStatus();
      };

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  std::unique_ptr<FakeTask> task1 = std::make_unique<FakeTask>(
      8, tsl::criticality::Criticality::kCritical, /*id=*/1);
  std::unique_ptr<FakeTask> task2 = std::make_unique<FakeTask>(
      8, tsl::criticality::Criticality::kSheddable, /*id=*/2);
  EXPECT_OK(queue.Schedule(&task1));
  EXPECT_OK(queue.Schedule(&task2));

  // Task2 is split into two tasks of size 2 and 6. The first task of size 2 is
  // scheduled along with task1.
  auto batch1 = queue.ScheduleBatch();
  ASSERT_NE(batch1, nullptr);
  EXPECT_EQ(batch1->size(), 10);
  EXPECT_EQ(batch1->task(0).id(), 1);
  EXPECT_EQ(batch1->task(1).id(), 2);
  EXPECT_TRUE(batch1->task(1).is_subtask());
  queue.ProcessBatch(std::move(batch1));

  // The remaining task of size 6 is left in the queue.
  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue.SchedulingCapacity(), 24);

  std::unique_ptr<FakeTask> task3 = std::make_unique<FakeTask>(
      8, tsl::criticality::Criticality::kCritical, /*id=*/3);
  EXPECT_OK(queue.Schedule(&task3));

  // The remaining task2 of size 6 is split into two tasks of size 2 and 4.
  // The first task of size 2 is scheduled along with task3.
  auto batch2 = queue.ScheduleBatch();
  ASSERT_NE(batch2, nullptr);
  EXPECT_EQ(batch2->size(), 10);
  EXPECT_EQ(batch2->task(0).id(), 3);
  EXPECT_EQ(batch2->task(1).id(), 2);
  EXPECT_TRUE(batch2->task(1).is_subtask());
  queue.ProcessBatch(std::move(batch2));

  // The remaining task2 of size 4 is left in the queue.
  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  EXPECT_EQ(queue.SchedulingCapacity(), 26);

  env.AdvanceByMicroseconds(1001);
  auto batch3 = queue.ScheduleBatch();
  ASSERT_NE(batch3, nullptr);
  EXPECT_EQ(batch3->size(), 4);
  EXPECT_EQ(batch3->task(0).id(), 2);
  queue.ProcessBatch(std::move(batch3));

  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, BatchPaddingPolicyBatchDown) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 16;
  options.input_batch_size_limit = 16;
  options.batch_timeout_micros = 1000;
  options.allowed_batch_sizes = {4, 8, 16};
  options.batch_padding_policy = std::string(kBatchDownPolicy);
  options.priority_aware_batching_options.max_queue_depth = 32;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Add 5 tasks of size 2 -> total size 10
  for (int i = 0; i < 5; ++i) {
    auto task = std::make_unique<FakeTask>(
        2, tsl::criticality::Criticality::kCritical, /*id=*/i);
    EXPECT_OK(queue.Schedule(&task));
  }

  env.AdvanceByMicroseconds(1001);

  // BatchDown should schedule up to allowed batch size 8 (4 tasks of size 2)
  auto batch1 = queue.ScheduleBatch();
  ASSERT_NE(batch1, nullptr);
  EXPECT_EQ(batch1->size(), 8);
  EXPECT_EQ(batch1->num_tasks(), 4);
  queue.ProcessBatch(std::move(batch1));

  // 1 task of size 2 remains in queue
  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  auto batch2 = queue.ScheduleBatch();
  ASSERT_NE(batch2, nullptr);
  EXPECT_EQ(batch2->size(), 2);
  queue.ProcessBatch(std::move(batch2));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, PreservesOriginalStartTimeInBatch) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.priority_aware_batching_options.max_queue_depth = 20;

  bool schedulable_called = false;
  auto schedulable_callback = [&]() { schedulable_called = true; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Record time when task is enqueued.
  uint64_t enqueue_time_micros = env.NowMicros();

  auto task = std::make_unique<FakeTask>(
      2, tsl::criticality::Criticality::kCritical, /*id=*/1);
  EXPECT_OK(queue.Schedule(&task));

  // Advance time so that the time when batch is scheduled is different
  // from the enqueue time.
  env.AdvanceByMicroseconds(1001);
  uint64_t schedule_time_micros = env.NowMicros();
  EXPECT_GT(schedule_time_micros, enqueue_time_micros);

  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->size(), 2);

  // The earliest task start time in the batch should match the time the task
  // was originally enqueued, NOT the time the batch was created.
  EXPECT_EQ(batch->EarliestTaskStartTime(), enqueue_time_micros);

  queue.ProcessBatch(std::move(batch));
}

TEST(PriorityAwareQueueTest, ScheduleBatchSplitFailureReturnsNullptr) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 30;
  options.batch_timeout_micros = 1000;
  options.enable_large_batch_splitting = true;
  options.priority_aware_batching_options.max_queue_depth = 20;
  // Configure split_input_task_func to fail.
  options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int remaining_slots,
         int max_execution_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
        return absl::InternalError("Failed to split task");
      };
  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);

  // Input task of size 15 is larger than max_execution_batch_size (10)
  // and requires splitting. Use kSheddable so it can be evicted later.
  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(
      15, tsl::criticality::Criticality::kSheddable, /*id=*/0);
  EXPECT_OK(queue.Schedule(&task));
  // The queue is schedulable (size 15 >= max_execution_batch_size 10), but
  // ScheduleBatch() cannot dequeue any task because task splitting fails.
  auto batch = queue.ScheduleBatch();

  EXPECT_EQ(batch, nullptr);
  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  EXPECT_FALSE(queue.IsEmpty());

  // Evict the unsplit task by scheduling a higher priority task that fits
  // directly (size 10 <= max_execution_batch_size 10) to empty the queue
  // before destruction (required by ~PriorityAwareQueue).
  std::unique_ptr<FakeTask> critical_task = std::make_unique<FakeTask>(
      10, tsl::criticality::Criticality::kCritical, /*id=*/1);
  EXPECT_OK(queue.Schedule(&critical_task));
  auto batch2 = queue.ScheduleBatch();
  EXPECT_EQ(batch2->size(), 10);
  EXPECT_EQ(batch2->task(0).id(), 1);
  queue.ProcessBatch(std::move(batch2));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(PriorityAwareQueueTest, TaskSizeLargerThanMaxQueueDepth) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 20;
  options.priority_aware_batching_options.max_queue_depth = 10;
  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  PriorityAwareQueue<FakeTask> queue(options, &env, process_batch_callback,
                                     schedulable_callback);
  // A task whose size (15) is within input_batch_size_limit (20) but exceeds
  // max_queue_depth (10).
  auto task = std::make_unique<FakeTask>(15);
  EXPECT_THAT(
      queue.Schedule(&task),
      StatusIs(
          absl::StatusCode::kUnavailable,
          HasSubstr("task size 15 is larger than the max queue depth 10")));
  EXPECT_TRUE(queue.IsEmpty());
}


}  // namespace
}  // namespace internal
}  // namespace xla_serving

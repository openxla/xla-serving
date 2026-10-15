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

#include "runtime/core/batching/fifo_queue.h"  // from @xla_serving

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/fixed_array.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/synchronization/notification.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/fake_clock_env.h"  // from @xla_serving
#include "runtime/core/batching/input_split_metadata.h"  // from @xla_serving
#include "runtime/core/batching/queue_options.h"  // from @xla_serving
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla

namespace xla_serving {
namespace internal {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class FakeTask : public BatchTask {
 public:
  explicit FakeTask(size_t size, tsl::criticality::Criticality criticality =
                                     tsl::criticality::Criticality::kCritical)
      : size_(size), criticality_(criticality) {}
  ~FakeTask() override = default;
  size_t size() const override { return size_; }

  tsl::criticality::Criticality criticality() const override {
    return criticality_;
  }

 private:
  const size_t size_;
  const tsl::criticality::Criticality criticality_;
};

using QueueOptions = ::xla_serving::QueueOptions<FakeTask>;

TEST(FifoQueueTest, BasicSchedule) {
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;

  bool schedulable_called = false;
  auto schedulable_callback = [&]() { schedulable_called = true; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  EXPECT_EQ(queue.max_task_size(), 10);
  EXPECT_EQ(queue.max_execution_batch_size(), 10);
  EXPECT_EQ(queue.SchedulingCapacity(), 20);
  EXPECT_FALSE(queue.closed());

  std::unique_ptr<FakeTask> task1 = std::make_unique<FakeTask>(5);
  EXPECT_OK(queue.Schedule(&task1));

  EXPECT_EQ(queue.NumEnqueuedTasks(), 1);
  // Capacity is max_enqueued_batches * max_execution_batch_size -
  // num_enqueued_tasks = 2 * 10 - 5 = 15.
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
  EXPECT_NE(batch, nullptr);

  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(FifoQueueTest, IsEmptyTracksProcessedBatches) {
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  EXPECT_TRUE(queue.IsEmpty());

  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(10);
  EXPECT_OK(queue.Schedule(&task));

  auto batch = queue.ScheduleBatch();
  EXPECT_NE(batch, nullptr);

  // IsEmpty() should be false because the batch is being processed (even though
  // num_enqueued_ is 0).
  EXPECT_FALSE(queue.IsEmpty());

  queue.ProcessBatch(std::move(batch));
  EXPECT_TRUE(queue.IsEmpty());
}

TEST(FifoQueueTest, CloseAndWaitUntilEmpty) {
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(10);
  EXPECT_OK(queue.Schedule(&task));

  auto batch = queue.ScheduleBatch();
  EXPECT_NE(batch, nullptr);

  absl::Notification closed_notification;
  std::unique_ptr<tsl::Thread> thread(
      tsl::Env::Default()->StartThread({}, "CloseQueueThread", [&]() {
        queue.CloseAndWaitUntilEmpty();
        closed_notification.Notify();
      }));

  // Wait for a little bit to ensure the thread is blocked.
  EXPECT_FALSE(closed_notification.WaitForNotificationWithTimeout(
      absl::Milliseconds(50)));

  // Return the batch. This should unblock the waiting thread.
  queue.ProcessBatch(std::move(batch));

  closed_notification.WaitForNotification();
  EXPECT_TRUE(queue.closed());
}

TEST(FifoQueueTest, StrictFifoOrderingWithCriticality) {
  QueueOptions options;
  options.max_execution_batch_size = 30;
  options.input_batch_size_limit = 30;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};

  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  std::unique_ptr<FakeTask> task1 =
      std::make_unique<FakeTask>(10, tsl::criticality::Criticality::kSheddable);
  std::unique_ptr<FakeTask> task2 =
      std::make_unique<FakeTask>(10, tsl::criticality::Criticality::kCritical);
  std::unique_ptr<FakeTask> task3 = std::make_unique<FakeTask>(
      10, tsl::criticality::Criticality::kCriticalPlus);

  EXPECT_OK(queue.Schedule(&task1));
  EXPECT_OK(queue.Schedule(&task2));
  EXPECT_OK(queue.Schedule(&task3));

  auto batch = queue.ScheduleBatch();
  ASSERT_NE(batch, nullptr);
  EXPECT_EQ(batch->num_tasks(), 3);
  EXPECT_EQ(batch->task(0).criticality(),
            tsl::criticality::Criticality::kSheddable);
  EXPECT_EQ(batch->task(1).criticality(),
            tsl::criticality::Criticality::kCritical);
  EXPECT_EQ(batch->task(2).criticality(),
            tsl::criticality::Criticality::kCriticalPlus);

  queue.ProcessBatch(std::move(batch));
}

TEST(FifoQueueTest, TaskLargerThanInputBatchSizeLimitFails) {
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.max_enqueued_batches = 2;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(11);
  EXPECT_THAT(queue.Schedule(&task),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Task size 11 is larger than maximum "
                                 "input batch size 10")));
}

TEST(FifoQueueTest, TaskLargerThanSchedulingCapacityWithSplitEnabledFails) {
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 100;
  options.max_enqueued_batches = 2;
  options.enable_large_batch_splitting = true;
  options.split_input_task_func = [](std::unique_ptr<FakeTask>*, int, int,
                                     std::vector<std::unique_ptr<FakeTask>>*) {
    return absl::OkStatus();
  };

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  FifoQueue<FakeTask> queue(options, tsl::Env::Default(),
                            process_batch_callback, schedulable_callback);

  // Capacity is max_execution_batch_size * max_enqueued_batches = 20.
  std::unique_ptr<FakeTask> task = std::make_unique<FakeTask>(21);
  EXPECT_THAT(
      queue.Schedule(&task),
      StatusIs(absl::StatusCode::kUnavailable,
               HasSubstr("The batch scheduling queue to which this task was "
                         "submitted is full; task size is 21 but scheduling "
                         "capacity is only 20")));
}

TEST(FifoQueueTest, TaskExceedsMaxEnqueuedBatchesWithSplitDisabledFails) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;
  options.enable_large_batch_splitting = false;

  auto schedulable_callback = []() {};
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  FifoQueue<FakeTask> queue(options, &env, process_batch_callback,
                            schedulable_callback);

  std::unique_ptr<FakeTask> task1 = std::make_unique<FakeTask>(10);
  EXPECT_OK(queue.Schedule(&task1));

  std::unique_ptr<FakeTask> task2 = std::make_unique<FakeTask>(10);
  EXPECT_OK(queue.Schedule(&task2));

  std::unique_ptr<FakeTask> task3 = std::make_unique<FakeTask>(10);
  // Hit max_enqueued_batches limit in queue.
  EXPECT_THAT(
      queue.Schedule(&task3),
      StatusIs(absl::StatusCode::kUnavailable,
               HasSubstr("The batch scheduling queue to which this task was "
                         "submitted is full; currently 2 batches enqueued and "
                         "max_enqueued_batches is 2")));

  auto batch1 = queue.ScheduleBatch();
  ASSERT_NE(batch1, nullptr);
  queue.ProcessBatch(std::move(batch1));
  auto batch2 = queue.ScheduleBatch();
  ASSERT_NE(batch2, nullptr);
  queue.ProcessBatch(std::move(batch2));
}

TEST(FifoQueueTest, LargeTaskSplitting) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 30;
  options.max_enqueued_batches = 3;
  options.batch_timeout_micros = 1000;
  options.enable_large_batch_splitting = true;
  options.split_input_task_func =
      [](std::unique_ptr<FakeTask>* input_task, int remaining_slots,
         int max_execution_size,
         std::vector<std::unique_ptr<FakeTask>>* output_tasks) {
        int task_size = (*input_task)->size();
        InputSplitMetadata metadata(task_size, remaining_slots,
                                    max_execution_size);
        absl::FixedArray<int> sizes = metadata.task_sizes();
        for (int size : sizes) {
          output_tasks->push_back(std::make_unique<FakeTask>(size));
        }
        return absl::OkStatus();
      };

  int schedule_calls = 0;
  auto schedulable_callback = [&]() { schedule_calls++; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  FifoQueue<FakeTask> queue(options, &env, process_batch_callback,
                            schedulable_callback);

  std::unique_ptr<FakeTask> task =
      std::make_unique<FakeTask>(15);  // Splits into 10 and 5
  EXPECT_OK(queue.Schedule(&task));

  EXPECT_EQ(queue.NumEnqueuedTasks(), 2);
  EXPECT_GT(schedule_calls, 0);

  auto batch1 = queue.ScheduleBatch();
  ASSERT_NE(batch1, nullptr);
  EXPECT_EQ(batch1->size(), 10);
  queue.ProcessBatch(std::move(batch1));

  env.AdvanceByMicroseconds(1001);

  auto batch2 = queue.ScheduleBatch();
  ASSERT_NE(batch2, nullptr);
  EXPECT_EQ(batch2->size(), 5);
  queue.ProcessBatch(std::move(batch2));
}

TEST(FifoQueueTest, TimeoutSchedulesBatch) {
  test_util::FakeClockEnv env(tsl::Env::Default());
  QueueOptions options;
  options.max_execution_batch_size = 10;
  options.input_batch_size_limit = 10;
  options.batch_timeout_micros = 1000;
  options.max_enqueued_batches = 2;

  bool schedulable_called = false;
  auto schedulable_callback = [&]() { schedulable_called = true; };
  auto process_batch_callback = [](std::unique_ptr<Batch<FakeTask>> batch) {};
  FifoQueue<FakeTask> queue(options, &env, process_batch_callback,
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
}

}  // namespace
}  // namespace internal
}  // namespace xla_serving

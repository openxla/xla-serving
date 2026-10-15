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

#include "runtime/kernel/batching/batch_resource_base.h"  // from @xla_serving

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/synchronization/notification.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/threadsafe_status.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_utils.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving
#include "runtime/kernel/batching/test_batch_resource_base.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/lib/monitoring/test_utils.h"  // from @xla
#include "xla/tsl/platform/criticality.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/env_time.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tensorflow/core/common_runtime/request_cost.h"  // from @org_tensorflow
#include "tsl/platform/refcount.h"  // from @tsl

namespace xla_serving {

using ::absl_testing::StatusIs;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

constexpr char kModelName[] = "test_model";
constexpr int64_t kModelVersion = 1;

namespace {

SharedBatchScheduler<BatchResourceBase::BatchTask>::Options
GetDefaultBatcherOptions(int num_batch_threads = 4) {
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options options;
  options.num_batch_threads = num_batch_threads;
  return options;
}

class BatchResourceBaseTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    padding_sizes_reader_ = std::make_unique<CellReader<Histogram>>(
        "/orbax/serving/batching/padding_sizes");
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "BatcherThreadPool",
        /*num_threads=*/4);
    allocator_ = DefaultAllocator::GetInstance();
  }
  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_sizes_reader_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  Allocator* allocator_;
};

TEST_F(BatchResourceBaseTest, BatchingSuccess) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 108;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0, /*max_enqueued_batches=*/num_requests,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  absl::Mutex mu;
  int sent_requests = 0;
  auto [all_request_sent_promise, all_request_sent_future] =
      tsl::MakePromise<>();
  // Use a different thread pool to send requests because we want this thread
  // pool to have the same number of thread as the number of requests to
  // simulate the case where all requests are sent from different threads.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), tsl::ThreadOptions(),
                               "SentRequestsThreadPool",
                               /*num_threads=*/num_requests);
  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
    contexts[i].inputs = inputs[i];
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    pool.Schedule([&, index = i]() {
      auto [promise, future] = tsl::MakePromise<>();
      futures[index] = std::move(future);
      auto done_callback =
          [promise = std::move(promise)](absl::Status status) mutable {
            promise.Set(status);
          };
      EXPECT_OK(batch_resource->RegisterInput(
          /*guid=*/index, &contexts[index], /*num_outputs=*/1, "batch_queue",
          create_batch_task_fn, std::move(done_callback)));
      absl::MutexLock lock(mu);
      ++sent_requests;
      if (sent_requests == num_requests) {
        all_request_sent_promise.Set(absl::OkStatus());
      }
    });
  }
  TF_ASSERT_OK(all_request_sent_future.Await());

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              {2, 3, 4, 5}, xla::ifrt::Shape({1, 4}))));
    }
  }
}

TEST_F(BatchResourceBaseTest, BatchingSuccessWithLargeBatchSplitting) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(1)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> input_data(20);
    std::iota(input_data.begin(), input_data.end(), i);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(input_data, xla::ifrt::Shape({5, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> expected_output_data(20);
    std::iota(expected_output_data.begin(), expected_output_data.end(), i + 1);
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              expected_output_data, xla::ifrt::Shape({5, 4}))));
    }
  }
  // Input's batch size is 5, and there are 4 requests, so total batch size is
  // 20. The max batch size is 8, so there are 2 batches with batch size 8 and
  // 1 batch with batch size 4.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            2);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
}

TEST_F(BatchResourceBaseTest, BatchingFailedWithLargeBatchSplitting) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestFailedBatchResource::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(1)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestFailedBatchResource(batcher, queue_options, allowed_batch_sizes,
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> input_data(20);
    std::iota(input_data.begin(), input_data.end(), i);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(input_data, xla::ifrt::Shape({5, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    EXPECT_THAT(futures[i].Await(), StatusIs(absl::StatusCode::kInternal));
  }
  // Input's batch size is 5, and there are 4 requests, so total batch size is
  // 20. The max batch size is 8, so there are 2 batches with batch size 8 and
  // 1 batch with batch size 4.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            2);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
}

TEST_F(BatchResourceBaseTest, BatchingSuccessWithoutLargeBatchSplitting) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  // Use a different thread pool to send requests because we want this thread
  // pool to have the same number of thread as the number of requests to
  // simulate the case where all requests are sent from different threads.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), tsl::ThreadOptions(),
                               "SentRequestsThreadPool",
                               /*num_threads=*/num_requests);
  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(Value(AsHostBuffer<int32_t>(
        {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20},
        xla::ifrt::Shape({5, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              {2,  3,  4,  5,  6,  7,  8,  9,  10, 11,
                               12, 13, 14, 15, 16, 17, 18, 19, 20, 21},
                              xla::ifrt::Shape({5, 4}))));
    }
  }
  // Input's batch size is 5, and there are 4 requests. When large batch
  // splitting is disabled, each request take one batch, so the total number of
  // batches is 4 and each has size 8 with 3 padding.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            4);
  Histogram padding_sizes_histogram = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "8");
  EXPECT_EQ(padding_sizes_histogram.sum(), 12);
  // Padding size is 3 and is put in the 4th bucket.
  EXPECT_EQ(padding_sizes_histogram.num(4), 4);
}

TEST_F(BatchResourceBaseTest, BatchingFailedInputBatchSizeMismatch) {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 108;
  BatchResourceBase::BatcherT::QueueOptions queue_options;
  queue_options.allowed_batch_sizes = allowed_batch_sizes;
  queue_options.input_batch_size_limit = 8;
  queue_options.batch_timeout_micros = 0;
  queue_options.max_enqueued_batches = num_requests;
  queue_options.allocator = allocator_;
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  absl::Mutex mu;
  int sent_requests = 0;
  auto [all_request_sent_promise, all_request_sent_future] =
      tsl::MakePromise<>();
  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), tsl::ThreadOptions(),
                               "TestBatchResourceBaseThreadPool",
                               /*num_threads=*/num_requests);
  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
    inputs[i].push_back(Value(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                                    xla::ifrt::Shape({2, 4}))));
    contexts[i].inputs = inputs[i];
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    pool.Schedule([&, index = i]() {
      auto [promise, future] = tsl::MakePromise<>();
      futures[index] = std::move(future);
      auto done_callback =
          [promise = std::move(promise)](absl::Status status) mutable {
            promise.Set(status);
          };
      absl::Status status = batch_resource->RegisterInput(
          /*guid=*/index, &contexts[index], /*num_outputs=*/1, "batch_queue",
          create_batch_task_fn, std::move(done_callback));
      if (!status.ok()) {
        futures[index] = tsl::Future<>(status);
      }
      absl::MutexLock lock(mu);
      ++sent_requests;
      if (sent_requests == num_requests) {
        all_request_sent_promise.Set(absl::OkStatus());
      }
    });
  }
  TF_ASSERT_OK(all_request_sent_future.Await());

  for (int i = 0; i < num_requests; ++i) {
    EXPECT_THAT(futures[i].Await(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(BatchResourceBaseTest, BatchingWithBatchDownPaddingPolicyTrimmed) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  int num_requests = 3;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kBatchDownPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  // Use a different thread pool to send requests because we want this thread
  // pool to have the same number of thread as the number of requests to
  // simulate the case where all requests are sent from different threads.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), tsl::ThreadOptions(),
                               "SentRequestsThreadPool",
                               /*num_threads=*/num_requests);
  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), i);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(input_data, xla::ifrt::Shape({1, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> expected_output_data(4);
    std::iota(expected_output_data.begin(), expected_output_data.end(), i + 1);
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              expected_output_data, xla::ifrt::Shape({1, 4}))));
    }
  }
  // Input's batch size is 1, and there are 3 requests, so total batch size is
  // 3. With batch down padding policy, the batch size is trimmed to the smaller
  // allowed batch size 2. So there are 1 batch with batch size 2 and 1 batch
  // with batch size 1. No padding needed.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "1"),
            1);
  Histogram padding_sizes_histogram_size_2 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "2");
  EXPECT_EQ(padding_sizes_histogram_size_2.sum(), 0);
  Histogram padding_sizes_histogram_size_1 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "1");
  EXPECT_EQ(padding_sizes_histogram_size_1.sum(), 0);
}

TEST_F(BatchResourceBaseTest, BatchingWithBatchDownPaddingPolicyNotTrimmed) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4, 8};
  int num_requests = 3;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kBatchDownPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  // Use a different thread pool to send requests because we want this thread
  // pool to have the same number of thread as the number of requests to
  // simulate the case where all requests are sent from different threads.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), tsl::ThreadOptions(),
                               "SentRequestsThreadPool",
                               /*num_threads=*/num_requests);
  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), i);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(input_data, xla::ifrt::Shape({1, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> expected_output_data(4);
    std::iota(expected_output_data.begin(), expected_output_data.end(), i + 1);
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              expected_output_data, xla::ifrt::Shape({1, 4}))));
    }
  }
  // Input's batch size is 1, and there are 3 requests, so total batch size is
  // 3. With batch down padding policy, the batch size is not trimmed because
  // it's less than the smallest allowed batch size 4. Therefore the batch size
  // (3) is padded up (padding size is 1) to the next allowed batch size 4.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  Histogram padding_sizes_histogram = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "4");
  EXPECT_EQ(padding_sizes_histogram.sum(), 1);
  // Padding size is 1 and is put in the 2nd bucket.
  EXPECT_EQ(padding_sizes_histogram.num(2), 1);
}

TEST_F(BatchResourceBaseTest, BatchingSuccessWithVariedDeadlines) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/1, /*max_batch_size=*/4,
          /*batch_timeout_micros=*/absl::ToInt64Microseconds(absl::Seconds(5)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false, /*disable_padding=*/false,
          /*pad_variable_length_inputs=*/false, kBatchDownPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  // Request 0: Deadline in 1s.
  // Request 1: Deadline in 3s.
  // Request 2: Deadline in 1s.
  // Request 3: Deadline in 1s.
  std::vector<absl::Time> deadlines = {
      absl::Now() + absl::Seconds(1), absl::Now() + absl::Seconds(3),
      absl::Now() + absl::Seconds(1), absl::Now() + absl::Seconds(1)};

  for (int i = 0; i < num_requests; ++i) {
    contexts[i].metadata = {.model_name = kModelName,
                            .model_version = kModelVersion};
    contexts[i].run_context.cancellation_context.deadline = deadlines[i];
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 4}))));
    contexts[i].inputs = inputs[i];
  }

  auto create_batch_task_fn = [&]() {
    auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
    return batch_task;
  };

  for (int i = 0; i < num_requests; ++i) {
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  // The batch should be formed and processed within the latest deadline (3s).
  // All requests should succeed.
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_THAT(outputs[0], HostBufferEq(AsHostBuffer<int32_t>(
                                {2, 3, 4, 5}, xla::ifrt::Shape({1, 4}))));
  }
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  // The propagated task should have the latest deadline (3s).
  EXPECT_EQ(batch_resource.get()->propagated_task_deadline(), deadlines[1]);
}

TEST_F(BatchResourceBaseTest, GetLatestDeadlineTaskSelectsLargestDeadline) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, /*batcher_queue_options=*/{},
                                /*allowed_batch_sizes=*/{4},
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  auto batch = std::make_unique<BatchResourceBase::BatchT>();
  std::vector<std::unique_ptr<Kernel::ExecuteContext>> contexts;

  auto create_context = [&](absl::Time deadline) {
    auto context = std::make_unique<Kernel::ExecuteContext>();
    context->run_context.cancellation_context.deadline = deadline;
    return context;
  };

  contexts.push_back(create_context(absl::Now() + absl::Seconds(1)));
  contexts.push_back(create_context(absl::Now() + absl::Seconds(2)));
  contexts.push_back(create_context(absl::Now() + absl::Seconds(5)));
  contexts.push_back(create_context(absl::Now() + absl::Seconds(3)));

  for (const auto& context : contexts) {
    auto task = std::make_unique<BatchResourceBase::BatchTask>();
    task->context = context.get();
    task->inputs.push_back(
        AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 4})));
    batch->AddTask(std::move(task));
  }
  batch->Close();

  TF_ASSERT_OK_AND_ASSIGN(const BatchResourceBase::BatchTask& latest_task,
                          batch_resource->GetLatestDeadlineTask(*batch));
  // The latest deadline is from contexts[2] (5s).
  EXPECT_EQ(latest_task.context->run_context.cancellation_context.deadline,
            contexts[2]->run_context.cancellation_context.deadline);
}

TEST_F(BatchResourceBaseTest, GetLatestDeadlineTaskExitsEarlyOnInfiniteFuture) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, /*batcher_queue_options=*/{},
                                /*allowed_batch_sizes=*/{4},
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  auto batch = std::make_unique<BatchResourceBase::BatchT>();
  std::vector<std::unique_ptr<Kernel::ExecuteContext>> contexts;

  auto create_context = [&](absl::Time deadline) {
    auto context = std::make_unique<Kernel::ExecuteContext>();
    context->run_context.cancellation_context.deadline = deadline;
    return context;
  };

  contexts.push_back(create_context(absl::Now() + absl::Seconds(1)));
  contexts.push_back(create_context(absl::InfiniteFuture()));
  contexts.push_back(create_context(absl::InfiniteFuture()));
  contexts.push_back(create_context(absl::InfiniteFuture()));

  for (const auto& context : contexts) {
    auto task = std::make_unique<BatchResourceBase::BatchTask>();
    task->context = context.get();
    task->inputs.push_back(
        AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({1, 4})));
    batch->AddTask(std::move(task));
  }
  batch->Close();

  TF_ASSERT_OK_AND_ASSIGN(const BatchResourceBase::BatchTask& latest_task,
                          batch_resource->GetLatestDeadlineTask(*batch));
  // Verify that the latest task returned is the second task.
  EXPECT_EQ(std::addressof(latest_task), std::addressof(batch->task(1)));
}

TEST_F(BatchResourceBaseTest, GetLatestDeadlineTaskCrashesOnEmptyBatch) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  tsl::core::RefCountPtr<TestBatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, /*batcher_queue_options=*/{},
                                /*allowed_batch_sizes=*/{4},
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  auto batch = std::make_unique<BatchResourceBase::BatchT>();
  // The batch is empty.
  batch->Close();

  EXPECT_THAT(batch_resource->GetLatestDeadlineTask(*batch),
              StatusIs(absl::StatusCode::kFailedPrecondition));
}

TEST_F(BatchResourceBaseTest, BatchFunctionCancelledBeforeStart) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(/*num_batch_threads=*/1), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestFailedBatchResource::GetBatcherQueueOptions(
          /*num_batch_threads=*/1, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0, /*max_enqueued_batches=*/1,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestFailedBatchResource(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          absl::CancelledError(
              "Function was cancelled before it was started")));
  std::vector<Value> inputs = {
      Value(AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2, 1})))};
  Kernel::ExecuteContext context;
  context.inputs = inputs;
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  auto [promise, future] = tsl::MakePromise<>();
  auto done_callback = [promise =
                            std::move(promise)](absl::Status status) mutable {
    promise.Set(status);
  };
  auto create_batch_task_fn = []() {
    return std::make_unique<BatchResourceBase::BatchTask>();
  };
  ASSERT_OK(batch_resource->RegisterInput(
      /*guid=*/0, &context, /*num_outputs=*/1, "test_batch",
      create_batch_task_fn, std::move(done_callback)));
  // CleanUpFunctionHelper catches "Function was cancelled before it was
  // started" error and finishes the task with OkStatus.
  EXPECT_TRUE(future.Await().ok());
}

class RaggedBatchResourceBaseTest : public BatchResourceBaseTest {
 protected:
  void SetUp() override {
    BatchResourceBaseTest::SetUp();
    absl::SetFlag(&FLAGS_xla_serving_support_batching_for_ragged_inputs, true);
  }
};

TEST_F(RaggedBatchResourceBaseTest, RaggedBatchingSuccess) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_,
          {{.flat_values_index = 1, .row_splits_index = 0}});
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 1, .row_splits_index = 0}}));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>({0, 2, 5}, xla::ifrt::Shape({3}))));
    inputs[i].push_back(Value(AsHostBuffer<int32_t>(
        {i, i + 1, i + 2, i + 3, i + 4}, xla::ifrt::Shape({5}))));
    contexts[i].inputs = inputs[i];
    contexts[i].metadata = {.model_name = kModelName,
                            .model_version = kModelVersion};
    auto create_batch_task_fn = [&]() {
      return std::make_unique<BatchResourceBase::BatchTask>();
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback =
        [promise = std::move(promise)](absl::Status status) mutable {
          promise.Set(status);
        };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    EXPECT_THAT(outputs[0],
                HostBufferEq(AsHostBuffer<int32_t>({2 * i + 1, 3 * i + 9},
                                                   xla::ifrt::Shape({2, 1}))));
  }
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            1);
}

TEST_F(RaggedBatchResourceBaseTest,
       RaggedBatchingSuccessWithLargeBatchSplitting) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(1)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_,
          {{.flat_values_index = 1, .row_splits_index = 0}});
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 1, .row_splits_index = 0}}));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(Value(
        AsHostBuffer<int32_t>({0, 2, 4, 6, 8, 10}, xla::ifrt::Shape({6}))));
    std::vector<int32_t> values(10, i + 1);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(values, xla::ifrt::Shape({10}))));
    contexts[i].inputs = inputs[i];
    contexts[i].metadata = {.model_name = kModelName,
                            .model_version = kModelVersion};
    auto create_batch_task_fn = [&]() {
      return std::make_unique<BatchResourceBase::BatchTask>();
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback =
        [promise = std::move(promise)](absl::Status status) mutable {
          promise.Set(status);
        };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    std::vector<int32_t> expected_sums(5, 2 * (i + 1));
    EXPECT_THAT(outputs[0], HostBufferEq(AsHostBuffer<int32_t>(
                                expected_sums, xla::ifrt::Shape({5, 1}))));
  }
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            2);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
}

TEST_F(RaggedBatchResourceBaseTest, RaggedBatchingWithoutLargeBatchSplitting) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_,
          {{.flat_values_index = 1, .row_splits_index = 0}});
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 1, .row_splits_index = 0}}));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(Value(
        AsHostBuffer<int32_t>({0, 2, 4, 6, 8, 10}, xla::ifrt::Shape({6}))));
    std::vector<int32_t> values(10, i + 1);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(values, xla::ifrt::Shape({10}))));
    contexts[i].inputs = inputs[i];
    contexts[i].metadata = {.model_name = kModelName,
                            .model_version = kModelVersion};
    auto create_batch_task_fn = [&]() {
      return std::make_unique<BatchResourceBase::BatchTask>();
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback =
        [promise = std::move(promise)](absl::Status status) mutable {
          promise.Set(status);
        };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    std::vector<int32_t> expected_sums(5, 2 * (i + 1));
    EXPECT_THAT(outputs[0], HostBufferEq(AsHostBuffer<int32_t>(
                                expected_sums, xla::ifrt::Shape({5, 1}))));
  }
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            4);
  Histogram padding_sizes_histogram = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "8");
  EXPECT_EQ(padding_sizes_histogram.sum(), 12);
}

TEST_F(RaggedBatchResourceBaseTest, RaggedBatchingWarmupOnly) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0, /*max_enqueued_batches=*/10,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_,
          {{.flat_values_index = 1, .row_splits_index = 0}});
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 1, .row_splits_index = 0}}));

  Kernel::ExecuteContext context;
  std::vector<Value> inputs;
  inputs.push_back(Value(AsHostBuffer<int32_t>({0, 2}, xla::ifrt::Shape({2}))));
  inputs.push_back(
      Value(AsHostBuffer<int32_t>({10, 20}, xla::ifrt::Shape({2}))));
  context.inputs = inputs;
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  auto create_batch_task_fn = [&]() {
    return std::make_unique<BatchResourceBase::BatchTask>();
  };
  auto [promise, future] = tsl::MakePromise<>();
  auto done_callback =
      [promise = std::move(promise)](absl::Status status) mutable {
        promise.Set(status);
      };
  ASSERT_OK(batch_resource->RegisterInput(
      /*guid=*/0, &context, /*num_outputs=*/1, "batch_queue",
      create_batch_task_fn, std::move(done_callback),
      /*forced_warmup_batch_size=*/4));

  TF_ASSERT_OK(future.Await());
  EXPECT_TRUE(context.outputs.empty());
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
}

TEST_F(RaggedBatchResourceBaseTest,
       RaggedBatchingRegisterInputsForAllBatchSizes) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0,
          /*max_enqueued_batches=*/10,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(), allocator_,
          {{.flat_values_index = 1, .row_splits_index = 0}});
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 1, .row_splits_index = 0}}));

  Kernel::ExecuteContext context;
  std::vector<Value> inputs;
  inputs.push_back(
      Value(AsHostBuffer<int32_t>({0, 2, 5}, xla::ifrt::Shape({3}))));
  inputs.push_back(Value(
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5}, xla::ifrt::Shape({5}))));
  context.inputs = inputs;
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  auto create_batch_task_fn = [&]() {
    return std::make_unique<BatchResourceBase::BatchTask>();
  };
  auto [promise, future] = tsl::MakePromise<>();
  auto done_callback =
      [promise = std::move(promise)](absl::Status status) mutable {
        promise.Set(status);
      };
  ASSERT_OK(batch_resource->RegisterInputsForAllBatchSizes(
      /*guid=*/0, &context, /*num_outputs=*/1, "batch_queue",
      create_batch_task_fn, std::move(done_callback)));

  TF_ASSERT_OK(future.Await());
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<HostBuffer> outputs,
      internal::UnpackFromValues<HostBuffer>(std::move(context.outputs)));
  ASSERT_EQ(outputs.size(), 1);
  EXPECT_THAT(outputs[0],
              HostBufferEq(AsHostBuffer<int32_t>({3, 12},
                                                 xla::ifrt::Shape({2, 1}))));
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            2);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            1);
}

TEST_F(RaggedBatchResourceBaseTest,
       RaggedBatchingFailedInputBatchSizeMismatch) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options;
  queue_options.allowed_batch_sizes = allowed_batch_sizes;
  queue_options.input_batch_size_limit = 8;
  queue_options.batch_timeout_micros = 0;
  queue_options.max_enqueued_batches = 10;
  queue_options.allocator = allocator_;
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestRaggedBatchResourceBase(
          batcher, queue_options, allowed_batch_sizes, thread_pool_.get(),
          {{.flat_values_index = 2, .row_splits_index = 1}}));

  Kernel::ExecuteContext context;
  std::vector<Value> inputs;
  inputs.push_back(Value(AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               xla::ifrt::Shape({4, 2}))));
  inputs.push_back(
      Value(AsHostBuffer<int32_t>({0, 1, 2, 3}, xla::ifrt::Shape({4}))));
  inputs.push_back(
      Value(AsHostBuffer<int32_t>({10, 20, 30}, xla::ifrt::Shape({3}))));
  context.inputs = inputs;
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  auto create_batch_task_fn = [&]() {
    return std::make_unique<BatchResourceBase::BatchTask>();
  };
  auto [promise, future] = tsl::MakePromise<>();
  auto done_callback =
      [promise = std::move(promise)](absl::Status status) mutable {
        promise.Set(status);
      };
  absl::Status status = batch_resource->RegisterInput(
      /*guid=*/0, &context, /*num_outputs=*/1, "batch_queue",
      create_batch_task_fn, std::move(done_callback));
  EXPECT_THAT(status, StatusIs(absl::StatusCode::kInvalidArgument,
                               ::testing::HasSubstr("must have size 5")));
}

TEST(BatchTaskCriticalityTest, CriticalityDefaultsToCritical) {
  BatchResourceBase::BatchTask batch_task;
  EXPECT_EQ(batch_task.criticality(), tsl::criticality::Criticality::kCritical);
}

struct MaxExecutionBatchSizeTestParams {
  std::string test_name;
  bool enable_large_batch_splitting;
  int max_batch_size;
  std::vector<int> allowed_batch_sizes;
  absl::flat_hash_map<int, int> expected_batch_size_and_count;
  absl::flat_hash_map<int, int> expected_batch_size_and_padding_sum;
};

class BatchResourceBaseMaxExecutionBatchSizeTest
    : public ::testing::TestWithParam<MaxExecutionBatchSizeTestParams> {
 protected:
  void SetUp() override {
    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    padding_sizes_reader_ = std::make_unique<CellReader<Histogram>>(
        "/orbax/serving/batching/padding_sizes");
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "BatcherThreadPool",
        /*num_threads=*/4);
    allocator_ = DefaultAllocator::GetInstance();
  }
  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_sizes_reader_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  Allocator* allocator_;
};

TEST_P(BatchResourceBaseMaxExecutionBatchSizeTest,
       BatchingWithMaxExecutionBatchSize) {
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  int num_requests = 10;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/GetParam().max_batch_size,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, GetParam().allowed_batch_sizes,
          /*enable_large_batch_splitting=*/
          GetParam().enable_large_batch_splitting,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{},
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(
          batcher, queue_options, GetParam().allowed_batch_sizes,
          /*low_priority_allowed_batch_sizes=*/{}, thread_pool_.get()));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> input_data(4);
    std::iota(input_data.begin(), input_data.end(), i);
    inputs[i].push_back(
        Value(AsHostBuffer<int32_t>(input_data, xla::ifrt::Shape({1, 4}))));
    contexts[i] = {
        .metadata = {.model_name = kModelName, .model_version = kModelVersion},
        .inputs = inputs[i],
    };
    auto create_batch_task_fn = [&]() {
      auto batch_task = std::make_unique<BatchResourceBase::BatchTask>();
      return batch_task;
    };
    auto [promise, future] = tsl::MakePromise<>();
    futures[i] = std::move(future);
    auto done_callback = [promise =
                              std::move(promise)](absl::Status status) mutable {
      promise.Set(status);
    };
    ASSERT_OK(batch_resource->RegisterInput(
        /*guid=*/i, &contexts[i], /*num_outputs=*/1, "batch_queue",
        create_batch_task_fn, std::move(done_callback)));
  }

  for (int i = 0; i < num_requests; ++i) {
    std::vector<int32_t> expected_output_data(4);
    std::iota(expected_output_data.begin(), expected_output_data.end(), i + 1);
    TF_ASSERT_OK(futures[i].Await());
    TF_ASSERT_OK_AND_ASSIGN(
        std::vector<HostBuffer> outputs,
        internal::UnpackFromValues<HostBuffer>(std::move(contexts[i].outputs)));
    ASSERT_EQ(outputs.size(), 1);
    for (const auto& output : outputs) {
      EXPECT_THAT(output, HostBufferEq(AsHostBuffer<int32_t>(
                              expected_output_data, xla::ifrt::Shape({1, 4}))));
    }
  }

  for (const auto& [batch_size, expected_count] :
       GetParam().expected_batch_size_and_count) {
    EXPECT_EQ(
        executed_batch_size_reader_->Delta(
            kModelName, absl::StrCat(kModelVersion), absl::StrCat(batch_size)),
        expected_count);
  }
  for (const auto& [batch_size, expected_padding_sum] :
       GetParam().expected_batch_size_and_padding_sum) {
    EXPECT_EQ(padding_sizes_reader_
                  ->Delta(kModelName, absl::StrCat(kModelVersion),
                          absl::StrCat(batch_size))
                  .sum(),
              expected_padding_sum);
  }
}

INSTANTIATE_TEST_SUITE_P(
    BatchResourceBaseMaxExecutionBatchSizeTests,
    BatchResourceBaseMaxExecutionBatchSizeTest,
    testing::ValuesIn<MaxExecutionBatchSizeTestParams>({
        // There are 10 requests and each request has task size 1. When batch
        // splitting is enabled and allowed_batch_sizes is empty, the
        // max_execution_batch_size is assigned by the max_batch_size 16. Since
        // allowed_batch_sizes is empty, any batch size <= 16 is allowed.
        // Therefore an input batch of size 10 is processed directly with no
        // padding.
        {
            "batch_splitting_enabled_and_allowed_batch_sizes_empty",
            /*enable_large_batch_splitting=*/true,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{},
            /*expected_batch_size_and_count=*/{{10, 1}},
            /*expected_batch_size_and_padding_sum=*/{{10, 0}},
        },
        // Same requests as above. With batch splitting disabled,
        // max_execution_batch_size is set by input_batch_size_limit, which
        // inherits its value from max_batch_size (16). Since
        // allowed_batch_sizes is empty, any batch size <= 16 is permitted.
        // Therefore, an input batch of size 10 is processed directly with no
        // padding.
        {
            "batch_splitting_disabled_and_allowed_batch_sizes_empty",
            /*enable_large_batch_splitting=*/false,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{},
            /*expected_batch_size_and_count=*/{{10, 1}},
            /*expected_batch_size_and_padding_sum=*/{{10, 0}},
        },
        // Same requests as above. When batch splitting is enabled and
        // allowed_batch_sizes is not empty, the max_execution_batch_size is
        // assigned to the largest allowed_batch_size 8. There are two batches.
        // The first batch has 8 requests with total size 8, no padding. The
        // second batch has 2 requests with total size 2, padding to size 4
        // with 2 paddings.
        {
            "batch_splitting_enabled_and_allowed_batch_sizes_not_empty",
            /*enable_large_batch_splitting=*/true,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{4, 8},
            /*expected_batch_size_and_count=*/{{4, 1}, {8, 1}},
            /*expected_batch_size_and_padding_sum=*/{{4, 2}, {8, 0}},
        },
        // Same requests as above. When batch splitting is disabled, the
        // max_execution_batch_size is assigned to the max_batch_size 16. Since
        // allowed_batch_sizes is not empty and the padding policy is pad up,
        // there is one batch of total size 10 which is padded to size 16 with 6
        // paddings.
        {
            "batch_splitting_disabled_and_allowed_batch_sizes_not_empty",
            /*enable_large_batch_splitting=*/false,
            /*max_batch_size=*/16,
            /*allowed_batch_sizes=*/{4, 8, 16},
            /*expected_batch_size_and_count=*/{{4, 0}, {8, 0}, {16, 1}},
            /*expected_batch_size_and_padding_sum=*/{{4, 0}, {8, 0}, {16, 6}},
        },
    }),
    [](const ::testing::TestParamInfo<
        BatchResourceBaseMaxExecutionBatchSizeTest::ParamType>& info) {
      return info.param.test_name;
    });

std::unique_ptr<BatchResourceBase::BatchTask> MakeBatchTask(
    const int64_t task_size, tensorflow::RequestCost* request_cost,
    absl::Time start_time = absl::FromUnixMicros(tsl::EnvTime::NowMicros())) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->inputs.push_back(AsHostBuffer<int32_t>(
      std::vector<int32_t>(task_size, 0), xla::ifrt::Shape({task_size, 1})));
  task->request_cost = request_cost;
  task->start_time = absl::ToUnixMicros(start_time);
  return task;
}

TEST(RecordBatchDelayMetricsTest, ScheduleBatchAtTimeout) {
  CellReader<Histogram> batch_delay_us_reader(
      "/orbax/serving/batching/batch_delay_us");
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_arrival_delay = batch_timeout / 4;
  const absl::Time task1_start_time = absl::FromUnixMicros(1000000);
  const absl::Time task2_start_time = task1_start_time + task2_arrival_delay;
  const absl::Time batch_schedule_time = task1_start_time + batch_timeout;

  BatchResourceBase::BatchT batch;
  tensorflow::RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", kModelVersion, /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_EQ(
      batch_delay_us_reader
          .Delta("model_name", absl::StrCat(kModelVersion), "20")
          .sum(),
      absl::ToDoubleMicroseconds(batch_timeout) +
          absl::ToDoubleMicroseconds(batch_timeout - task2_arrival_delay));
}

TEST(RecordBatchDelayMetricsTest, ScheduleBatchBeforeTimeout) {
  CellReader<Histogram> batch_delay_us_reader(
      "/orbax/serving/batching/batch_delay_us");
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_arrival_delay = batch_timeout / 4;
  const absl::Duration scheduling_delay = batch_timeout / 10;
  const absl::Time task1_start_time = absl::FromUnixMicros(1000000);
  const absl::Time task2_start_time = task1_start_time + task2_arrival_delay;
  const absl::Time batch_schedule_time =
      task1_start_time + task2_arrival_delay + scheduling_delay;

  BatchResourceBase::BatchT batch;
  tensorflow::RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", kModelVersion, /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_EQ(batch_delay_us_reader
                .Delta("model_name", absl::StrCat(kModelVersion), "20")
                .sum(),
            absl::ToDoubleMicroseconds(task2_arrival_delay + scheduling_delay) +
                absl::ToDoubleMicroseconds(scheduling_delay));
}

TEST(RecordBatchDelayMetricsTest, ScheduleBatchAtTimeoutWithQueueingDelay) {
  CellReader<Histogram> batch_delay_us_reader(
      "/orbax/serving/batching/batch_delay_us");
  const absl::Duration batch_timeout = absl::Seconds(1);
  const absl::Duration task2_arrival_delay = batch_timeout / 4;
  const absl::Duration queueing_delay = 5 * batch_timeout;
  const absl::Time task1_start_time = absl::FromUnixMicros(1000000);
  const absl::Time task2_start_time = task1_start_time + task2_arrival_delay;
  const absl::Time batch_schedule_time =
      task1_start_time + batch_timeout + queueing_delay;

  BatchResourceBase::BatchT batch;
  tensorflow::RequestCost cost1, cost2;
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost1, task1_start_time));
  batch.AddTask(MakeBatchTask(/*task_size=*/1, &cost2, task2_start_time));
  batch.Close();

  BatchResourceBase::RecordBatchDelayMetrics(
      batch, "model_name", kModelVersion, /*processed_size=*/20,
      batch_schedule_time, batch_timeout);

  EXPECT_EQ(batch_delay_us_reader
                .Delta("model_name", absl::StrCat(kModelVersion), "20")
                .sum(),
            absl::ToDoubleMicroseconds(batch_timeout + queueing_delay) +
                absl::ToDoubleMicroseconds(batch_timeout - task2_arrival_delay +
                                           queueing_delay));
}

TEST(BatchTaskTest, FinishTaskPropagatesErrorStatus) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  absl::Status received_status;
  task->set_done_callback([&](absl::Status status) {
    received_status = status;
  });

  absl::Status error = absl::InternalError("Something went wrong");
  task->FinishTask(error);

  EXPECT_EQ(received_status, error);
  EXPECT_EQ(task->status->status(), error);
}

TEST(BatchTaskTest, FinishTaskIsIdempotent) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  int call_count = 0;
  task->set_done_callback([&](absl::Status status) { call_count++; });

  // Call FinishTask multiple times
  task->FinishTask(absl::OkStatus());
  task->FinishTask(absl::OkStatus());
  task->FinishTask(absl::InternalError("Late error"));

  // Should only run once.
  EXPECT_EQ(call_count, 1);
}

TEST(BatchTaskTest, FinishTaskPreservesAccumulatedErrors) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  // Update task status with an error
  absl::Status prior_error = absl::UnavailableError(
      "Something went wrong earlier.");
  task->status->Update(prior_error);

  absl::Status received_status;
  task->set_done_callback([&](absl::Status status) {
    received_status = status;
  });

  // Finish with OK should still propagate the accumulated error
  task->FinishTask(absl::OkStatus());

  EXPECT_EQ(task->status->status(), prior_error);
  EXPECT_EQ(received_status, prior_error);
}

TEST(BatchTaskTest, FinishTaskHandlesNullCallback) {
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  task->status = std::make_shared<ThreadSafeStatus>();

  // Should not crash even without setting done_callback.
  EXPECT_NO_FATAL_FAILURE(task->FinishTask(absl::OkStatus()));
}

TEST(SplitInputTaskTest, ConcatFailurePropagatesError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "SplitTaskThreadPool",
                                      /*num_threads=*/2);

  Kernel::ExecuteContext context;
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  // Input buffer with shape {6, 4}, batch size = 6
  std::vector<int32_t> data(24, 1);
  task->inputs.push_back(AsHostBuffer<int32_t>(data, xla::ifrt::Shape({6, 4})));
  task->context = &context;
  task->num_outputs = 1;
  task->output = std::make_shared<BatchResourceBase::BufferMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  absl::Notification done;
  absl::Status final_status;
  task->set_done_callback([&](absl::Status status) {
    final_status = status;
    done.Notify();
  });

  // Split task {6, 4} into two tasks: {2, 4} and {4, 4}.
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> output_tasks;
  ASSERT_OK(BatchResourceBase::SplitInputTask(
      &task, /*open_batch_remaining_slot=*/2, /*max_batch_size=*/4,
      &thread_pool, &output_tasks, allocator));
  ASSERT_EQ(output_tasks.size(), 2);

  // Intentionally populate incompatible buffer shapes ({2, 4} vs {4, 5})
  // to trigger a Concat failure.
  std::vector<int32_t> output_data_0(8, 1);
  std::vector<int32_t> output_data_1(20, 1);
  (*output_tasks[0]->output)[0][0] =
      AsHostBuffer<int32_t>(output_data_0, xla::ifrt::Shape({2, 4}));
  (*output_tasks[1]->output)[1][0] =
      AsHostBuffer<int32_t>(output_data_1, xla::ifrt::Shape({4, 5}));

  output_tasks[0]->FinishTask(absl::OkStatus());
  output_tasks[1]->FinishTask(absl::OkStatus());

  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_THAT(
      final_status,
      StatusIs(absl::StatusCode::kInvalidArgument,
               ::testing::HasSubstr("Dimensions of inputs should match")));
}

TEST(SplitInputTaskTest, SubtaskFailurePropagatesError) {
  Allocator* allocator = DefaultAllocator::GetInstance();
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "SplitTaskThreadPool",
                                      /*num_threads=*/2);

  Kernel::ExecuteContext context;
  auto task = std::make_unique<BatchResourceBase::BatchTask>();
  // Input buffer with shape {6, 4}, batch size = 6
  std::vector<int32_t> data(24, 1);
  task->inputs.push_back(
      AsHostBuffer<int32_t>(data, xla::ifrt::Shape({6, 4})));
  task->context = &context;
  task->num_outputs = 1;
  task->output = std::make_shared<BatchResourceBase::BufferMatrix>();
  task->status = std::make_shared<ThreadSafeStatus>();
  task->guid = 0;
  task->start_time = 0;

  absl::Notification done;
  absl::Status final_status;
  task->set_done_callback([&](absl::Status status) {
    final_status = status;
    done.Notify();
  });

  // Split task {6, 4} into two tasks: {2, 4} and {4, 4}.
  std::vector<std::unique_ptr<BatchResourceBase::BatchTask>> output_tasks;
  ASSERT_OK(BatchResourceBase::SplitInputTask(
      &task, /*open_batch_remaining_slot=*/2, /*max_batch_size=*/4,
      &thread_pool, &output_tasks, allocator));
  ASSERT_EQ(output_tasks.size(), 2);

  // Complete subtask 0 with an error.
  output_tasks[0]->FinishTask(
      absl::UnavailableError("Subtask is failed."));
  output_tasks[1]->FinishTask(absl::OkStatus());

  ASSERT_TRUE(done.WaitForNotificationWithTimeout(absl::Seconds(5)));
  EXPECT_THAT(final_status,
              StatusIs(absl::StatusCode::kUnavailable,
                       ::testing::HasSubstr("Subtask is failed.")));
  // Verify concatenation was skipped and outputs are empty.
  EXPECT_TRUE(context.outputs.empty());
}

TEST_F(BatchResourceBaseTest, DefaultPriorityAwareBatchingOptions) {
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0, /*max_enqueued_batches=*/10,
          /*allowed_batch_sizes=*/{2, 4, 8},
          /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, /*pad_variable_length_inputs=*/false,
          kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kPriorityAware,
          /*priority_aware_batching_options=*/{}, thread_pool_.get(),
          allocator_);

  EXPECT_EQ(queue_options.priority_aware_batching_options.max_queue_depth, 0);
  EXPECT_FALSE(
      queue_options.priority_aware_batching_options.enable_task_resplit);
  EXPECT_EQ(queue_options.priority_batching_policy,
            PriorityBatchingPolicy::kPriorityAware);
}

class PadVariableLengthInputsTest
    : public BatchResourceBaseTest,
      public ::testing::WithParamInterface<bool> {};

TEST_P(PadVariableLengthInputsTest, PadVariableLengthInputs) {
  const bool pad_variable_length_inputs = GetParam();
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      GetDefaultBatcherOptions(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4};
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/1, /*max_batch_size=*/3,
          /*batch_timeout_micros=*/1000000, /*max_enqueued_batches=*/10,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, pad_variable_length_inputs, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          PriorityBatchingPolicy::kStrictFifo,
          /*priority_aware_batching_options=*/{}, thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestBatchResourceBase(batcher, queue_options, allowed_batch_sizes,
                                /*low_priority_allowed_batch_sizes=*/{},
                                thread_pool_.get()));

  // Request 0: shape [1, 1, 2]
  // Request 1: shape [2, 2, 1]
  // Common non-batch shape is [_, 2, 2].
  std::vector<Value> inputs0 = {
      Value(AsHostBuffer<int32_t>({10, 20}, xla::ifrt::Shape({1, 1, 2})))};
  std::vector<Value> inputs1 = {Value(
      AsHostBuffer<int32_t>({30, 40, 50, 60}, xla::ifrt::Shape({2, 2, 1})))};

  Kernel::ExecuteContext context0{
      .metadata = {.model_name = kModelName, .model_version = kModelVersion},
      .inputs = inputs0};
  Kernel::ExecuteContext context1{
      .metadata = {.model_name = kModelName, .model_version = kModelVersion},
      .inputs = inputs1};

  auto [promise0, future0] = tsl::MakePromise<>();
  auto [promise1, future1] = tsl::MakePromise<>();
  auto create_batch_task_fn = []() {
    return std::make_unique<BatchResourceBase::BatchTask>();
  };

  ASSERT_OK(batch_resource->RegisterInput(
      /*guid=*/0, &context0, /*num_outputs=*/1, "batch_queue",
      create_batch_task_fn,
      [p = std::move(promise0)](absl::Status status) mutable {
        p.Set(status);
      }));
  ASSERT_OK(batch_resource->RegisterInput(
      /*guid=*/1, &context1, /*num_outputs=*/1, "batch_queue",
      create_batch_task_fn,
      [p = std::move(promise1)](absl::Status status) mutable {
        p.Set(status);
      }));

  if (pad_variable_length_inputs) {
    ASSERT_OK(future0.Await());
    ASSERT_OK(future1.Await());

    HostBuffer expected_output0 =
        AsHostBuffer<int32_t>({11, 21, 1, 1}, xla::ifrt::Shape({1, 2, 2}));
    HostBuffer expected_output1 = AsHostBuffer<int32_t>(
        {31, 1, 41, 1, 51, 1, 61, 1}, xla::ifrt::Shape({2, 2, 2}));

    ASSERT_EQ(context0.outputs.size(), 1);
    ASSERT_EQ(context1.outputs.size(), 1);
    EXPECT_THAT(context0.outputs[0].Get<HostBuffer>(),
                HostBufferEq(expected_output0));
    EXPECT_THAT(context1.outputs[0].Get<HostBuffer>(),
                HostBufferEq(expected_output1));
  } else {
    EXPECT_THAT(
        future0.Await(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 ::testing::HasSubstr(
                     "Dimensions of inputs should match: shape[0] = [1,1,2] vs."
                     " shape[1] = [2,2,1]")));
    EXPECT_THAT(
        future1.Await(),
        StatusIs(absl::StatusCode::kInvalidArgument,
                 ::testing::HasSubstr(
                     "Dimensions of inputs should match: shape[0] = [1,1,2] vs."
                     " shape[1] = [2,2,1]")));
  }
}

INSTANTIATE_TEST_SUITE_P(PadVariableLengthInputsTests,
                         PadVariableLengthInputsTest, ::testing::Bool());

}  // namespace
}  // namespace xla_serving

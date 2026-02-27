#include "third_party/neptune/runtime/kernel/batching/batch_resource_base.h"

#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/allocator.h"
#include "third_party/neptune/runtime/core/batching/batch_scheduler.h"
#include "third_party/neptune/runtime/core/batching/shared_batch_scheduler.h"
#include "third_party/neptune/runtime/core/default_allocator.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_utils.h"
#include "third_party/neptune/runtime/kernel/batching/batch_options_utils.h"
#include "third_party/neptune/runtime/kernel/batching/test_batch_resource_base.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/test_utils.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/criticality.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/tsl/platform/refcount.h"

namespace jsv {

using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

constexpr char kModelName[] = "test_model";
constexpr int64_t kModelVersion = 1;

namespace {

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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 108;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/0, /*max_enqueued_batches=*/num_requests,
          allowed_batch_sizes, /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
      ASSERT_OK(batch_resource->RegisterInput(
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestFailedBatchResource::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
          thread_pool_.get(),
          allocator_);
  tsl::core::RefCountPtr<BatchResourceBase> batch_resource(
      new TestFailedBatchResource(batcher, queue_options, allowed_batch_sizes,
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
    EXPECT_THAT(futures[i].Await(),
                testing::status::StatusIs(absl::StatusCode::kInternal));
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  int num_requests = 3;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, kBatchDownPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4, 8};
  int num_requests = 3;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/8,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/true,
          /*disable_padding=*/false, kBatchDownPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4};
  int num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/1, /*max_batch_size=*/4,
          /*batch_timeout_micros=*/absl::ToInt64Microseconds(absl::Seconds(5)),
          /*max_enqueued_batches=*/num_requests, allowed_batch_sizes,
          /*enable_large_batch_splitting=*/false, /*disable_padding=*/false,
          kBatchDownPolicy, /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
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
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
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
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
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
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  int num_requests = 10;
  BatchResourceBase::BatcherT::QueueOptions queue_options =
      TestBatchResourceBase::GetBatcherQueueOptions(
          /*num_batch_threads=*/4, /*max_batch_size=*/GetParam().max_batch_size,
          /*batch_timeout_micros=*/
          absl::ToInt64Microseconds(absl::Seconds(3)),
          /*max_enqueued_batches=*/num_requests, GetParam().allowed_batch_sizes,
          /*enable_large_batch_splitting=*/
          GetParam().enable_large_batch_splitting,
          /*disable_padding=*/false, kPadUpPolicy,
          /*low_priority_max_batch_size=*/0,
          /*low_priority_batch_timeout_micros=*/0,
          /*low_priority_max_enqueued_batches=*/0,
          /*low_priority_allowed_batch_sizes=*/{},
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize,
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

// TODO: Add tests for batch delay metric as in
// google3/third_party/tensorflow/core/kernels/batching_util/batch_resource_base_test.cc

}  // namespace
}  // namespace jsv

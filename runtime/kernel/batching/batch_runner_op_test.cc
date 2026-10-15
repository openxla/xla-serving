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

// TODO: Use requests of various batch sizes to improve test
// coverage.
#include "runtime/kernel/batching/batch_runner_op.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/blocking_counter.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "runtime/core/batching/batch_stats.h"  // from @xla_serving
#include "runtime/core/batching/shared_batch_scheduler.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_matcher.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving
#include "runtime/kernel/batching/batch_resource_base.h"  // from @xla_serving
#include "runtime/kernel/batching/test_batch_resource_base.h"  // from @xla_serving
#include "runtime/kernel/batching/warmup.h"  // from @xla_serving
#include "runtime/kernel/identity.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving
#include "runtime/kernel/test_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/lib/monitoring/test_utils.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/status.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/common_runtime/cost_constants.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/cost_measurement.h"  // from @org_tensorflow
#include "tensorflow/core/common_runtime/cost_measurement_registry.h"  // from @org_tensorflow
#include "tsl/platform/refcount.h"  // from @tsl

namespace xla_serving {

namespace {

using ::absl_testing::StatusIs;
using ::tensorflow::CostMeasurement;
using ::tensorflow::kTpuCostName;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

constexpr absl::string_view kSimpleModelWithBatchingPath =
    "runtime/kernel/testdata/"
    "simple_batching.mlir";

constexpr char kModelName[] = "test_model";
constexpr int kModelVersion = 1;

class TestCostMeasurement : public CostMeasurement {
 public:
  using CostMeasurement::CostMeasurement;

  absl::Duration GetTotalCost() override { return absl::Milliseconds(100); }
  absl::string_view GetCostType() const override { return kTpuCostName; }
};
REGISTER_COST_MEASUREMENT(kTpuCostName, TestCostMeasurement);

class BatchRunnerOpTest : public ::testing::Test {
 protected:
  BatchRunnerOpTest() : replicated_sharding_(xla::HloSharding::Replicate()) {}

  void SetUp() override {
    stablehlo_spec_ = std::make_shared<StableHlo::Spec>(StableHlo::Spec{
        .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({2, 8}),
                    .sharding = replicated_sharding_},
                   {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({std::nullopt, 2}),
                    .sharding = replicated_sharding_}},
        .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({std::nullopt, 8}),
                     .sharding = replicated_sharding_}},
        .mlir_module_kept_var_idx = {0, 1},
    });

    assigned_devices_ = std::make_shared<const AssignedDevices>(
        AssignedDevices{.device_assignment = std::vector<int>{0, 0, 0, 0},
                        .num_replicas = 1,
                        .num_cores_per_replica = 1});
    helper_ = std::make_unique<StableHloRunnerOpTestHelper>();
    x_ = AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
    y_ = AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                              11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                             xla::ifrt::Shape({2, 8}));
    expected_out_0_ =
        AsHostBuffer<float>({19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0,
                             39.0, 46.0, 53.0, 60.0, 67.0, 74.0, 81.0, 88.0},
                            xla::ifrt::Shape({2, 8}));
    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    padding_sizes_reader_ = std::make_unique<CellReader<Histogram>>(
        "/orbax/serving/batching/padding_sizes");
    allocator_ = DefaultAllocator::GetInstance();
  }

  std::unique_ptr<StableHloRunnerOpTestHelper> helper_;
  HostBuffer x_;
  HostBuffer y_;
  HostBuffer expected_out_0_;
  std::shared_ptr<StableHlo::Spec> stablehlo_spec_;
  std::shared_ptr<const AssignedDevices> assigned_devices_;
  xla::HloSharding replicated_sharding_;
  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::unique_ptr<CellReader<Histogram>> padding_sizes_reader_;
  Allocator* allocator_;
};

TEST_F(BatchRunnerOpTest, RunOneRequestSucceeds) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<int> device_assignment = {0};
  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(helper_->RegisterVariable(y_, device_assignment,
                                         replicated_sharding_.ToProto(), "y"));
  Kernel::ExecuteContext execute_context{
      .metadata = {.model_name = kModelName, .model_version = kModelVersion},
      .inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = 5,
      .max_enqueued_batches = 1,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  EXPECT_EQ(batch_runner_op.GetNumInputs(), 1);
  EXPECT_EQ(batch_runner_op.GetNumOutputs(), 1);

  TF_ASSERT_OK(batch_runner_op.Run(execute_context));
  EXPECT_THAT(execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_))}));
  // The batch size is 2 because the zeroth dimension size of `x_` is 2.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1);
}

TEST_F(BatchRunnerOpTest, RunEmptyInputsFails) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<int> device_assignment = {0};
  TF_ASSERT_OK(helper_->RegisterVariable(y_, device_assignment,
                                         replicated_sharding_.ToProto(), "y"));

  std::vector<Value> inputs;
  Kernel::ExecuteContext execute_context{
      .metadata = {.model_name = kModelName, .model_version = kModelVersion},
      .inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = 5,
      .max_enqueued_batches = 1,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  EXPECT_THAT(batch_runner_op.Run(execute_context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Batching inputs cannot be empty.")));
}

TEST_F(BatchRunnerOpTest, RunFourRequestsSucceedsNoPadding) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4};
  size_t num_requests = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 4,
      // Use an infinite batch timeout to enforce desired batch size without
      // padding.
      .batch_timeout_micros =
          absl::ToInt64Microseconds(absl::InfiniteDuration()),
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  KernelBuilderContext identity_context;
  TF_ASSERT_OK(
      identity_context.SetConfig(kNumKernelInputs, runner->GetNumInputs()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> identity_kernel,
                          IdentityKernel::Create(identity_context));
  batched_graph.push_back(std::move(identity_kernel));
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);

  absl::Mutex mu;

  int sent_requests = 0;
  auto [all_request_sent_promise, all_request_sent_future] =
      tsl::MakePromise<>();

  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "BatchRunnerOpTest",
                               /*num_threads=*/10);
  for (int i = 0; i < num_requests; ++i) {
    pool.Schedule([&, index = i]() {
      contexts[index].inputs = absl::MakeSpan(inputs);
      contexts[index].metadata = {.model_name = kModelName,
                                  .model_version = kModelVersion};
      futures[index] = batch_runner_op.RunAsync(contexts[index]);
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
    EXPECT_THAT(contexts[i].outputs, ElementsAreArray({ValueOfType<HostBuffer>(
                                         HostBufferEq(expected_out_0_))}));
  }
  // The batch size is 4 because the allowed_batch_size is 4. so the total
  // number of processed batches is 2 and each batch holds 2 requests.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            2);
}

TEST_F(BatchRunnerOpTest, RunMultipleRequestsConcurrentlySucceeds) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8, 16};
  size_t num_requests = 5;
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 16,
      .batch_timeout_micros = 5,
      .max_enqueued_batches = num_requests,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  KernelBuilderContext identity_context;
  TF_ASSERT_OK(
      identity_context.SetConfig(kNumKernelInputs, runner->GetNumInputs()));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> identity_kernel,
                          IdentityKernel::Create(identity_context));
  batched_graph.push_back(std::move(identity_kernel));
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);

  absl::Mutex mu;

  int sent_requests = 0;
  auto [all_request_sent_promise, all_request_sent_future] =
      tsl::MakePromise<>();

  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "BatchRunnerOpTest",
                               /*num_threads=*/10);
  for (int i = 0; i < num_requests; ++i) {
    pool.Schedule([&, index = i]() {
      contexts[index].inputs = absl::MakeSpan(inputs);
      futures[index] = batch_runner_op.RunAsync(contexts[index]);
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
    EXPECT_THAT(contexts[i].outputs, ElementsAreArray({ValueOfType<HostBuffer>(
                                         HostBufferEq(expected_out_0_))}));
  }
}

TEST_F(BatchRunnerOpTest, RunMultipleRequestsFailsUnmatchedInputShapes) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<Value> inputs_0{Value(
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({1, 4})))};
  std::vector<Value> inputs_1{
      Value(AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2})))};
  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  batcher_options.num_batch_threads = 1;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  size_t num_requests = 2;
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      // Give a large enough batching timeout to enforce desired batch size.
      .batch_timeout_micros = absl::ToInt64Microseconds(absl::Seconds(5)),
      .max_enqueued_batches = num_requests,
      .allowed_batch_sizes = {1, 2, 4, 8},
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(
          std::move(batched_graph), batcher, queue_options,
          /*allowed_batch_sizes=*/{}, thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);

  contexts[0].inputs = absl::MakeSpan(inputs_0);
  futures[0] = batch_runner_op.RunAsync(contexts[0]);
  contexts[1].inputs = absl::MakeSpan(inputs_1);
  futures[1] = batch_runner_op.RunAsync(contexts[1]);

  for (int i = 0; i < num_requests; ++i) {
    EXPECT_THAT(futures[i].Await(),
                StatusIs(absl::StatusCode::kInvalidArgument));
  }
}

TEST_F(BatchRunnerOpTest, CreateFailedNoBatchedKernel) {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);

  EXPECT_THAT(
      CreateBatchRunnerOp({}, BatchRunnerOpOptions(), thread_pool.get()),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BatchRunnerOpTest, DefaultPriorityBatchingPolicy) {
  BatchRunnerOpOptions options;
  EXPECT_EQ(options.priority_batching_policy,
            PriorityBatchingPolicy::kStrictFifo);
}

TEST_F(BatchRunnerOpTest, DefaultPriorityAwareBatchingOptions) {
  BatchRunnerOpOptions options;

  EXPECT_EQ(options.priority_aware_batching_options.max_queue_depth, 0);
  EXPECT_FALSE(options.priority_aware_batching_options.enable_task_resplit);
}

TEST_F(BatchRunnerOpTest, RunFailsBatchedKernelFailsWrongInputShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<int> device_assignment = {0};
  // Wrong shape of input `x_` should fail the stablehlo runner.
  x_ = AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                           xla::ifrt::Shape({2, 4}));
  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(helper_->RegisterVariable(y_, device_assignment,
                                         replicated_sharding_.ToProto(), "y"));
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .max_enqueued_batches = 1,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  EXPECT_THAT(batch_runner_op.Run(execute_context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

// This is to test that callback is called when `RegusterInput` return non-ok
// status.
TEST_F(BatchRunnerOpTest, RunFailsBatchedKernelFailsInputHasNoDimension) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<int> device_assignment = {0};

  // Wrong shape of input `x_` should fail the stablehlo runner.
  x_ = AsHostBuffer<float>(1.0);
  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(helper_->RegisterVariable(y_, device_assignment,
                                         replicated_sharding_.ToProto(), "y"));
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {1, 2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .max_enqueued_batches = 1,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  EXPECT_THAT(batch_runner_op.Run(execute_context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(BatchRunnerOpIdentityKernelTest, RunAsyncSucceeds) {
  KernelBuilderContext context;
  TF_ASSERT_OK(context.SetConfig(kNumKernelInputs, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> identity_kernel,
                          IdentityKernel::Create(context));
  // Only allow one batch size to test that the batcher is working as expected.
  int allowed_batch_size = 4;
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = static_cast<size_t>(allowed_batch_size),
      // Disable batch timeout.
      .batch_timeout_micros =
          absl::ToInt64Microseconds(absl::InfiniteDuration()),
      .max_enqueued_batches = 20,
      .allowed_batch_sizes = {allowed_batch_size},
      .allocator = DefaultAllocator::GetInstance(),
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(identity_kernel));
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_threads",
                                      /*num_threads=*/4);
  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, {allowed_batch_size},
                                      &thread_pool));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  EXPECT_EQ(batch_runner_op.GetNumInputs(), 1);
  EXPECT_EQ(batch_runner_op.GetNumOutputs(), 1);

  // Test multiple cycles of batching.
  int num_cycles = 10;
  int num_requests = allowed_batch_size * num_cycles;
  std::vector<Kernel::ExecuteContext> execute_contexts(num_requests);
  std::vector<std::vector<Value>> inputs(num_requests);
  for (int i = 0; i < num_requests; ++i) {
    inputs[i].push_back(
        Value(AsHostBuffer<int>({i, i + 10}, xla::ifrt::Shape({1, 2}))));
    execute_contexts[i].inputs = inputs[i];
  }
  std::vector<tsl::Future<>> futures(num_requests);

  absl::BlockingCounter counter(num_requests);
  for (int i = 0; i < num_requests; ++i) {
    thread_pool.Schedule([&, &context = execute_contexts[i], i]() {
      futures[i] = batch_runner_op.RunAsync(context);
      counter.DecrementCount();
    });
  }

  // Wait until all futures have been assigned by `RunAsync`.
  counter.Wait();

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    EXPECT_THAT(execute_contexts[i].outputs,
                ElementsAre(ValueOfType<HostBuffer>(
                    HostBufferEq(inputs[i][0].Get<HostBuffer>()))));
  }
}

TEST_F(BatchRunnerOpTest, RunWithBatchDownPaddingPolicyTrimmed) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  size_t num_requests = 3;
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = absl::ToInt64Microseconds(absl::Seconds(5)),
      .allowed_batch_sizes = allowed_batch_sizes,
      .batch_padding_policy = std::string(kBatchDownPolicy),
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  KernelBuilderContext identity_context;
  TF_ASSERT_OK(
      identity_context.SetConfig(kNumKernelInputs, runner->GetNumInputs()));
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);

  absl::Mutex mu;

  int sent_requests = 0;
  auto [all_request_sent_promise, all_request_sent_future] =
      tsl::MakePromise<>();

  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "BatchRunnerOpTest",
                               /*num_threads=*/10);
  for (int i = 0; i < num_requests; ++i) {
    pool.Schedule([&, index = i]() {
      contexts[index].inputs = absl::MakeSpan(inputs);
      contexts[index].metadata = {.model_name = kModelName,
                                  .model_version = kModelVersion};
      futures[index] = batch_runner_op.RunAsync(contexts[index]);
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
    EXPECT_THAT(contexts[i].outputs, ElementsAreArray({ValueOfType<HostBuffer>(
                                         HostBufferEq(expected_out_0_))}));
  }
  // There are 3 requests with total size 6. With batch down padding policy, the
  // batch size is trimmed to the smaller allowed batch size 4. So there are 1
  // batch with batch size 4 and 1 batch with batch size 2. No padding needed.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1);
  Histogram padding_sizes_histogram_size_4 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "4");
  EXPECT_EQ(padding_sizes_histogram_size_4.sum(), 0);
  Histogram padding_sizes_histogram_size_2 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "2");
  EXPECT_EQ(padding_sizes_histogram_size_2.sum(), 0);
}

TEST_F(BatchRunnerOpTest, RunWithBatchDownPaddingPolicyNotTrimmed) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  std::vector<int> device_assignment = {0};
  std::vector<Value> inputs{Value(std::move(x_))};
  TF_ASSERT_OK(helper_->RegisterVariable(y_, device_assignment,
                                         replicated_sharding_.ToProto(), "y"));
  Kernel::ExecuteContext execute_context{
      .metadata = {.model_name = kModelName, .model_version = kModelVersion},
      .inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(), &batcher));
  std::vector<int32_t> allowed_batch_sizes = {4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = 5,
      .max_enqueued_batches = 1,
      .allowed_batch_sizes = allowed_batch_sizes,
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  EXPECT_EQ(batch_runner_op.GetNumInputs(), 1);
  EXPECT_EQ(batch_runner_op.GetNumOutputs(), 1);

  TF_ASSERT_OK(batch_runner_op.Run(execute_context));
  EXPECT_THAT(execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_))}));

  // There is 1 request of size 2. With batch down padding policy, the input
  // batch size is not trimmed because it's less than the smallest allowed batch
  // size 4. Therefore the batch size 2 is padded up (padding size is 2) to the
  // next allowed batch size 4.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  Histogram padding_sizes_histogram = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "4");
  EXPECT_EQ(padding_sizes_histogram.sum(), 2);
  // Padding size is 2 and is put in the 3rd bucket.
  EXPECT_EQ(padding_sizes_histogram.num(3), 1);
}

TEST_F(BatchRunnerOpTest, RunWithMinimizeCostPaddingPolicyToBatchDown) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = absl::ToInt64Microseconds(absl::Seconds(5)),
      .allowed_batch_sizes = allowed_batch_sizes,
      .batch_padding_policy = std::string(kMinimizeTpuCostPerRequestPolicy),
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  KernelBuilderContext identity_context;
  TF_ASSERT_OK(
      identity_context.SetConfig(kNumKernelInputs, runner->GetNumInputs()));
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  // Reset the batch stats and register batch stats for batch size 4 and 8.
  ModelBatchStats& model_stats =
      GlobalBatchStatsRegistry().model(kModelName, kModelVersion);
  CostTracker* tracker_4 = &model_stats.batch_size(4).tpu_cost();
  CostTracker* tracker_8 = &model_stats.batch_size(8).tpu_cost();
  CostTrackerTestPeer::Reset(tracker_4);
  CostTrackerTestPeer::Reset(tracker_8);
  tracker_4->Register(absl::Milliseconds(40));
  tracker_8->Register(absl::Milliseconds(120));

  size_t num_requests = 3;

  std::vector<Value> inputs{Value(std::move(x_))};

  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  absl::BlockingCounter counter(num_requests);

  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "BatchRunnerOpTest",
                               /*num_threads=*/10);
  for (int i = 0; i < num_requests; ++i) {
    pool.Schedule([&, index = i]() {
      contexts[index].inputs = absl::MakeSpan(inputs);
      contexts[index].metadata = {.model_name = kModelName,
                                  .model_version = kModelVersion};
      futures[index] = batch_runner_op.RunAsync(contexts[index]);
      counter.DecrementCount();
    });
  }
  // Wait until all futures have been assigned by `RunAsync`.
  counter.Wait();

  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    EXPECT_THAT(contexts[i].outputs, ElementsAreArray({ValueOfType<HostBuffer>(
                                         HostBufferEq(expected_out_0_))}));
  }

  // There are 3 requests with total size 6. For MINIMIZE_TPU_COST_PER_REQUEST
  // padding policy, the batch down size is 4 and the batch down cost is 40ms/4
  // = 10ms. The pad up size is 8 and the pad up cost is 120ms/6 = 20ms. Since
  // the batch down cost is less than the pad up cost, BATCH_DOWN is selected.
  // The batch size is trimmed to the smaller allowed batch size 4. So there are
  // 1 batch with batch size 4 and 1 batch with batch size 2. No padding needed.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1);
  Histogram padding_sizes_histogram_size_4 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "4");
  EXPECT_EQ(padding_sizes_histogram_size_4.sum(), 0);
  Histogram padding_sizes_histogram_size_2 = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "2");
  EXPECT_EQ(padding_sizes_histogram_size_2.sum(), 0);
}

TEST_F(BatchRunnerOpTest, RunWithMinimizeCostPaddingPolicyToPadUp) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module = stablehlo_str;

  TF_ASSERT_OK(
      helper_->RegisterVariable(y_, {0}, replicated_sharding_.ToProto(), "y"));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_, assigned_devices_));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool",
      /*num_threads=*/4);
  SharedBatchScheduler<BatchResourceBase::BatchTask>::Options batcher_options;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher;
  TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
      batcher_options, &batcher));
  std::vector<int32_t> allowed_batch_sizes = {2, 4, 8};
  BatchResourceBase::BatcherT::QueueOptions queue_options{
      .input_batch_size_limit = 8,
      .batch_timeout_micros = absl::ToInt64Microseconds(absl::Seconds(5)),
      .allowed_batch_sizes = allowed_batch_sizes,
      .batch_padding_policy = std::string(kMinimizeTpuCostPerRequestPolicy),
      .allocator = allocator_,
  };

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  KernelBuilderContext identity_context;
  TF_ASSERT_OK(
      identity_context.SetConfig(kNumKernelInputs, runner->GetNumInputs()));
  batched_graph.push_back(std::move(runner));

  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(std::move(batched_graph), batcher,
                                      queue_options, allowed_batch_sizes,
                                      thread_pool.get()));

  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  // Reset the batch stats and register batch stats for batch size 4 and 8.
  ModelBatchStats& model_stats =
      GlobalBatchStatsRegistry().model(kModelName, kModelVersion);
  CostTracker* tracker_4 = &model_stats.batch_size(4).tpu_cost();
  CostTracker* tracker_8 = &model_stats.batch_size(8).tpu_cost();
  CostTrackerTestPeer::Reset(tracker_4);
  CostTrackerTestPeer::Reset(tracker_8);
  tracker_4->Register(absl::Milliseconds(100));
  tracker_8->Register(absl::Milliseconds(100));

  size_t num_requests = 3;
  std::vector<Value> inputs{Value(std::move(x_))};
  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  absl::BlockingCounter counter(num_requests);

  // Create a thread pool after other variables are initialized to avoid
  // race.
  tsl::thread::ThreadPool pool(tsl::Env::Default(), "BatchRunnerOpTest",
                               /*num_threads=*/10);
  for (int i = 0; i < num_requests; ++i) {
    pool.Schedule([&, index = i]() {
      contexts[index].inputs = absl::MakeSpan(inputs);
      contexts[index].metadata = {.model_name = kModelName,
                                  .model_version = kModelVersion};
      futures[index] = batch_runner_op.RunAsync(contexts[index]);
      counter.DecrementCount();
    });
  }
  // Wait until all futures have been assigned by `RunAsync`.
  counter.Wait();
  for (int i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    EXPECT_THAT(contexts[i].outputs, ElementsAreArray({ValueOfType<HostBuffer>(
                                         HostBufferEq(expected_out_0_))}));
  }

  // There are 3 requests with total size 6. For MINIMIZE_TPU_COST_PER_REQUEST
  // padding policy, the batch down size is 4 and the batch down cost is 100ms/4
  // = 25ms. The pad up size is 8 and the pad up cost is 100ms/6 = 16ms. Since
  // the batch down cost is higher than the pad up cost, PAD_UP is selected.
  // Therefore the batch size 6 is padded to 8. Padding size is 2.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            1);
  Histogram padding_sizes_histogram = padding_sizes_reader_->Delta(
      kModelName, absl::StrCat(kModelVersion), "8");
  EXPECT_EQ(padding_sizes_histogram.sum(), 2);
  // Padding size is 2 and is put in the 3rd bucket.
  EXPECT_EQ(padding_sizes_histogram.num(3), 1);
}

class BatchRunnerOpWarmupTest : public ::testing::Test {
 protected:
  void SetUp() override {
    executed_batch_size_reader_ = std::make_unique<CellReader<int64_t>>(
        "/orbax/serving/batching/executed_batch_size");
    TF_ASSERT_OK(SharedBatchScheduler<BatchResourceBase::BatchTask>::Create(
        SharedBatchScheduler<BatchResourceBase::BatchTask>::Options(),
        &batcher_));
    allowed_batch_sizes_ = {1, 2, 4, 8};
    queue_options_ = {
        .input_batch_size_limit = 8,
        .batch_timeout_micros = absl::ToInt64Microseconds(absl::Seconds(1)),
        .max_enqueued_batches = 1000,
        .allowed_batch_sizes = allowed_batch_sizes_,
        .allocator = DefaultAllocator::GetInstance(),
    };
    batch_thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "batch_thread_pool",
        /*num_threads=*/4);
  }

  std::vector<std::unique_ptr<Kernel>> CreateBatchedGraph() {
    KernelBuilderContext identity_context;
    TF_CHECK_OK(identity_context.SetConfig(kNumKernelInputs, 1));
    std::unique_ptr<Kernel> identity_kernel =
        IdentityKernel::Create(identity_context).value();
    std::vector<std::unique_ptr<Kernel>> batched_graph;
    batched_graph.push_back(std::move(identity_kernel));
    return batched_graph;
  }

  std::unique_ptr<CellReader<int64_t>> executed_batch_size_reader_;
  std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>> batcher_;
  std::vector<int32_t> allowed_batch_sizes_;
  BatchResourceBase::BatcherT::QueueOptions queue_options_;
  std::unique_ptr<tsl::thread::ThreadPool> batch_thread_pool_;
};

// Verifies that when parallel warmup is enabled, warmup requests are not
// unexpectedly batched, but instead are executed separately in order to
// successfully warm up all allowed batch sizes.
TEST_F(BatchRunnerOpWarmupTest, ParallelWarmupRequestsNotBatched) {
  // Create BatchRunnerOp.
  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(CreateBatchedGraph(), batcher_,
                                      queue_options_, allowed_batch_sizes_,
                                      batch_thread_pool_.get()));
  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  // Populate testing data.
  const int num_requests = allowed_batch_sizes_.size();
  std::vector<std::vector<Value>> inputs(num_requests);
  std::vector<Kernel::ExecuteContext> contexts(num_requests);
  std::vector<tsl::Future<>> futures(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    inputs[i].push_back(
        Value(AsHostBuffer<int>(std::vector<int>(allowed_batch_sizes_[i], 0),
                                xla::ifrt::Shape({allowed_batch_sizes_[i]}))));
    contexts[i].inputs = absl::MakeSpan(inputs[i]);
    contexts[i].metadata = {.model_name = kModelName,
                            .model_version = kModelVersion};
  }
  // Register warmup state.
  TF_ASSERT_OK_AND_ASSIGN(
      auto warmup_handle,
      GetGlobalWarmupStateRegistry().Register(
          {kModelName, kModelVersion},
          std::make_unique<WarmupStateRegistry::PerModelData>()));
  // Run warmup requests.
  tsl::thread::ThreadPool request_thread_pool(tsl::Env::Default(),
                                              "request_thread_pool",
                                              /*num_threads=*/10);
  absl::BlockingCounter warmup_request_counter(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    request_thread_pool.Schedule([&, i]() {
      futures[i] = batch_runner_op.RunAsync(contexts[i]);
      warmup_request_counter.DecrementCount();
    });
  }
  warmup_request_counter.Wait();
  for (size_t i = 0; i < num_requests; ++i) {
    TF_ASSERT_OK(futures[i].Await());
    EXPECT_THAT(
        contexts[i].outputs,
        ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
            AsHostBuffer<int>(std::vector<int>(allowed_batch_sizes_[i], 0),
                              xla::ifrt::Shape({allowed_batch_sizes_[i]}))))));
  }
  // Verify the warmup requests are not batched.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "1"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            1);
  // After warmup, run a normal request.
  warmup_handle.Release();
  std::vector<Value> normal_inputs{
      Value(AsHostBuffer<int>({123}, xla::ifrt::Shape({1})))};
  Kernel::ExecuteContext normal_context;
  normal_context.inputs = absl::MakeSpan(normal_inputs);
  normal_context.metadata = {.model_name = kModelName,
                             .model_version = kModelVersion};
  tsl::Future<> normal_future = batch_runner_op.RunAsync(normal_context);
  TF_ASSERT_OK(normal_future.Await());
  EXPECT_THAT(normal_context.outputs,
              ElementsAre(ValueOfType<HostBuffer>(
                  HostBufferEq(normal_inputs[0].Get<HostBuffer>()))));
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "1"),
            1);
}

TEST_F(BatchRunnerOpWarmupTest, WarmupAllBatchSizes) {
  // Create BatchRunnerOp.
  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(CreateBatchedGraph(), batcher_,
                                      queue_options_, allowed_batch_sizes_,
                                      batch_thread_pool_.get()));
  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  // Populate testing data.
  std::vector<Value> inputs{
      Value(AsHostBuffer<int>({0}, xla::ifrt::Shape({1})))};
  Kernel::ExecuteContext context;
  context.inputs = absl::MakeSpan(inputs);
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  // Register warmup state.
  auto per_model_data = std::make_unique<WarmupStateRegistry::PerModelData>(
      WarmupStateRegistry::PerModelData{.warmup_all_batch_sizes = true});
  TF_ASSERT_OK_AND_ASSIGN(
      auto warmup_handle,
      GetGlobalWarmupStateRegistry().Register({kModelName, kModelVersion},
                                              std::move(per_model_data)));
  // Run a single warmup request.
  tsl::Future<> future = batch_runner_op.RunAsync(context);
  TF_ASSERT_OK(future.Await());
  // Verify that all batch sizes are executed.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "1"),
            2 /* auto warmup + original warmup */);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "2"),
            1 /* auto warmup */);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1 /* auto warmup */);
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "8"),
            1 /* auto warmup */);
}

// Fails the batch with size `failing_batch_size` for testing failure
// propagation from auto warmup to original warmup request.
class FailingTestXlaServingBatchResource : public XlaServingBatchResource {
 public:
  FailingTestXlaServingBatchResource(
      std::vector<std::unique_ptr<Kernel>> batched_graph,
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatchResourceBase::BatcherT::QueueOptions& queue_options,
      const std::vector<int32_t>& allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool, size_t failing_batch_size)
      : XlaServingBatchResource(std::move(batched_graph), batcher,
                                queue_options, allowed_batch_sizes,
                                thread_pool),
        failing_batch_size_(failing_batch_size) {}

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      std::vector<int32_t> batch_request_ids,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    if (!inputs.empty() && inputs[0].shape().dims()[0] == failing_batch_size_) {
      done(absl::InternalError("Intentional failure for testing."));
      return;
    }
    XlaServingBatchResource::ProcessFuncBatchImpl(
        last_task, std::move(batch_request_ids), inputs, combined_outputs,
        done);
  }

 private:
  const size_t failing_batch_size_;
};

TEST_F(BatchRunnerOpWarmupTest,
       AutoWarmupBatchFailurePropagatedToOriginalRequest) {
  // Create BatchRunnerOp.
  tsl::core::RefCountPtr<FailingTestXlaServingBatchResource> batch_resource(
      new FailingTestXlaServingBatchResource(
          CreateBatchedGraph(), batcher_, queue_options_, allowed_batch_sizes_,
          batch_thread_pool_.get(), /*failing_batch_size=*/4));
  BatchRunnerOp batch_runner_op(std::move(batch_resource));
  // Populate testing data.
  std::vector<Value> inputs{
      Value(AsHostBuffer<int>({0}, xla::ifrt::Shape({1})))};
  Kernel::ExecuteContext context;
  context.inputs = absl::MakeSpan(inputs);
  context.metadata = {.model_name = kModelName, .model_version = kModelVersion};
  // Register warmup state.
  auto per_model_data = std::make_unique<WarmupStateRegistry::PerModelData>(
      WarmupStateRegistry::PerModelData{.warmup_all_batch_sizes = true});
  TF_ASSERT_OK_AND_ASSIGN(
      auto warmup_handle,
      GetGlobalWarmupStateRegistry().Register({kModelName, kModelVersion},
                                              std::move(per_model_data)));
  // Run warmup request. The failure from the auto warmup batch (batch size 4)
  // is propagated to the original request.
  tsl::Future<> future = batch_runner_op.RunAsync(context);
  EXPECT_THAT(future.Await(), StatusIs(absl::StatusCode::kInternal,
                                       "Intentional failure for testing."));
}

class RaggedVerifierKernel : public Kernel {
 public:
  RaggedVerifierKernel(int num_inputs, int num_outputs)
      : num_inputs_(num_inputs), num_outputs_(num_outputs) {}

  std::string Name() const override { return "RaggedVerifierKernel"; }

  absl::Status Run(ExecuteContext& context) override {
    for (const auto& input : context.inputs) {
      captured_inputs.push_back(input.Get<HostBuffer>());
    }
    int batch_size = context.inputs[0].Get<HostBuffer>().shape().dims()[0];
    for (int i = 0; i < num_outputs_; ++i) {
      context.outputs.push_back(Value(HostBuffer(
          xla::ifrt::DType(xla::ifrt::DType::kF32),
          xla::ifrt::Shape({batch_size}), DefaultAllocator::GetInstance())));
    }
    return absl::OkStatus();
  }

  int GetNumInputs() const override { return num_inputs_; }
  int GetNumOutputs() const override { return num_outputs_; }
  IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  mutable std::vector<HostBuffer> captured_inputs;

 private:
  int num_inputs_;
  int num_outputs_;
};

TEST_F(BatchRunnerOpWarmupTest, RaggedBatchingVerifyConcatenationAndPadding) {
  auto verifier_kernel = std::make_unique<RaggedVerifierKernel>(3, 1);
  RaggedVerifierKernel* verifier_ptr = verifier_kernel.get();

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(verifier_kernel));

  BatchRunnerOpOptions options;
  options.allowed_batch_sizes = {8};
  options.max_batch_size = 8;
  options.batch_timeout_micros = 1000000;
  options.num_batch_threads = 1;
  options.max_enqueued_batches = 10;
  options.ragged_inputs = {{2, 1}};
  absl::SetFlag(&FLAGS_xla_serving_support_batching_for_ragged_inputs, true);

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          CreateBatchRunnerOp(std::move(batched_graph), options,
                                              batch_thread_pool_.get()));

  Kernel::ExecuteContext context1;
  std::vector<Value> inputs1 = {
      Value(AsHostBuffer<float>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<int32_t>({0, 2, 3}, xla::ifrt::Shape({3}))),
      Value(AsHostBuffer<float>({1, 1, 1, 2, 2, 2, 3, 3, 3},
                                xla::ifrt::Shape({3, 3})))};
  context1.inputs = absl::MakeSpan(inputs1);
  context1.metadata = {.model_name = kModelName,
                       .model_version = kModelVersion};
  context1.outputs.resize(1);

  Kernel::ExecuteContext context2;
  std::vector<Value> inputs2 = {
      Value(AsHostBuffer<float>({5, 6, 7, 8}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<int32_t>({0, 1, 3}, xla::ifrt::Shape({3}))),
      Value(AsHostBuffer<float>({4, 4, 4, 5, 5, 5, 6, 6, 6},
                                xla::ifrt::Shape({3, 3})))};
  context2.inputs = absl::MakeSpan(inputs2);
  context2.metadata = {.model_name = kModelName,
                       .model_version = kModelVersion};
  context2.outputs.resize(1);

  auto* async_runner = dynamic_cast<CancellableAsyncKernel*>(runner.get());
  ASSERT_NE(async_runner, nullptr);

  tsl::Future<> future1 = async_runner->RunAsync(context1);
  tsl::Future<> future2 = async_runner->RunAsync(context2);

  TF_ASSERT_OK(future1.Await());
  TF_ASSERT_OK(future2.Await());

  ASSERT_EQ(verifier_ptr->captured_inputs.size(), 3);

  const HostBuffer& dense_input = verifier_ptr->captured_inputs[0];
  std::vector<float> expected_dense = {1, 2, 3, 4, 5, 6, 7, 8,
                                       1, 2, 1, 2, 1, 2, 1, 2};
  EXPECT_THAT(dense_input, HostBufferEq(AsHostBuffer<float>(
                               expected_dense, xla::ifrt::Shape({8, 2}))));

  const HostBuffer& splits_input = verifier_ptr->captured_inputs[1];
  std::vector<int32_t> expected_splits = {0, 2, 3, 4, 6, 6, 6, 6, 6};
  EXPECT_THAT(splits_input, HostBufferEq(AsHostBuffer<int32_t>(
                                expected_splits, xla::ifrt::Shape({9}))));

  const HostBuffer& values_input = verifier_ptr->captured_inputs[2];
  std::vector<float> expected_values = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                                        4, 4, 4, 5, 5, 5, 6, 6, 6};
  EXPECT_THAT(values_input, HostBufferEq(AsHostBuffer<float>(
                                expected_values, xla::ifrt::Shape({6, 3}))));
}

TEST_F(BatchRunnerOpWarmupTest, RaggedBatchingVerifySplitting) {
  auto verifier_kernel = std::make_unique<RaggedVerifierKernel>(3, 1);
  RaggedVerifierKernel* verifier_ptr = verifier_kernel.get();

  std::vector<std::unique_ptr<Kernel>> batched_graph;
  batched_graph.push_back(std::move(verifier_kernel));

  BatchRunnerOpOptions options;
  options.allowed_batch_sizes = {4};
  options.max_batch_size = 4;
  options.batch_timeout_micros = 1000000;
  options.num_batch_threads = 1;
  options.max_enqueued_batches = 10;
  options.enable_large_batch_splitting = true;
  options.ragged_inputs = {{2, 1}};
  absl::SetFlag(&FLAGS_xla_serving_support_batching_for_ragged_inputs, true);

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          CreateBatchRunnerOp(std::move(batched_graph), options,
                                              batch_thread_pool_.get()));

  Kernel::ExecuteContext context1;
  std::vector<Value> inputs1 = {
      Value(AsHostBuffer<float>({1, 1, 2, 2, 3, 3}, xla::ifrt::Shape({3, 2}))),
      Value(AsHostBuffer<int32_t>({0, 2, 3, 5}, xla::ifrt::Shape({4}))),
      Value(AsHostBuffer<float>({1, 1, 1, 2, 2, 2, 3, 3, 3, 4, 4, 4, 5, 5, 5},
                                xla::ifrt::Shape({5, 3})))};
  context1.inputs = absl::MakeSpan(inputs1);
  context1.metadata = {.model_name = kModelName,
                       .model_version = kModelVersion};

  Kernel::ExecuteContext context2;
  std::vector<Value> inputs2 = {
      Value(AsHostBuffer<float>({4, 4, 5, 5, 6, 6}, xla::ifrt::Shape({3, 2}))),
      Value(AsHostBuffer<int32_t>({0, 1, 3, 5}, xla::ifrt::Shape({4}))),
      Value(
          AsHostBuffer<float>({6, 6, 6, 7, 7, 7, 8, 8, 8, 9, 9, 9, 10, 10, 10},
                              xla::ifrt::Shape({5, 3})))};
  context2.inputs = absl::MakeSpan(inputs2);
  context2.metadata = {.model_name = kModelName,
                       .model_version = kModelVersion};

  auto* async_runner = dynamic_cast<CancellableAsyncKernel*>(runner.get());
  ASSERT_NE(async_runner, nullptr);

  tsl::Future<> future1 = async_runner->RunAsync(context1);
  tsl::Future<> future2 = async_runner->RunAsync(context2);
  TF_ASSERT_OK(future1.Await());
  TF_ASSERT_OK(future2.Await());

  ASSERT_EQ(verifier_ptr->captured_inputs.size(), 6);

  const HostBuffer& dense_1 = verifier_ptr->captured_inputs[0];
  std::vector<float> expected_dense_1 = {1, 1, 2, 2, 3, 3, 4, 4};
  EXPECT_THAT(dense_1, HostBufferEq(AsHostBuffer<float>(
                           expected_dense_1, xla::ifrt::Shape({4, 2}))));

  const HostBuffer& splits_1 = verifier_ptr->captured_inputs[1];
  std::vector<int32_t> expected_splits_1 = {0, 2, 3, 5, 6};
  EXPECT_THAT(splits_1, HostBufferEq(AsHostBuffer<int32_t>(
                            expected_splits_1, xla::ifrt::Shape({5}))));

  const HostBuffer& values_1 = verifier_ptr->captured_inputs[2];
  std::vector<float> expected_values_1 = {1, 1, 1, 2, 2, 2, 3, 3, 3,
                                          4, 4, 4, 5, 5, 5, 6, 6, 6};
  EXPECT_THAT(values_1, HostBufferEq(AsHostBuffer<float>(
                            expected_values_1, xla::ifrt::Shape({6, 3}))));

  const HostBuffer& dense_2 = verifier_ptr->captured_inputs[3];
  std::vector<float> expected_dense_2 = {5, 5, 6, 6, 5, 5, 5, 5};
  EXPECT_THAT(dense_2, HostBufferEq(AsHostBuffer<float>(
                           expected_dense_2, xla::ifrt::Shape({4, 2}))));

  const HostBuffer& splits_2 = verifier_ptr->captured_inputs[4];
  std::vector<int32_t> expected_splits_2 = {0, 2, 4, 4, 4};
  EXPECT_THAT(splits_2, HostBufferEq(AsHostBuffer<int32_t>(
                            expected_splits_2, xla::ifrt::Shape({5}))));

  const HostBuffer& values_2 = verifier_ptr->captured_inputs[5];
  std::vector<float> expected_values_2 = {7, 7, 7, 8,  8,  8,
                                          9, 9, 9, 10, 10, 10};
  EXPECT_THAT(values_2, HostBufferEq(AsHostBuffer<float>(
                            expected_values_2, xla::ifrt::Shape({4, 3}))));
}

TEST_F(BatchRunnerOpWarmupTest, PadVariableLengthInputs) {
  queue_options_.pad_variable_length_inputs = true;
  queue_options_.enable_large_batch_splitting = false;
  queue_options_.batch_timeout_micros = 100000;
  tsl::core::RefCountPtr<TestXlaServingBatchResource> batch_resource(
      new TestXlaServingBatchResource(
          CreateBatchedGraph(), batcher_, queue_options_, allowed_batch_sizes_,
          batch_thread_pool_.get()));
  BatchRunnerOp batch_runner_op(std::move(batch_resource));

  // Input 1 has shape [1, 2] and is padded to [1, 4].
  std::vector<Value> input1{
      Value(AsHostBuffer<int>({10, 20}, xla::ifrt::Shape({1, 2})))};
  Kernel::ExecuteContext ctx1;
  ctx1.inputs = absl::MakeSpan(input1);
  ctx1.metadata = {.model_name = kModelName, .model_version = kModelVersion};

  // Input 2 has shape [2, 4] and is not padded.
  std::vector<Value> input2{Value(AsHostBuffer<int>(
      {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({2, 4})))};
  Kernel::ExecuteContext ctx2;
  ctx2.inputs = absl::MakeSpan(input2);
  ctx2.metadata = {.model_name = kModelName, .model_version = kModelVersion};

  tsl::Future<> future1 = batch_runner_op.RunAsync(ctx1);
  tsl::Future<> future2 = batch_runner_op.RunAsync(ctx2);
  TF_ASSERT_OK(future1.Await());
  TF_ASSERT_OK(future2.Await());

  EXPECT_THAT(
      ctx1.outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(
          AsHostBuffer<int>({10, 20, 0, 0}, xla::ifrt::Shape({1, 4}))))));
  EXPECT_THAT(
      ctx2.outputs,
      ElementsAre(ValueOfType<HostBuffer>(
          HostBufferEq(input2[0].Get<HostBuffer>()))));

  // Form a batch of size 3 and padded to 4.
  EXPECT_EQ(executed_batch_size_reader_->Delta(
                kModelName, absl::StrCat(kModelVersion), "4"),
            1);
  for (const char* batch_size_str : {"1", "2", "8"}) {
    EXPECT_EQ(executed_batch_size_reader_->Delta(
                  kModelName, absl::StrCat(kModelVersion), batch_size_str),
              0);
  }
}

}  // namespace
}  // namespace xla_serving

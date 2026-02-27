// TODO: Use requests of various batch sizes to improve test
// coverage.
#include "third_party/neptune/runtime/kernel/batching/batch_runner_op.h"

#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/blocking_counter.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/time/time.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/allocator.h"
#include "third_party/neptune/runtime/core/batching/batch_stats.h"
#include "third_party/neptune/runtime/core/batching/shared_batch_scheduler.h"
#include "third_party/neptune/runtime/core/default_allocator.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/stablehlo.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/neptune/runtime/kernel/batching/batch_options_utils.h"
#include "third_party/neptune/runtime/kernel/batching/batch_resource_base.h"
#include "third_party/neptune/runtime/kernel/batching/test_batch_resource_base.h"
#include "third_party/neptune/runtime/kernel/batching/warmup.h"
#include "third_party/neptune/runtime/kernel/identity.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_constants.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"
#include "third_party/neptune/runtime/kernel/test_utils.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/test_utils.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/status.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/common_runtime/cost_constants.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement.h"
#include "third_party/tensorflow/core/common_runtime/cost_measurement_registry.h"
#include "third_party/tensorflow/tsl/platform/refcount.h"

namespace jsv {

namespace {

using ::tensorflow::CostMeasurement;
using ::tensorflow::kTpuCostName;
using ::testing::ElementsAre;
using ::testing::ElementsAreArray;
using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

constexpr absl::string_view kSimpleModelWithBatchingPath =
    "google3/third_party/neptune/runtime/kernel/testdata/"
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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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

TEST_F(BatchRunnerOpTest, RunFourRequestsSucceedsNoPadding) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
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

TEST_F(BatchRunnerOpTest, RunFailsBatchedKernelFailsWrongInputShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(kSimpleModelWithBatchingPath));
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               {allowed_batch_size}, &thread_pool));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  stablehlo_spec_->mlir_module_string = stablehlo_str;

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

  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(std::move(batched_graph), batcher, queue_options,
                               allowed_batch_sizes, thread_pool.get()));

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
  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(CreateBatchedGraph(), batcher_, queue_options_,
                               allowed_batch_sizes_, batch_thread_pool_.get()));
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
  tsl::core::RefCountPtr<TestJsvBatchResource> batch_resource(
      new TestJsvBatchResource(CreateBatchedGraph(), batcher_, queue_options_,
                               allowed_batch_sizes_, batch_thread_pool_.get()));
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
class FailingTestJsvBatchResource : public JsvBatchResource {
 public:
  FailingTestJsvBatchResource(
      std::vector<std::unique_ptr<Kernel>> batched_graph,
      std::shared_ptr<SharedBatchScheduler<BatchResourceBase::BatchTask>>
          batcher,
      const BatchResourceBase::BatcherT::QueueOptions& queue_options,
      const std::vector<int32_t>& allowed_batch_sizes,
      tsl::thread::ThreadPool* thread_pool, size_t failing_batch_size)
      : JsvBatchResource(std::move(batched_graph), batcher, queue_options,
                         allowed_batch_sizes, thread_pool),
        failing_batch_size_(failing_batch_size) {}

 protected:
  void ProcessFuncBatchImpl(
      const BatchResourceBase::BatchTask& last_task,
      absl::Span<const HostBuffer> inputs,
      std::vector<HostBuffer>* combined_outputs,
      std::function<void(const absl::Status&)> done) const override {
    if (!inputs.empty() && inputs[0].shape().dims()[0] == failing_batch_size_) {
      done(absl::InternalError("Intentional failure for testing."));
      return;
    }
    JsvBatchResource::ProcessFuncBatchImpl(last_task, inputs, combined_outputs,
                                           done);
  }

 private:
  const size_t failing_batch_size_;
};

TEST_F(BatchRunnerOpWarmupTest,
       AutoWarmupBatchFailurePropagatedToOriginalRequest) {
  // Create BatchRunnerOp.
  tsl::core::RefCountPtr<FailingTestJsvBatchResource> batch_resource(
      new FailingTestJsvBatchResource(
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

}  // namespace
}  // namespace jsv

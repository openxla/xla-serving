#include "third_party/neptune/runtime/interpreter/linear_graph_execute.h"

#include <sys/stat.h>

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "third_party/neptune/runtime/interpreter/context.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {
namespace {

using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

constexpr absl::string_view kModelName = "valid_model_name";
constexpr int64_t kModelVersion = 0;

// A helper function to multiply the input host buffer by 2.
absl::StatusOr<HostBuffer> DoubleHostBuffer(const HostBuffer& host_buffer) {
  HostBuffer out_hostbuffer(host_buffer.dtype(), host_buffer.shape());
  if (host_buffer.dtype() == xla::ifrt::DType(xla::ifrt::DType::Kind::kS32)) {
    for (int i = 0; i < host_buffer.shape().num_elements(); ++i) {
      out_hostbuffer.flat<int32_t>()(i) = host_buffer.flat<int32_t>()(i) * 2;
    }
  } else if (host_buffer.dtype() ==
             xla::ifrt::DType(xla::ifrt::DType::Kind::kF32)) {
    for (int i = 0; i < host_buffer.shape().num_elements(); ++i) {
      out_hostbuffer.flat<float>()(i) = host_buffer.flat<float>()(i) * 2;
    }
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported dtype: ", host_buffer.dtype()));
  }
  return out_hostbuffer;
}

// A simple synchronous kernel that doubles the input values.
class SyncTestKernel : public CancellableKernel {
 public:
  explicit SyncTestKernel(int num_inputs,
                          absl::Duration delay = absl::ZeroDuration())
      : num_inputs_(num_inputs), delay_(delay) {}

  std::string Name() const override { return "SyncTestKernel"; }

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return num_inputs_; }

  int GetNumOutputs() const override { return num_inputs_; }

 private:
  absl::Status RunInternal(ExecuteContext& context) override {
    // Simulate work with a configurable delay.
    absl::SleepFor(delay_);

    if (context.metadata.model_name != kModelName) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid model name: ", context.metadata.model_name));
    }
    if (context.metadata.model_version != kModelVersion) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid model version: ", context.metadata.model_version));
    }
    if (context.inputs.size() != GetNumInputs()) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid number of inputs. Expected: %d, Actual: %d",
                          GetNumInputs(), context.inputs.size()));
    }
    for (const Value& input : context.inputs) {
      const HostBuffer& host_buffer = input.Get<HostBuffer>();

      TF_ASSIGN_OR_RETURN(HostBuffer out_hostbuffer,
                          DoubleHostBuffer(host_buffer));
      context.outputs.push_back(Value(std::move(out_hostbuffer)));
    }
    return absl::OkStatus();
  }

  int num_inputs_;
  absl::Duration delay_;
};

class AsyncTestKernel : public CancellableAsyncKernel {
 public:
  explicit AsyncTestKernel(int num_inputs,
                           absl::Duration delay = absl::ZeroDuration())
      : num_inputs_(num_inputs), delay_(delay) {
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "async_kernel_pool", 2);
  }

  std::string Name() const override { return "AsyncTestKernel"; }

  tsl::Future<> RunAsyncInternal(ExecuteContext& context) override {
    if (context.metadata.model_name != kModelName) {
      return tsl::Future<>(absl::InvalidArgumentError(
          absl::StrCat("Invalid model name: ", context.metadata.model_name)));
    }
    if (context.metadata.model_version != kModelVersion) {
      return tsl::Future<>(absl::InvalidArgumentError(absl::StrCat(
          "Invalid model version: ", context.metadata.model_version)));
    }

    if (context.inputs.size() != GetNumInputs()) {
      return tsl::Future<>(absl::InvalidArgumentError(
          absl::StrFormat("Invalid number of inputs. Expected: %d, Actual: %d",
                          GetNumInputs(), context.inputs.size())));
    }
    auto [promise, future] = tsl::MakePromise<>();

    thread_pool_->Schedule(
        [this, &context, promise = std::move(promise).ToShared()]() mutable {
          auto work = [&]() {
            // Simulate work with a configurable delay.
            absl::SleepFor(delay_);
            for (const Value& input : context.inputs) {
              const HostBuffer& host_buffer = input.Get<HostBuffer>();
              TF_ASSIGN_OR_RETURN(HostBuffer out_hostbuffer,
                                  DoubleHostBuffer(host_buffer));
              context.outputs.push_back(Value(std::move(out_hostbuffer)));
            }
            return absl::OkStatus();
          };
          absl::Status status = work();
          std::move(promise)->Set(status);
        });
    return future;
  }

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return num_inputs_; }

  int GetNumOutputs() const override { return num_inputs_; }

 private:
  int num_inputs_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  absl::Duration delay_;
};

class LinearGraphExecuteTest : public ::testing::Test {
 protected:
  void SetUp() override {
    graph_.push_back(std::make_unique<SyncTestKernel>(/*num_inputs=*/2));
    graph_.push_back(std::make_unique<SyncTestKernel>(/*num_inputs=*/2));
    graph_.push_back(std::make_unique<SyncTestKernel>(/*num_inputs=*/2));

    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), "test_thread_pool", 1);
  }
  std::vector<std::unique_ptr<Kernel>> graph_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(LinearGraphExecuteTest, SyncTest) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  auto in2 = AsHostBuffer<int32_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs,
                          ExecuteLinearGraph(RunContext(), inputs, graph_,
                                             kModelName, kModelVersion));
  const HostBuffer& expected_out1 = AsHostBuffer<float>(
      {1.0 * 8, 2.0 * 8, 3.0 * 8, 4.0 * 8}, xla::ifrt::Shape({2, 2}));
  const HostBuffer& expected_out2 =
      AsHostBuffer<int32_t>({7 * 8, 8 * 8, 9 * 8}, xla::ifrt::Shape({3, 1}));

  EXPECT_THAT(
      outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(expected_out1)),
                  ValueOfType<HostBuffer>(HostBufferEq(expected_out2))));
}

TEST_F(LinearGraphExecuteTest, AsyncTest) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  auto in2 = AsHostBuffer<int32_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};

  tsl::Future<std::vector<Value>> async_outputs =
      AsyncExecuteLinearGraph(RunContext(), inputs, graph_, thread_pool_.get(),
                              kModelName, kModelVersion);
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> outputs, async_outputs.Await());
  const HostBuffer& expected_out1 = AsHostBuffer<float>(
      {1.0 * 8, 2.0 * 8, 3.0 * 8, 4.0 * 8}, xla::ifrt::Shape({2, 2}));
  const HostBuffer& expected_out2 =
      AsHostBuffer<int32_t>({7 * 8, 8 * 8, 9 * 8}, xla::ifrt::Shape({3, 1}));

  EXPECT_THAT(
      outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(expected_out1)),
                  ValueOfType<HostBuffer>(HostBufferEq(expected_out2))));
}

TEST_F(LinearGraphExecuteTest, SyncPropagateFailure) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  // Invalid input type that will cause the kernel to fail.
  auto in2 = AsHostBuffer<int8_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};
  EXPECT_THAT(ExecuteLinearGraph(RunContext(), inputs, graph_, kModelName,
                                 kModelVersion)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported dtype: S8")));
}

TEST_F(LinearGraphExecuteTest, AsyncPropagateFailure) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  // Invalid input type that will cause the kernel to fail.
  auto in2 = AsHostBuffer<int8_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};

  EXPECT_THAT(
      AsyncExecuteLinearGraph(RunContext(), inputs, graph_, thread_pool_.get(),
                              kModelName, kModelVersion)
          .Await()
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Unsupported dtype: S8")));
}

TEST_F(LinearGraphExecuteTest, ModelNamePropagatedToKernel) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  // Invalid model name that will cause the kernel to fail.
  auto in2 = AsHostBuffer<int8_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};

  EXPECT_THAT(ExecuteLinearGraph(RunContext(), inputs, graph_,
                                 "invalid_model_name", kModelVersion)
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid model name: invalid_model_name")));
}

TEST_F(LinearGraphExecuteTest, ModelVersionPropagatedToKernel) {
  auto in1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  // Invalid model version that will cause the kernel to fail.
  auto in2 = AsHostBuffer<int8_t>({7, 8, 9}, xla::ifrt::Shape({3, 1}));
  const std::vector<Value> inputs = {Value(in1), Value(in2)};

  EXPECT_THAT(
      ExecuteLinearGraph(RunContext(), inputs, graph_, kModelName, 1).status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid model version: 1")));
}

TEST(AsyncExecuteLinearGraphTest, SlowKernelDoesNotBlock) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      1);
  std::vector<std::unique_ptr<Kernel>> graph1;
  // The first kernel will sleep for 5 seconds to simulate a slow execution.
  graph1.push_back(std::make_unique<AsyncTestKernel>(
      /*num_inputs=*/1, /*delay=*/absl::Seconds(5)));
  std::vector<std::unique_ptr<Kernel>> graph2;
  // The second kernel will not sleep. This is to verify that the second
  // execution does not wait for the first execution to finish.
  graph2.push_back(std::make_unique<AsyncTestKernel>(
      /*num_inputs=*/1, /*delay=*/absl::ZeroDuration()));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({0, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  const std::vector<Value> inputs2 = {
      Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  // Limit the thread pool size to 1. This is key to test that the
  // AsynExecuteLinearGraph does not block on the first execution.  The test is
  // crafted such that the second execution will finish before the first
  // execution if the first execution does not block on the only available
  // thread.
  tsl::Future<std::vector<Value>> first_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs1, graph1, &thread_pool, kModelName, kModelVersion);
  tsl::Future<std::vector<Value>> second_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs2, graph2, &thread_pool, kModelName, kModelVersion);

  // The first execution is still running, but does not block the second
  // execution.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> second_outputs,
                          second_execution.Await());
  ASSERT_FALSE(first_execution.IsReady());

  // The first execution succeeds.
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> first_outputs,
                          first_execution.Await());

  EXPECT_THAT(
      second_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {1 * 2, 2 * 2, 3 * 2, 4 * 2}, xla::ifrt::Shape({2, 2}))))));
  EXPECT_THAT(
      first_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {0 * 2, 2 * 2, 3 * 2, 4 * 2}, xla::ifrt::Shape({2, 2}))))));
}

TEST(AsyncExecuteLinearGraphTest, MultipleAsyncKernels) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      2);
  std::vector<std::unique_ptr<Kernel>> graph;
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/1));
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/1));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({5, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  const std::vector<Value> inputs2 = {
      Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  tsl::Future<std::vector<Value>> first_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs1, graph, &thread_pool, kModelName, kModelVersion);
  tsl::Future<std::vector<Value>> second_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs2, graph, &thread_pool, kModelName, kModelVersion);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> first_outputs,
                          first_execution.Await());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> second_outputs,
                          second_execution.Await());

  EXPECT_THAT(
      first_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {5 * 4, 2 * 4, 3 * 4, 4 * 4}, xla::ifrt::Shape({2, 2}))))));
  EXPECT_THAT(
      second_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {1 * 4, 2 * 4, 3 * 4, 4 * 4}, xla::ifrt::Shape({2, 2}))))));
}

TEST(AsyncExecuteLinearGraphTest, MixedSyncAndAsyncKernels) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      1);
  std::vector<std::unique_ptr<Kernel>> graph;
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/1));
  graph.push_back(std::make_unique<SyncTestKernel>(/*num_inputs=*/1));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({5, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  const std::vector<Value> inputs2 = {
      Value(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})))};
  tsl::Future<std::vector<Value>> first_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs1, graph, &thread_pool, kModelName, kModelVersion);
  tsl::Future<std::vector<Value>> second_execution = AsyncExecuteLinearGraph(
      RunContext(), inputs2, graph, &thread_pool, kModelName, kModelVersion);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> first_outputs,
                          first_execution.Await());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> second_outputs,
                          second_execution.Await());

  EXPECT_THAT(
      first_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {5 * 4, 2 * 4, 3 * 4, 4 * 4}, xla::ifrt::Shape({2, 2}))))));
  EXPECT_THAT(
      second_outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int32_t>(
          {1 * 4, 2 * 4, 3 * 4, 4 * 4}, xla::ifrt::Shape({2, 2}))))));
}

TEST(AsyncExecuteLinearGraphTest, PropagateAsyncKernelFailure) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      1);
  std::vector<std::unique_ptr<Kernel>> graph;
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/2));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({5, 2, 3, 4}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<int8_t>({6, 7, 8}, xla::ifrt::Shape({3, 1})))};
  EXPECT_THAT(AsyncExecuteLinearGraph(RunContext(), inputs1, graph,
                                      &thread_pool, kModelName, kModelVersion)
                  .Await()
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unsupported dtype: S8")));
}

TEST(AsyncExecuteLinearGraphTest, PropagateModelNameToKernel) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      1);
  std::vector<std::unique_ptr<Kernel>> graph;
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/2));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({5, 2, 3, 4}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<int32_t>({6, 7, 8}, xla::ifrt::Shape({3, 1})))};

  EXPECT_THAT(
      AsyncExecuteLinearGraph(RunContext(), inputs1, graph, &thread_pool,
                              "invalid_model_name", kModelVersion)
          .Await()
          .status(),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Invalid model name: invalid_model_name")));
}

TEST(AsyncExecuteLinearGraphTest, PropagateModelVersionToKernel) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      1);
  std::vector<std::unique_ptr<Kernel>> graph;
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/2));

  const std::vector<Value> inputs1 = {
      Value(AsHostBuffer<int32_t>({5, 2, 3, 4}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<int32_t>({6, 7, 8}, xla::ifrt::Shape({3, 1})))};

  EXPECT_THAT(AsyncExecuteLinearGraph(RunContext(), inputs1, graph,
                                      &thread_pool, kModelName, 1)
                  .Await()
                  .status(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid model version: 1")));
}

TEST_F(LinearGraphExecuteTest, SyncDeadlineExceeded) {
  std::vector<std::unique_ptr<Kernel>> graph;
  // Create a SyncTestKernel with a delay longer than the deadline.
  graph.push_back(std::make_unique<SyncTestKernel>(/*num_inputs=*/1,
                                                   absl::Milliseconds(500)));
  graph.push_back(
      std::make_unique<SyncTestKernel>(/*num_inputs=*/1, absl::ZeroDuration()));

  auto in = AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}));
  const std::vector<Value> inputs = {Value(in)};

  // Set a deadline in the near future (e.g., 1 second from now).
  absl::Time deadline = absl::Now() + absl::Milliseconds(100);

  // The execution should time out because the kernel's delay (500ms) exceeds
  // the deadline (100ms).
  EXPECT_THAT(ExecuteLinearGraph(
                  RunContext{.cancellation_context =
                                 CancellationContext{.deadline = deadline}},
                  inputs, graph, kModelName, kModelVersion)
                  .status(),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST_F(LinearGraphExecuteTest, AsyncDeadlineExceeded) {
  std::vector<std::unique_ptr<Kernel>> graph;
  // Create an AsyncTestKernel with a delay longer than the deadline.
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/1,
                                                    absl::Milliseconds(500)));
  graph.push_back(std::make_unique<AsyncTestKernel>(/*num_inputs=*/1,
                                                    absl::ZeroDuration()));

  auto in = AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1}));
  const std::vector<Value> inputs = {Value(in)};

  // Set a deadline in the near future (e.g., 1 second from now).
  absl::Time deadline = absl::Now() + absl::Milliseconds(100);

  // The execution should time out because the kernel's delay (500ms) exceeds
  // the deadline (100ms).
  tsl::Future<std::vector<Value>> async_outputs = AsyncExecuteLinearGraph(
      RunContext{.cancellation_context =
                     CancellationContext{.deadline = deadline}},
      inputs, graph, thread_pool_.get(), kModelName, kModelVersion);

  // The error message comes from the framework timing out, not the kernel.
  EXPECT_THAT(async_outputs.Await().status(),
              StatusIs(absl::StatusCode::kCancelled));
}

}  // namespace
}  // namespace jsv

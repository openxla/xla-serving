#include "third_party/neptune/runtime/kernel/kernel.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_constants.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {
namespace {

using ::testing::ElementsAre;
using ::testing::status::StatusIs;

class TestKernel : public CancellableKernel {
 public:
  TestKernel() = default;
  ~TestKernel() override = default;
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context) {
    TF_ASSIGN_OR_RETURN(int test_config, context.GetConfig<int>("test_config"));
    if (test_config != 100) {
      return absl::InvalidArgumentError("test_config should be 100");
    }
    return std::make_unique<TestKernel>();
  }

  std::string Name() const override { return "TestKernel"; }

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return 1; }

  int GetNumOutputs() const override { return 0; }

 private:
  absl::Status RunInternal(ExecuteContext& context) override {
    if (context.inputs.size() != GetNumInputs()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected exactly ", GetNumInputs(), " input, but got ",
                       context.inputs.size()));
    }
    return absl::OkStatus();
  }
};

TEST(KernelRegistryTest, Basic) {
  KernelRegistry registry;
  TF_EXPECT_OK(registry.Register("test_kernel", TestKernel::Create));
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig("test_config", 100));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> test_kernel,
                          registry.Create("test_kernel", builder_context));
  Kernel::ExecuteContext execute_context;
  std::vector<Value> inputs = {Value(AsHostBuffer<int>(1))};
  execute_context.inputs = inputs;
  TF_EXPECT_OK(test_kernel->Run(execute_context));
}

TEST(KernelRegistryTest, DuplicateRegistrationShallFail) {
  KernelRegistry registry;
  TF_EXPECT_OK(registry.Register("test_kernel", TestKernel::Create));
  EXPECT_THAT(registry.Register("test_kernel", TestKernel::Create),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(KernelRegistryTest, InvalidConfigShallFail) {
  KernelRegistry registry;
  TF_EXPECT_OK(registry.Register("test_kernel", TestKernel::Create));
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig("test_config", 10));
  EXPECT_THAT(registry.Create("test_kernel", builder_context),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(KernelTest, RunReturnsCancelledWhenCancelled) {
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig("test_config", 100));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> test_kernel,
                          TestKernel::Create(builder_context));

  Kernel::ExecuteContext execute_context;
  // Set a deadline in the past to simulate a cancelled state.
  execute_context.run_context.cancellation_context.deadline =
      absl::Now() - absl::Seconds(1);

  EXPECT_THAT(test_kernel->Run(execute_context),
              StatusIs(absl::StatusCode::kCancelled));
}

class TestAsyncKernel : public CancellableAsyncKernel {
 public:
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context) {
    TF_ASSIGN_OR_RETURN(int num_host_buffers,
                        context.GetConfig<int>(kNumKernelInputs));
    if (num_host_buffers <= 0) {
      return absl::InvalidArgumentError(
          "Number of host buffers must be positive.");
    }
    return absl::WrapUnique(new TestAsyncKernel(num_host_buffers));
  }

  std::string Name() const override { return "TestAsyncKernel"; }

  Kernel::IODataTypes GetIODataTypes() const override {
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return num_host_buffers_; }

  int GetNumOutputs() const override { return num_host_buffers_; }

 private:
  explicit TestAsyncKernel(int num_host_buffers)
      : num_host_buffers_(num_host_buffers) {}

  tsl::Future<> RunAsyncInternal(ExecuteContext& context) override {
    if (context.inputs.size() != GetNumInputs()) {
      return tsl::Future<>(absl::InvalidArgumentError(
          absl::StrCat("Expected exactly ", GetNumInputs(), " input, but got ",
                       context.inputs.size())));
    }

    auto [promise, future] = tsl::MakePromise<>();

    thread_pool_.Schedule(
        [&context, promise = std::move(promise).ToShared()]() mutable {
          // Sleep for 1 second to simulate a long running kernel.
          absl::SleepFor(absl::Seconds(1));

          context.outputs.clear();
          context.outputs.reserve(context.inputs.size());
          for (const Value& input : context.inputs) {
            context.outputs.push_back(input);
          }
          promise->Set();
        });
    return future;
  }

  tsl::thread::ThreadPool thread_pool_ = tsl::thread::ThreadPool(
      tsl::Env::Default(), "test_threads", /*num_threads=*/4);

  int num_host_buffers_;
};

TEST(AsyncKernelTest, RunAsyncSucceeds) {
  Kernel::ExecuteContext execute_context;
  std::vector<Value> inputs = {Value(AsHostBuffer<int>(1)),
                               Value(AsHostBuffer<int>(2))};
  execute_context.inputs = inputs;
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 2));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));
  EXPECT_TRUE(kernel->CanRunAsync());
  CancellableAsyncKernel* async_kernel =
      dynamic_cast<CancellableAsyncKernel*>(kernel.get());
  ASSERT_NE(async_kernel, nullptr);
  tsl::Future<> output_future = async_kernel->RunAsync(execute_context);
  // Too quick to check the output.
  EXPECT_FALSE(output_future.IsKnownReady());
  EXPECT_OK(output_future.Await());
  EXPECT_THAT(
      execute_context.outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int>(1))),
                  ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int>(2)))));
}

TEST(AsyncKernelTest, MultipleRunAsyncSucceeds) {
  int num_tasks = 10;
  std::vector<Kernel::ExecuteContext> execute_contexts(num_tasks);
  std::vector<std::vector<Value>> inputs(num_tasks);
  for (int i = 0; i < num_tasks; ++i) {
    inputs[i].push_back(Value(AsHostBuffer<int>(i)));
    execute_contexts[i].inputs = inputs[i];
  }
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));
  CancellableAsyncKernel* async_kernel =
      dynamic_cast<CancellableAsyncKernel*>(kernel.get());
  ASSERT_NE(async_kernel, nullptr);
  std::vector<tsl::Future<>> output_futures;

  for (int i = 0; i < num_tasks; ++i) {
    output_futures.push_back(async_kernel->RunAsync(execute_contexts[i]));
    // Too quick to check the output.
    EXPECT_FALSE(output_futures[i].IsKnownReady());
  }

  for (int i = 0; i < num_tasks; ++i) {
    EXPECT_OK(output_futures[i].Await());
    EXPECT_THAT(execute_contexts[i].outputs,
                ElementsAre(ValueOfType<HostBuffer>(
                    HostBufferEq(AsHostBuffer<int>(i)))));
  }
}

TEST(AsyncKernelTest, RunAsyncFails) {
  Kernel::ExecuteContext execute_context;
  std::vector<Value> inputs = {};
  execute_context.inputs = inputs;
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));
  CancellableAsyncKernel* async_kernel =
      dynamic_cast<CancellableAsyncKernel*>(kernel.get());
  ASSERT_NE(async_kernel, nullptr);
  tsl::Future<> output_future = async_kernel->RunAsync(execute_context);
  EXPECT_TRUE(output_future.IsKnownReady());
  EXPECT_THAT(output_future.Await(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(AsyncKernelTest, RunSucceeds) {
  Kernel::ExecuteContext execute_context;
  std::vector<Value> inputs = {Value(AsHostBuffer<int>(1)),
                               Value(AsHostBuffer<int>(2))};
  execute_context.inputs = inputs;
  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 2));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));
  TF_ASSERT_OK(kernel->Run(execute_context));
  EXPECT_THAT(
      execute_context.outputs,
      ElementsAre(ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int>(1))),
                  ValueOfType<HostBuffer>(HostBufferEq(AsHostBuffer<int>(2)))));
}

TEST(AsyncKernelTest, RunReturnsCancelledWhenCancelled) {
  Kernel::ExecuteContext execute_context;
  // Set a deadline in the past to simulate a cancelled state.
  execute_context.run_context.cancellation_context.deadline =
      absl::Now() - absl::Seconds(1);

  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));

  EXPECT_THAT(kernel->Run(execute_context),
              StatusIs(absl::StatusCode::kCancelled));
}

TEST(AsyncKernelTest, RunAsyncReturnsCancelledWhenCancelled) {
  Kernel::ExecuteContext execute_context;
  // Set a deadline in the past to simulate a cancelled state.
  execute_context.run_context.cancellation_context.deadline =
      absl::Now() - absl::Seconds(1);
  std::vector<Value> inputs = {Value(AsHostBuffer<int>(1))};
  execute_context.inputs = inputs;

  KernelBuilderContext builder_context;
  TF_ASSERT_OK(builder_context.SetConfig(kNumKernelInputs, 1));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> kernel,
                          TestAsyncKernel::Create(builder_context));
  CancellableAsyncKernel* async_kernel =
      dynamic_cast<CancellableAsyncKernel*>(kernel.get());
  ASSERT_NE(async_kernel, nullptr);

  tsl::Future<> output_future = async_kernel->RunAsync(execute_context);
  EXPECT_TRUE(output_future.IsKnownReady());
  EXPECT_THAT(output_future.Await(), StatusIs(absl::StatusCode::kCancelled));
}

}  // namespace
}  // namespace jsv

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

#include "runtime/kernel/kernel.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_matcher.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

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
}  // namespace xla_serving

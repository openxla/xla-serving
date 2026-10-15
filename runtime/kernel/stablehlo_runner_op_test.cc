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

#include "runtime/kernel/stablehlo_runner_op.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/aot_stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/core/value_matcher.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/test_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tests/test_utils.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/file_system.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

class StableHloRunnerTestPeer {
 public:
  static xla::ifrt::LoadedExecutableRef GetIfrtExecutable(
      StableHloRunnerBase* runner,
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes) {
    auto future = runner->LookUpOrCreateExecutable(input_shapes, input_dtypes);
    auto bundle = future.Await().value();
    return bundle->ifrt_executable;
  }

  static xla::ifrt::LoadedExecutableRef GetIfrtExecutable(
      Kernel* kernel, absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes) {
    auto* runner_op = static_cast<StableHloRunnerOp*>(kernel);
    return GetIfrtExecutable(runner_op->stablehlo_runner_.get(), input_shapes,
                             input_dtypes);
  }

  static const std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>>&
  GetAotMemoryRegions(Kernel* kernel) {
    auto* runner_op = static_cast<StableHloRunnerOp*>(kernel);
    auto* aot_runner =
        static_cast<AotStableHloRunner*>(runner_op->stablehlo_runner_.get());
    return aot_runner->aot_memory_regions_;
  }
};

namespace {
using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

class TestMemoryRegion : public tsl::ReadOnlyMemoryRegion {
 public:
  const void* data() override { return nullptr; }
  uint64_t length() override { return 0; }
};

class SimpleMlirStableHloRunnerOpTest : public ::testing::Test {
 protected:
  SimpleMlirStableHloRunnerOpTest()
      : replicated_sharding_(xla::HloSharding::Replicate()),
        other_sharding_(xla::HloSharding::Unknown()) {
    helper_ = std::make_unique<StableHloRunnerOpTestHelper>();

    stablehlo_spec_simple_ = std::make_shared<StableHlo::Spec>(StableHlo::Spec{
        .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({2, 2}),
                    .sharding = replicated_sharding_},
                   {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({2, 8}),
                    .sharding = replicated_sharding_}},
        .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({2, 8}),
                     .sharding = replicated_sharding_},
                    {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({2, 8}),
                     .sharding = replicated_sharding_}},
        .mlir_module_kept_var_idx = {0, 1}});
    assigned_devices_simple_ = std::make_shared<const AssignedDevices>(
        AssignedDevices{.device_assignment = std::vector<int>{0, 0, 0, 0},
                        .num_replicas = 1,
                        .num_cores_per_replica = 1});

    const std::string kSharding = R"pb(
      type: OTHER
      tile_assignment_dimensions: 2
      tile_assignment_dimensions: 1
      iota_reshape_dims: 2
      iota_transpose_perm: 0
    )pb";
    other_sharding_ = *xla::HloSharding::FromProto(
        *xla::ParseTextProto<xla::OpSharding>(kSharding));
    stablehlo_spec_spmd_ = std::make_shared<StableHlo::Spec>(StableHlo::Spec{
        .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({2, 2}),
                    .sharding = other_sharding_},
                   {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({2, 8}),
                    .sharding = other_sharding_}},
        .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({2, 8}),
                     .sharding = replicated_sharding_},
                    {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({2, 8}),
                     .sharding = replicated_sharding_}},
        .mlir_module_kept_var_idx = {0, 1}});
    assigned_devices_spmd_ =
        std::make_shared<const AssignedDevices>(AssignedDevices{
            .device_assignment = std::vector<int>{0, 0, 0, 0, 0, 0, 0, 1},
            .num_replicas = 1,
            .num_cores_per_replica = 2});

    x_ = AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
    y_ = AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                              11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                             xla::ifrt::Shape({2, 8}));
    expected_output_count_ = 2;
    expected_out_0_ =
        AsHostBuffer<float>({19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0,
                             39.0, 46.0, 53.0, 60.0, 67.0, 74.0, 81.0, 88.0},
                            xla::ifrt::Shape({2, 8}));
    expected_out_1_ =
        AsHostBuffer<float>({18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0,
                             38.0, 45.0, 52.0, 59.0, 66.0, 73.0, 80.0, 87.0},
                            xla::ifrt::Shape({2, 8}));
  }

  std::unique_ptr<StableHloRunnerOpTestHelper> helper_;
  std::shared_ptr<StableHlo::Spec> stablehlo_spec_simple_;
  std::shared_ptr<const AssignedDevices> assigned_devices_simple_;
  std::shared_ptr<StableHlo::Spec> stablehlo_spec_spmd_;
  std::shared_ptr<const AssignedDevices> assigned_devices_spmd_;
  xla::HloSharding replicated_sharding_;
  xla::HloSharding other_sharding_;
  HostBuffer x_;
  HostBuffer y_;
  int expected_output_count_;
  HostBuffer expected_out_0_;
  HostBuffer expected_out_1_;
};

TEST_F(SimpleMlirStableHloRunnerOpTest, Basic) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));

  std::vector<Value> inputs{Value(std::move(x_)), Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  TF_ASSERT_OK(runner->Run(execute_context));

  EXPECT_THAT(execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_)),
                   ValueOfType<HostBuffer>(HostBufferEq(expected_out_1_))}));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, Spmd) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/spmd.mlir"));
  stablehlo_spec_spmd_->mlir_module = stablehlo_str;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_spmd_, assigned_devices_spmd_));

  std::vector<Value> inputs{Value(std::move(x_)), Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  for (int i = 0; i < 10; ++i) {
    // Run the kernel multiple times to ensure the kernel is idempotent.
    TF_ASSERT_OK(runner->Run(execute_context));
  }

  EXPECT_THAT(execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_)),
                   ValueOfType<HostBuffer>(HostBufferEq(expected_out_1_))}));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, HasVariable) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/spmd.mlir"));
  stablehlo_spec_spmd_->mlir_module = stablehlo_str;

  std::vector<int> device_ids = {0, 1};
  const std::string kVariableName = "variable";
  TF_ASSERT_OK(helper_->RegisterVariable(
      x_, device_ids, other_sharding_.ToProto(), kVariableName));

  std::vector<Value> inputs{Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_spmd_, assigned_devices_spmd_));
  // Implicitly, the variable is the first input.
  EXPECT_EQ(runner->GetNumInputs(), 1);
  EXPECT_EQ(runner->GetNumOutputs(), 2);
  TF_ASSERT_OK(runner->Run(execute_context));

  EXPECT_THAT(execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_)),
                   ValueOfType<HostBuffer>(HostBufferEq(expected_out_1_))}));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, InputShapeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  HostBuffer input_0 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({1, 4}));
  HostBuffer input_1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));
  std::vector<Value> inputs{Value(std::move(input_0)),
                            Value(std::move(input_1))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));
  EXPECT_THAT(
      runner->Run(execute_context),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("No MLIR module found for input shapes [1,4],[2,8]")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, VariableDTypeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  std::vector<int> device_ids = {0};
  const std::string kVariableName = "variable";
  HostBuffer variable =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  TF_ASSERT_OK(helper_->RegisterVariable(
      variable, device_ids, replicated_sharding_.ToProto(), kVariableName));

  std::vector<Value> inputs{Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));

  EXPECT_THAT(runner->Run(execute_context),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("expected weight #0 'variable' to have "
                                 "dtype: F32, but got S32")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, VariableShapeMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  std::vector<int> device_ids = {0};
  const std::string kVariableName = "variable";
  HostBuffer variable =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({1, 4}));
  TF_ASSERT_OK(helper_->RegisterVariable(
      variable, device_ids, replicated_sharding_.ToProto(), kVariableName));

  std::vector<Value> inputs{Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));

  EXPECT_THAT(
      runner->Run(execute_context),
      StatusIs(
          absl::StatusCode::kFailedPrecondition,
          HasSubstr("StableHlo expected weight #0 (variable) to have shape: "
                    "[2,2], but got [1,4]")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, VariableShardingMismatch) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  std::vector<int> device_ids = {0, 1};
  xla::OpSharding sharding = xla::HloSharding::IotaTile({1, 2}).ToProto();
  const std::string kVariableName = "variable";
  HostBuffer variable =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({1, 4}));
  TF_ASSERT_OK(
      helper_->RegisterVariable(variable, device_ids, sharding, kVariableName));

  std::vector<Value> inputs{Value(std::move(y_))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));

  EXPECT_THAT(runner->Run(execute_context),
              StatusIs(absl::StatusCode::kNotFound,
                       HasSubstr("No loaded variable found")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, AotBasic) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<Kernel> standard_runner,
      helper_->MakeRunnerOp(stablehlo_spec_simple_, assigned_devices_simple_));

  HostBuffer x =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  HostBuffer y =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));

  std::vector<Value> inputs{Value(std::move(x)), Value(std::move(y))};
  Kernel::ExecuteContext execute_context{.inputs = absl::MakeSpan(inputs)};
  TF_ASSERT_OK(standard_runner->Run(execute_context));

  std::vector<xla::ifrt::Shape> input_shapes = {xla::ifrt::Shape({2, 2}),
                                                xla::ifrt::Shape({2, 8})};
  std::vector<xla::ifrt::DType> input_dtypes = {
      xla::ifrt::DType(xla::ifrt::DType::kF32),
      xla::ifrt::DType(xla::ifrt::DType::kF32)};

  xla::ifrt::LoadedExecutableRef executable_ref =
      StableHloRunnerTestPeer::GetIfrtExecutable(standard_runner.get(),
                                                 input_shapes, input_dtypes);

  ASSERT_NE(executable_ref, nullptr);

  KernelBuilderContext builder_context;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<StableHlo> stablehlo,
                          StableHlo::Create({*stablehlo_spec_simple_}));
  TF_ASSERT_OK(builder_context.SetConfig(kStableHlo, stablehlo));

  TF_ASSERT_OK(builder_context.SetConfig(
      kIfrtLoadedExecutables,
      std::vector<xla::ifrt::LoadedExecutableRef>{executable_ref}));

  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtClient, helper_->GetIfrtClient()));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtShardingTest", 16);

  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtWorkerThreadPool, thread_pool.get()));

  IfrtLoadedVariableRegistry* registry = helper_->GetLoadedVariableRegistry();

  HostBuffer x_var =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  std::vector<int> device_ids = {0};
  TF_ASSERT_OK(helper_->RegisterVariable(
      x_var, device_ids, replicated_sharding_.ToProto(), "variable"));

  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtLoadedVariableRegistry, registry));

  std::vector<std::string> weights = {"variable"};
  TF_ASSERT_OK(builder_context.SetConfig(kWeights, weights));

  TF_ASSERT_OK(
      builder_context.SetConfig(kAssignedDevices, assigned_devices_simple_));

  std::vector<std::shared_ptr<tsl::ReadOnlyMemoryRegion>> test_memory_regions;
  test_memory_regions.push_back(std::make_shared<TestMemoryRegion>());
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtAotMemoryRegions, test_memory_regions));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> aot_runner,
                          StableHloRunnerOp::Create(builder_context));

  EXPECT_EQ(
      StableHloRunnerTestPeer::GetAotMemoryRegions(aot_runner.get()).size(), 1);
  EXPECT_EQ(StableHloRunnerTestPeer::GetAotMemoryRegions(aot_runner.get())[0],
            test_memory_regions[0]);

  HostBuffer y_aot =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));

  std::vector<Value> aot_inputs{Value(std::move(y_aot))};
  Kernel::ExecuteContext aot_execute_context{.inputs =
                                                 absl::MakeSpan(aot_inputs)};
  TF_ASSERT_OK(aot_runner->Run(aot_execute_context));

  EXPECT_THAT(aot_execute_context.outputs,
              ElementsAreArray(
                  {ValueOfType<HostBuffer>(HostBufferEq(expected_out_0_)),
                   ValueOfType<HostBuffer>(HostBufferEq(expected_out_1_))}));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, AotMemoryRegionsConfigError) {
  KernelBuilderContext builder_context;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<StableHlo> stablehlo,
                          StableHlo::Create({*stablehlo_spec_simple_}));
  TF_ASSERT_OK(builder_context.SetConfig(kStableHlo, stablehlo));
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtClient, helper_->GetIfrtClient()));
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtShardingTest", 16);
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtWorkerThreadPool, thread_pool.get()));
  IfrtLoadedVariableRegistry* registry = helper_->GetLoadedVariableRegistry();
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtLoadedVariableRegistry, registry));
  std::vector<std::string> weights = {};
  TF_ASSERT_OK(builder_context.SetConfig(kWeights, weights));
  TF_ASSERT_OK(
      builder_context.SetConfig(kAssignedDevices, assigned_devices_simple_));

  std::vector<xla::ifrt::LoadedExecutableRef> test_executables;
  TF_ASSERT_OK(builder_context.SetConfig(kIfrtLoadedExecutables,
                                         std::move(test_executables)));

  // Inject a mismatched type (int) to trigger an InvalidArgumentError
  TF_ASSERT_OK(builder_context.SetConfig(kIfrtAotMemoryRegions, 42));

  EXPECT_THAT(StableHloRunnerOp::Create(builder_context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ifrt_aot_memory_regions")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, LoadedExecutablesConfigError) {
  KernelBuilderContext builder_context;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<StableHlo> stablehlo,
                          StableHlo::Create({*stablehlo_spec_simple_}));
  TF_ASSERT_OK(builder_context.SetConfig(kStableHlo, stablehlo));
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtClient, helper_->GetIfrtClient()));
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtShardingTest", 16);
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtWorkerThreadPool, thread_pool.get()));
  IfrtLoadedVariableRegistry* registry = helper_->GetLoadedVariableRegistry();
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtLoadedVariableRegistry, registry));
  std::vector<std::string> weights = {};
  TF_ASSERT_OK(builder_context.SetConfig(kWeights, weights));
  TF_ASSERT_OK(
      builder_context.SetConfig(kAssignedDevices, assigned_devices_simple_));

  // Inject a mismatched type (int) to trigger an InvalidArgumentError
  TF_ASSERT_OK(builder_context.SetConfig(kIfrtLoadedExecutables, 42));

  EXPECT_THAT(StableHloRunnerOp::Create(builder_context),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("ifrt_loaded_executables")));
}

TEST_F(SimpleMlirStableHloRunnerOpTest, JitFallbackOptimizerOmitted) {
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string stablehlo_str,
      helper_->GetStableHloStrFromDataDependency(
          "runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module = stablehlo_str;

  KernelBuilderContext builder_context;
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<StableHlo> stablehlo,
                          StableHlo::Create({*stablehlo_spec_simple_}));
  TF_ASSERT_OK(builder_context.SetConfig(kStableHlo, stablehlo));
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtClient, helper_->GetIfrtClient()));
  TF_ASSERT_OK(builder_context.SetConfig(
      kPersistentCompilationCache,
      static_cast<xla_serving::IfrtPersistentCompilationCache*>(nullptr)));
  TF_ASSERT_OK(builder_context.SetConfig(
      kCompilationEnvProto, static_cast<tsl::protobuf::Message*>(nullptr)));

  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtShardingTest", 16);
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtWorkerThreadPool, thread_pool.get()));
  IfrtLoadedVariableRegistry* registry = helper_->GetLoadedVariableRegistry();
  TF_ASSERT_OK(
      builder_context.SetConfig(kIfrtLoadedVariableRegistry, registry));
  std::vector<std::string> weights = {};
  TF_ASSERT_OK(builder_context.SetConfig(kWeights, weights));
  TF_ASSERT_OK(
      builder_context.SetConfig(kAssignedDevices, assigned_devices_simple_));

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<Kernel> op,
                          StableHloRunnerOp::Create(builder_context));
  EXPECT_NE(op, nullptr);
  EXPECT_EQ(op->Name(), "StableHloRunnerOp");
}

}  // namespace
}  // namespace xla_serving

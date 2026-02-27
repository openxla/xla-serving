#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/stablehlo.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_matcher.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/test_utils.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/tests/test_utils.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {
namespace {
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::status::StatusIs;

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
          "google3/third_party/neptune/runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module_string = stablehlo_str;
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
      helper_->GetStableHloStrFromDataDependency("google3/third_party/neptune/"
                                                 "runtime/kernel/testdata/"
                                                 "spmd.mlir"));
  stablehlo_spec_spmd_->mlir_module_string = stablehlo_str;

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
          "google3/third_party/neptune/runtime/kernel/testdata/spmd.mlir"));
  stablehlo_spec_spmd_->mlir_module_string = stablehlo_str;

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
          "google3/third_party/neptune/runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module_string = stablehlo_str;

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
          "google3/third_party/neptune/runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module_string = stablehlo_str;

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
          "google3/third_party/neptune/runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module_string = stablehlo_str;

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
          "google3/third_party/neptune/runtime/kernel/testdata/simple.mlir"));
  stablehlo_spec_simple_->mlir_module_string = stablehlo_str;

  std::vector<int> device_ids = {0};
  xla::OpSharding sharding = xla::HloSharding::IotaTile({2, 2}).ToProto();
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

}  // namespace
}  // namespace jsv

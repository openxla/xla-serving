#include "third_party/neptune/runtime/core/stablehlo_runner.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "testing/base/public/benchmark.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/neptune/runtime/core/ifrt_loaded_variable_registry.h"
#include "third_party/neptune/runtime/core/sharding_utils.h"
#include "third_party/neptune/runtime/core/stablehlo.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/core/value_utils.h"
#include "third_party/neptune/runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/tile_assignment.h"
#include "third_party/tensorflow/compiler/xla/pjrt/proto/compile_options.proto.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/framework/serving_device_selector.h"
#include "third_party/tensorflow/compiler/xla/tsl/framework/test_util/mock_serving_device_selector.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/monitoring/cell_reader.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

namespace jsv {

class StableHloRunnerTestPeer {
 public:
  static absl::StatusOr<std::string> GetProgramFingerprint(
      const StableHloRunner& runner) {
    std::vector<std::string> program_fingerprints;
    absl::MutexLock lock(runner.mutex_);
    for (const auto& [key, future] : runner.executable_bundles_) {
      program_fingerprints.push_back(
          future.Await().value()->program_fingerprint_with_timestamp);
    }
    if (program_fingerprints.size() != 1) {
      return absl::InternalError(
          absl::StrCat("Expected 1 program fingerprint, but got ",
                       program_fingerprints.size()));
    }
    return program_fingerprints[0];
  }
};

namespace {

using ::testing::_;
using ::testing::HasSubstr;
using ::testing::TestWithParam;
using ::testing::status::StatusIs;
using ::tsl::monitoring::testing::CellReader;

constexpr absl::string_view kSimpleModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/simple.mlir";
constexpr absl::string_view kLinearMultiPlatformModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/linear_multi_platform.mlir";
constexpr absl::string_view kSimpleDynamicShapeModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/simple_with_dynamic_shape.mlir";
constexpr absl::string_view kSimpleFalseDynamicShapeAttrModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/simple_false_dynamic_shape_attr.mlir";
constexpr absl::string_view kTupleArgsModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/tuple_args.mlir";
constexpr absl::string_view kTokenArgsModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/token_args.mlir";
constexpr absl::string_view kSpmdShardyModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/spmd_shardy.mlir";
constexpr absl::string_view kSpmdShardyCheckShardingPropagationModelPath =
    "google3/third_party/neptune/runtime/core/"
    "testdata/spmd_shardy_propagation.mlir";

struct StableHloRunnerCreateTestParams {
  std::string test_name;
  absl::string_view model_path;
  StableHlo::Spec stablehlo_spec;
  std::shared_ptr<const AssignedDevices> assigned_devices;
  absl::Status expected_status;
};

using StableHloRunnerCreateTest =
    TestWithParam<StableHloRunnerCreateTestParams>;

TEST_P(StableHloRunnerCreateTest, RunnerCreateFailure) {
  const StableHloRunnerCreateTestParams& params = GetParam();
  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(params.model_path);
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  StableHlo::Spec stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module_string = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  EXPECT_THAT(
      StableHloRunner::Create(
          std::move(stablehlo),
          /*compile_options=*/std::nullopt, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, std::make_shared<const AssignedDevices>(),
          /*serving_core_selector=*/nullptr),
      StatusIs(params.expected_status.code(),
               HasSubstr(params.expected_status.message())));
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerCreateTests, StableHloRunnerCreateTest,
    testing::ValuesIn<StableHloRunnerCreateTestParams>({
        // Test for creation failure due to mlir::TupleType to ifrt::DType
        // conversion error.
        {
            .test_name = "GetStableHloIOSpecFailed",
            // StableHlo input args: tuple<tensor<4xf32>, tensor<4xf32>>
            .model_path = kTupleArgsModelPath,
            .stablehlo_spec =
                {
                    .inputs = {},
                    .outputs = {},
                    .mlir_module_kept_var_idx = {},
                },
            .assigned_devices = std::make_shared<const AssignedDevices>(
                AssignedDevices{.device_assignment = std::vector<int>{
                                    /*device0*/ 0, 0, 0, 0},
                                .num_replicas = 1,
                                .num_cores_per_replica = 1}),
            .expected_status =
                absl::InvalidArgumentError("Invalid XLA PrimitiveType"),
        },
        // Test for creation failure due to mismatch error of ValidateIOSpec.
        {
            .test_name = "ValidateIOSpecFailedMismatchError",
            .model_path = kSimpleModelPath,
            .stablehlo_spec =
                {
                    .inputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 8}),
                          .sharding = xla::HloSharding::Replicate()}},
                    .outputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF16),
                          .shape = ArgShape({2, 8}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 8}),
                          .sharding = xla::HloSharding::Manual()}},
                    .mlir_module_kept_var_idx = {0, 1},
                },
            .assigned_devices = std::make_shared<const AssignedDevices>(
                AssignedDevices{.device_assignment = std::vector<int>{
                                    /*device0*/ 0, 0, 0, 0},
                                .num_replicas = 1,
                                .num_cores_per_replica = 1}),
            .expected_status = absl::FailedPreconditionError(
                "StableHlo expected arg#0 to have ArgSpec: {dtype: F32, "
                "shape: {2, 8}, sharding: {replicated}}, but got {dtype: F16, "
                "shape: {2, 8}, sharding: {replicated}} from model IO spec"),
        },
        // Test for creation failure due to unsupported type error of
        // ValidateIOSpec.
        {
            .test_name = "ValidateIOSpecFailedUnsupportedTypeError",
            // StableHlo output args: !stablehlo.token, tensor<f32>
            .model_path = kTokenArgsModelPath,
            .stablehlo_spec =
                {
                    .inputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kToken),
                          .shape = ArgShape({}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({}),
                          .sharding = xla::HloSharding::Replicate()}},
                    .outputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kToken),
                          .shape = ArgShape({}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({}),
                          .sharding = xla::HloSharding::Replicate()}},
                    .mlir_module_kept_var_idx = {0, 1},
                },
            .assigned_devices =
                std::make_shared<const AssignedDevices>(AssignedDevices{
                    .device_assignment = std::vector<int>{/*device0*/ 0, 0, 0,
                                                          0},
                    .num_replicas = 1,
                    .num_cores_per_replica = 1,
                }),
            .expected_status =
                absl::InvalidArgumentError("Type: TOKEN is not supported"),
        },
        // TODO: Add more failure tests for errors occurring
        // during model loading, io_spec validation, platform index setting etc.
    }),
    [](const testing::TestParamInfo<StableHloRunnerCreateTestParams>& info) {
      return info.param.test_name;
    });

struct StableHloRunnerExecuteTestParams {
  std::string test_name;
  absl::string_view model_path;
  StableHlo::Spec stablehlo_spec;
  std::shared_ptr<const AssignedDevices> assigned_devices;
  std::vector<HostBuffer> input_data;
  absl::StatusOr<std::vector<HostBuffer>> expected_outputs;
  bool use_xla_options_from_model = false;
  bool use_shardy_partitioner = false;
};

using StableHloRunnerCreateFailureTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;
using StableHloRunnerExecuteTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;
using StableHloRunnerExecuteFailureTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;
using StableHloRunnerExecuteSuccessTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;

xla::CompileOptionsProto GetCompileOptions(
    std::shared_ptr<const AssignedDevices> assigned_devices,
    bool use_shardy_partitioner = false) {
  xla::CompileOptionsProto compile_options;
  compile_options.mutable_executable_build_options()->set_num_replicas(
      assigned_devices->num_replicas);
  compile_options.mutable_executable_build_options()->set_num_partitions(
      assigned_devices->num_cores_per_replica);
  compile_options.mutable_executable_build_options()->set_use_spmd_partitioning(
      true);
  compile_options.mutable_executable_build_options()
      ->set_use_shardy_partitioner(use_shardy_partitioner);
  xla::DeviceAssignmentProto* device_assignment =
      compile_options.mutable_executable_build_options()
          ->mutable_device_assignment();
  device_assignment->set_replica_count(assigned_devices->num_replicas);
  device_assignment->set_computation_count(
      assigned_devices->num_cores_per_replica);
  int device_id_index = 0;
  for (int i = 0; i < device_assignment->computation_count(); ++i) {
    xla::DeviceAssignmentProto::ComputationDevice* computation_device =
        device_assignment->add_computation_devices();
    for (int j = 0; j < device_assignment->replica_count(); ++j) {
      computation_device->add_replica_device_ids(device_id_index++);
    }
  }
  return compile_options;
}

TEST_P(StableHloRunnerExecuteSuccessTest, RunnerExecuteSuccess) {
  const StableHloRunnerExecuteTestParams& params = GetParam();
  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(params.model_path);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  StableHlo::Spec stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module_string = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);

  std::optional<xla::CompileOptionsProto> compile_options;
  if (params.use_xla_options_from_model) {
    compile_options.emplace(GetCompileOptions(params.assigned_devices,
                                              params.use_shardy_partitioner));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, params.assigned_devices,
          /*serving_core_selector=*/nullptr));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          runner->Execute(params.input_data));

  for (int i = 0; i < outputs.size(); ++i) {
    EXPECT_THAT(outputs[i], HostBufferEq(params.expected_outputs.value()[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerExecuteSuccessTests, StableHloRunnerExecuteSuccessTest,
    testing::ValuesIn<StableHloRunnerExecuteTestParams>(
        {
            // Test for linear multi-platform model.
            {
                .test_name = "LinearMultiPlatformModel",
                .model_path = kLinearMultiPlatformModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape(1, 2),
                              .sharding = xla::HloSharding::PartialTile(
                                  xla::TileAssignment({2, 2}, {2, 2}, {1, 0}))},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 2}),
                              .sharding = xla::HloSharding::IotaTile({2, 2})},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({4, 2}),
                              .sharding = xla::HloSharding::PartialTile(
                                  xla::TileAssignment({2, 1, 2}))}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({4, 2}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1, 2},
                        .target_platforms = {"cpu", "tpu"},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4,
                    }),
                .input_data =
                    std::vector<HostBuffer>{
                        // offset
                        AsHostBuffer<float>({0.1, 0.2}, xla::ifrt::Shape({2})),
                        // weights
                        AsHostBuffer<float>({1.0, 0.0, 2.0, 4.0},
                                            xla::ifrt::Shape({2, 2})),
                        // user input
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0},
                                            xla::ifrt::Shape({4, 2}))},
                .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
                    {5.1, 8.2, 11.1, 16.2, 17.1, 24.2, 23.1, 32.2},
                    xla::ifrt::Shape({4, 2}))},
                .use_xla_options_from_model = false,
            },
            {
                .test_name = "LinearMultiPlatformModelUseXlaOptionsFromModel",
                .model_path = kLinearMultiPlatformModelPath,
                .stablehlo_spec{
                    .inputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape(1, 2),
                          .sharding = xla::HloSharding::PartialTile(
                              xla::TileAssignment({2, 2}, {2, 2}, {1, 0}))},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({4, 2}),
                          .sharding = xla::HloSharding::PartialTile(
                              xla::TileAssignment({2, 1, 2}))}},
                    .outputs = {{.dtype =
                                     xla::ifrt::DType(xla::ifrt::DType::kF32),
                                 .shape = ArgShape({4, 2}),
                                 .sharding = xla::HloSharding::Replicate()}},
                    .mlir_module_kept_var_idx = {0, 1, 2},
                    .target_platforms = {"cpu", "tpu"},
                },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        // offset
                        AsHostBuffer<float>({0.1, 0.2}, xla::ifrt::Shape({2})),
                        // weights
                        AsHostBuffer<float>({1.0, 0.0, 2.0, 4.0},
                                            xla::ifrt::Shape({2, 2})),
                        // user input
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0},
                                            xla::ifrt::Shape({4, 2}))},
                .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
                    {5.1, 8.2, 11.1, 16.2, 17.1, 24.2, 23.1, 32.2},
                    xla::ifrt::Shape({4, 2}))},
                .use_xla_options_from_model = true,
            },
            // Test for dynamic shape model.
            {
                .test_name = "DynamicShapeModel",
                .model_path = kSimpleDynamicShapeModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1, 2},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({3, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({2, 3})),
                        AsHostBuffer<float>(10.0)},
                .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
                    {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
                    xla::ifrt::Shape({3, 3}))},
                .use_xla_options_from_model = false,
            },
            {
                .test_name = "DynamicShapeModelUseXlaOptionsFromModel",
                .model_path = kSimpleDynamicShapeModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1, 2},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({3, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({2, 3})),
                        AsHostBuffer<float>(10.0)},
                .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
                    {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
                    xla::ifrt::Shape({3, 3}))},
                .use_xla_options_from_model = true,
            },
            {
                .test_name = "ShardyModel",
                .model_path = kSpmdShardyModelPath,
                .stablehlo_spec{
                    .inputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})}},
                    .outputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::Replicate()}},
                    .mlir_module_kept_var_idx = {0, 1},
                    .target_platforms = {"cpu", "tpu"},
                },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>({1.0, 0.0, 2.0, 4.0},
                                            xla::ifrt::Shape({2, 2})),
                        AsHostBuffer<float>({1.0, 0.0, 2.0, 4.0},
                                            xla::ifrt::Shape({2, 2}))},
                .expected_outputs =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>({1.0, 0.0, 10.0, 16.0},
                                            xla::ifrt::Shape({2, 2})),
                        AsHostBuffer<float>({0.0, -1.0, 9.0, 15.0},
                                            xla::ifrt::Shape({2, 2}))},
                .use_xla_options_from_model = true,
                .use_shardy_partitioner = true,
            },
            // Test for SPMD shardy model where shardy respects
            // allow_spmd_sharding_propagation_to_parameters/output flags and
            // doesn't propagate sharding to all parameters/outputs.
            {
                .test_name = "ShardyModelCheckShardingPropagation",
                .model_path = kSpmdShardyCheckShardingPropagationModelPath,
                .stablehlo_spec{
                    .inputs =
                        {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 8}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 8}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 16}),
                          .sharding = xla::HloSharding::Replicate()}},
                    .outputs = {{.dtype =
                                     xla::ifrt::DType(xla::ifrt::DType::kF32),
                                 .shape = ArgShape({8, 16}),
                                 .sharding = xla::HloSharding::Replicate()}},
                    .mlir_module_kept_var_idx = {0, 1, 2},
                    .target_platforms = {"cpu", "tpu"},
                },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>(std::vector<float>(8 * 8, 1.0f),
                                            xla::ifrt::Shape({8, 8})),
                        AsHostBuffer<float>(std::vector<float>(8 * 8, 2.0f),
                                            xla::ifrt::Shape({8, 8})),
                        AsHostBuffer<float>(std::vector<float>(8 * 16, 3.0f),
                                            xla::ifrt::Shape({8, 16}))},
                .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
                    std::vector<float>(8 * 16, 72.0f),
                    xla::ifrt::Shape({8, 16}))},
                .use_xla_options_from_model = true,
                .use_shardy_partitioner = true,
            },
        }),
    [](const testing::TestParamInfo<StableHloRunnerExecuteTestParams>& info) {
      return info.param.test_name;
    });

using StableHloRunnerWithCoreSelectionTest =
    TestWithParam<bool /*specify_device_assignment*/>;

TEST_P(StableHloRunnerWithCoreSelectionTest, CoreSelection) {
  // Set up.
  CellReader<int64_t> core_selection_count_reader(
      "/third_party/neptune/runtime/core_selection_count");
  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(kSimpleModelPath);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 2}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 8}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate()},
                  {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module_kept_var_idx = {0, 1},
  };
  stablehlo_spec.mlir_module_string = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  auto assigned_devices =
      GetParam()
          ? std::make_shared<const AssignedDevices>(AssignedDevices{
                .device_assignment = std::vector<int>{/*device0*/ 0, 0, 0, 0},
                .num_replicas = 1,
                .num_cores_per_replica = 1})
          : std::make_shared<const AssignedDevices>(
                AssignedDevices{.num_replicas = 1, .num_cores_per_replica = 1});
  std::optional<xla::CompileOptionsProto> compile_options =
      GetCompileOptions(assigned_devices, /*use_shardy_partitioner=*/false);
  auto device_selector =
      std::make_unique<tsl::test_util::MockServingDeviceSelector>();
  auto serving_core_selector = std::make_unique<ServingCoreSelector>(
      device_selector.get(), ifrt_client->addressable_device_count());
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, assigned_devices,
          serving_core_selector.get()));
  const auto input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};
  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  // The first few executions are for warmup.
  for (int i = 0; i < ifrt_client->addressable_device_count(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                            runner->Execute(input_data));
    for (int j = 0; j < outputs.size(); ++j) {
      EXPECT_THAT(outputs[j], HostBufferEq(expected_outputs[j]));
    }
    ASSERT_OK_AND_ASSIGN(
        const std::string program_fingerprint,
        StableHloRunnerTestPeer::GetProgramFingerprint(*runner));
    EXPECT_EQ(
        core_selection_count_reader.Delta(program_fingerprint, absl::StrCat(i)),
        1);
  }
  // After warmup, core selection will be delegated to `device_selector`.
  EXPECT_CALL(*device_selector, ReserveDevice(_))
      .WillOnce([&](testing::Unused) {
        return tsl::DeviceReservation(/*device_index=*/1,
                                      device_selector.get());
      });
  TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                          runner->Execute(input_data));
  for (int i = 0; i < outputs.size(); ++i) {
    EXPECT_THAT(outputs[i], HostBufferEq(expected_outputs[i]));
  }
  ASSERT_OK_AND_ASSIGN(const std::string program_fingerprint,
                       StableHloRunnerTestPeer::GetProgramFingerprint(*runner));
  EXPECT_EQ(core_selection_count_reader.Delta(program_fingerprint, "1"), 1);
}

INSTANTIATE_TEST_SUITE_P(StableHloRunnerWithCoreSelectionTests,
                         StableHloRunnerWithCoreSelectionTest,
                         ::testing::Bool());

using StableHloRunnerConcurrentExecuteTest = TestWithParam<bool>;

TEST_P(StableHloRunnerConcurrentExecuteTest, ConcurrentRuns) {
  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(kSimpleDynamicShapeModelPath);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  auto stablehlo_spec = StableHlo::Spec{
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({std::nullopt, std::nullopt}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({std::nullopt, 3}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({std::nullopt, 3}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module_string = hlo_str,
      .mlir_module_kept_var_idx = {0, 1, 2},
  };
  TF_ASSERT_OK_AND_ASSIGN(auto stablehlo, StableHlo::Create({stablehlo_spec}));
  auto assigned_devices = std::make_shared<const AssignedDevices>(
      AssignedDevices{.device_assignment = std::vector<int>{
                          /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                          /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                      .num_replicas = 1,
                      .num_cores_per_replica = 4});

  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  bool use_xla_options_from_model = GetParam();
  std::optional<xla::CompileOptionsProto> compile_options;
  if (use_xla_options_from_model) {
    compile_options = GetCompileOptions(assigned_devices);
  }
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, assigned_devices,
          /*serving_core_selector=*/nullptr));

  auto [promise1, future1] = tsl::MakePromise<>();

  auto runbase = [&](absl::Span<const HostBuffer> input_data,
                     absl::Span<const HostBuffer> expected_outputs) {
    TF_ASSERT_OK_AND_ASSIGN(std::vector<HostBuffer> outputs,
                            runner->Execute(input_data));

    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      EXPECT_THAT(outputs[i], HostBufferEq(expected_outputs[i]));
    }
  };
  auto run1 = [&]() {
    std::vector<HostBuffer> input_data = std::vector<HostBuffer>{
        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            xla::ifrt::Shape({3, 2})),
        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            xla::ifrt::Shape({2, 3})),
        AsHostBuffer<float>(10.0)};
    std::vector<HostBuffer> expected_outputs =
        std::vector<HostBuffer>{AsHostBuffer<float>(
            {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
            xla::ifrt::Shape({3, 3}))};
    runbase(input_data, expected_outputs);
    promise1.Set();
  };
  auto [promise2, future2] = tsl::MakePromise<>();
  auto run2 = [&]() {
    std::vector<HostBuffer> input_data = std::vector<HostBuffer>{
        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                            xla::ifrt::Shape({4, 2})),
        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                            xla::ifrt::Shape({2, 3})),
        AsHostBuffer<float>(10.0)};
    std::vector<HostBuffer> expected_outputs =
        std::vector<HostBuffer>{AsHostBuffer<float>(
            {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0, 49, 64, 79},
            xla::ifrt::Shape({4, 3}))};
    runbase(input_data, expected_outputs);
    promise2.Set();
  };
  thread_pool->Schedule(run1);
  thread_pool->Schedule(run2);
  EXPECT_OK(future1.Await());
  EXPECT_OK(future2.Await());
}

INSTANTIATE_TEST_SUITE_P(StableHloRunnerConcurrentExecuteTests,
                         StableHloRunnerConcurrentExecuteTest,
                         ::testing::Values(false, true));
TEST_P(StableHloRunnerCreateFailureTest, RunnerCreationFailure) {
  const StableHloRunnerExecuteTestParams& params = GetParam();

  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(params.model_path);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  auto stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module_string = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(auto stablehlo, StableHlo::Create({stablehlo_spec}));
  xla::CompileOptionsProto compile_options =
      GetCompileOptions(params.assigned_devices);
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);

  EXPECT_THAT(StableHloRunner::Create(
                  std::move(stablehlo), compile_options, ifrt_client,
                  /*persistent_compilation_cache=*/nullptr,
                  &ifrt_loaded_variable_registry, {},
                  /*compilation_environment_proto=*/nullptr, thread_pool.get(),
                  /*optimizer=*/nullptr, std::make_shared<AssignedDevices>(),
                  /*serving_core_selector=*/nullptr),
              StatusIs(params.expected_outputs.status().code(),
                       HasSubstr(params.expected_outputs.status().message())));
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerCreateFailureTests, StableHloRunnerCreateFailureTest,
    testing::ValuesIn<StableHloRunnerExecuteTestParams>(
        {
            // Test for StableHLO with polymorphic input shapes, but somehow
            // jax.uses_shape_polymorphism=false in the StableHLO module.
            // This is not a valid StableHLO and should fail.
            {
                .model_path = kSimpleFalseDynamicShapeAttrModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1, 2},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>(
                            {1, 2, 3, 4, 5, 6}, xla::ifrt::Shape({3, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({2, 3})),
                        AsHostBuffer<float>(10.0)},
                .expected_outputs = absl::InvalidArgumentError(
                    "StableHLO module has polymorphic shape in the input spec, "
                    "but jax.uses_shape_polymorphism is false"),
            },
        }));

TEST_P(StableHloRunnerExecuteFailureTest, RunnerExecuteFailure) {
  const StableHloRunnerExecuteTestParams& params = GetParam();

  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(params.model_path);
  std::string hlo_str;
  TF_ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  auto stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module_string = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(auto stablehlo, StableHlo::Create({stablehlo_spec}));
  xla::CompileOptionsProto compile_options =
      GetCompileOptions(params.assigned_devices);
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, std::make_shared<AssignedDevices>(),
          /*serving_core_selector=*/nullptr));

  std::vector<xla::ifrt::ArrayRef> inputs(params.input_data.size());
  for (int i = 0; i < params.input_data.size(); ++i) {
    xla::HloSharding& sharding = stablehlo_spec.inputs[i].sharding;
    TF_ASSERT_OK_AND_ASSIGN(
        inputs[i], sharding_internal::MakeArrayFromHostBuffer(
                       *ifrt_client, params.input_data[i],
                       *(params.assigned_devices->device_assignment), sharding,
                       *thread_pool));
  }
  std::vector<HostBuffer> host_buffers = params.input_data;
  std::vector<Value> values = internal::PackAsValues(std::move(host_buffers));
  EXPECT_THAT(runner->Execute(params.input_data),
              StatusIs(params.expected_outputs.status().code(),
                       HasSubstr(params.expected_outputs.status().message())));
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerExecuteFailureTests, StableHloRunnerExecuteFailureTest,
    testing::ValuesIn<StableHloRunnerExecuteTestParams>(
        {
            {
                .test_name = "SimpleModelInputDTypeMismatch",
                .model_path = kSimpleModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{// StableHlo expects the input to have type F32.
                              .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 2}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, 0, /*device1*/ 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        // The actual input has type S32.
                        AsHostBuffer<int32_t>({1, 2, 3, 4},
                                              xla::ifrt::Shape({2, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                             14.0, 15.0, 16.0},
                                            xla::ifrt::Shape({2, 8}))},
                .expected_outputs = absl::InvalidArgumentError(
                    "Input dtype mismatch. Expected: F32, but got: S32"),
            },
            {
                .test_name = "SimpleModelInputShapeMismatch",
                .model_path = kSimpleModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 2}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 8}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>({1, 2, 3, 4},
                                            xla::ifrt::Shape({1, 4})),
                        AsHostBuffer<float>(
                            {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0,
                             9.0, 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                            xla::ifrt::Shape({4, 4}))},
                .expected_outputs = absl::FailedPreconditionError(
                    "No MLIR module found for input shapes [1,4],[4,4]"),
            },
            {
                .test_name = "DynamicShapeModelInputShapeMismatch",
                .model_path = kSimpleDynamicShapeModelPath,
                .stablehlo_spec =
                    {
                        .inputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              // StableHlo expects the input to have rank 0
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .outputs =
                            {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()}},
                        .mlir_module_kept_var_idx = {0, 1, 2},
                    },
                .assigned_devices =
                    std::make_shared<const AssignedDevices>(AssignedDevices{
                        .device_assignment = std::vector<int>{
                            /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                            /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                        .num_replicas = 1,
                        .num_cores_per_replica = 4}),
                .input_data =
                    std::vector<HostBuffer>{
                        AsHostBuffer<float>(
                            {1, 2, 3, 4, 5, 6}, xla::ifrt::Shape({3, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                                            xla::ifrt::Shape({2, 3})),
                        // The actual input has rank 1.
                        AsHostBuffer<float>({10.0}, xla::ifrt::Shape({1}))},
                .expected_outputs =
                    absl::InvalidArgumentError("rank must match"),
            },
        }));

class StableHloRunnerFullyInitializedTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string file_path =
        devtools_build::GetDataDependencyFilepath(kSimpleDynamicShapeModelPath);
    std::string hlo_str;
    TF_ASSERT_OK(
        tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
    stablehlo_spec_ = StableHlo::Spec{
        .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({std::nullopt, std::nullopt}),
                    .sharding = xla::HloSharding::Replicate()},
                   {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({std::nullopt, 3}),
                    .sharding = xla::HloSharding::Replicate()},
                   {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                    .shape = ArgShape({}),
                    .sharding = xla::HloSharding::Replicate()}},
        .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                     .shape = ArgShape({std::nullopt, 3}),
                     .sharding = xla::HloSharding::Replicate()}},
        .mlir_module_string = hlo_str,
        .mlir_module_kept_var_idx = {0, 1, 2},
    };
    TF_ASSERT_OK_AND_ASSIGN(auto stablehlo,
                            StableHlo::Create({stablehlo_spec_}));

    TF_ASSERT_OK_AND_ASSIGN(ifrt_client_, xla::ifrt::test_util::GetClient());
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);

    std::shared_ptr<const AssignedDevices> assigned_devices =
        std::make_shared<const AssignedDevices>(
            AssignedDevices{.device_assignment = std::vector<int>{
                                /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                                /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
                            .num_replicas = 1,
                            .num_cores_per_replica = 4});
    xla::CompileOptionsProto compile_options =
        GetCompileOptions(assigned_devices);
    TF_ASSERT_OK_AND_ASSIGN(
        runner_,
        StableHloRunner::Create(
            std::move(stablehlo), compile_options, ifrt_client_,
            /*persistent_compilation_cache=*/nullptr,
            &ifrt_loaded_variable_registry_,
            /*weights=*/{},
            /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
            /*optimizer=*/nullptr, assigned_devices,
            /*serving_core_selector=*/nullptr));
  }

  absl::Status RunAndCheck(absl::Span<const HostBuffer> input_data,
                           absl::Span<const HostBuffer> expected_outputs) {
    TF_ASSIGN_OR_RETURN(auto outputs, runner_->Execute(input_data));
    EXPECT_EQ(outputs.size(), expected_outputs.size());
    for (int i = 0; i < outputs.size(); ++i) {
      EXPECT_THAT(outputs[i], HostBufferEq(expected_outputs[i]));
    }
    return absl::OkStatus();
  }

  StableHlo::Spec stablehlo_spec_;
  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  std::unique_ptr<StableHloRunner> runner_;
};

TEST_F(StableHloRunnerFullyInitializedTest,
       PreviouslyExecutedInputShapesSucceeds) {
  const std::vector<HostBuffer> input_data = std::vector<HostBuffer>{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({3, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({2, 3})),
      AsHostBuffer<float>(10.0)};
  const std::vector<HostBuffer> expected_outputs =
      std::vector<HostBuffer>{AsHostBuffer<float>(
          {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
          xla::ifrt::Shape({3, 3}))};
  {
    TF_ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  }
  TF_ASSERT_OK(runner_->MarkFullyInitialized());
  {
    TF_ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  }
}

TEST_F(StableHloRunnerFullyInitializedTest, NewInputShapesFails) {
  std::vector<HostBuffer> input_data = std::vector<HostBuffer>{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({3, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({2, 3})),
      AsHostBuffer<float>(10.0)};
  std::vector<HostBuffer> expected_outputs =
      std::vector<HostBuffer>{AsHostBuffer<float>(
          {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
          xla::ifrt::Shape({3, 3}))};
  std::vector<HostBuffer> new_input_data = std::vector<HostBuffer>{
      AsHostBuffer<float>({1.0}, xla::ifrt::Shape({1, 1})),
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3})),
      AsHostBuffer<float>(10.0)};

  TF_ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  TF_ASSERT_OK(runner_->MarkFullyInitialized());
  {
    EXPECT_THAT(
        RunAndCheck(new_input_data, expected_outputs),
        StatusIs(
            absl::StatusCode::kFailedPrecondition,
            HasSubstr(
                "Got new input shapes requiring a new executable, but can't "
                "create it after full initialization. Please warm up the model "
                "with all the input shapes that are possible during serving. "
                "(See go/tf-model-warmup for more details.)")));
  }
}

TEST_F(StableHloRunnerFullyInitializedTest,
       NewInputShapesBeforeFullyInitializedSucceeds) {
  std::vector<HostBuffer> input_data = std::vector<HostBuffer>{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({3, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0},
                          xla::ifrt::Shape({2, 3})),
      AsHostBuffer<float>(10.0)};
  std::vector<HostBuffer> expected_outputs =
      std::vector<HostBuffer>{AsHostBuffer<float>(
          {19.0, 22.0, 25.0, 29.0, 36.0, 43.0, 39.0, 50.0, 61.0},
          xla::ifrt::Shape({3, 3}))};
  std::vector<HostBuffer> new_input_data = std::vector<HostBuffer>{
      AsHostBuffer<float>({1.0}, xla::ifrt::Shape({1, 1})),
      AsHostBuffer<float>({1.0, 2.0, 3.0}, xla::ifrt::Shape({1, 3})),
      AsHostBuffer<float>(10.0)};
  std::vector<HostBuffer> new_expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>({11.0, 12.0, 13.0}, xla::ifrt::Shape({1, 3}))};
  TF_ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  TF_ASSERT_OK(RunAndCheck(new_input_data, new_expected_outputs));
  TF_ASSERT_OK(runner_->MarkFullyInitialized());
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

static void BM_StableHloModel(benchmark::State& state,
                              absl::string_view model_path,
                              StableHlo::Spec stablehlo_spec,
                              std::vector<HostBuffer> args) {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "benchmark", 8);

  const std::string file_path =
      devtools_build::GetDataDependencyFilepath(model_path);
  auto ifrt_client = xla::ifrt::test_util::GetClient().value();

  std::string str;
  CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &str));
  stablehlo_spec.mlir_module_string = str;

  std::unique_ptr<StableHlo> stablehlo =
      StableHlo::Create({stablehlo_spec}).value();

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  std::unique_ptr<StableHloRunner> runner =
      StableHloRunner::Create(
          std::move(stablehlo),
          /*compile_options=*/std::nullopt, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, std::make_shared<const AssignedDevices>(),
          /*serving_core_selector=*/nullptr)
          .value();

  // Warm up the model to avoid compilation inside the benchmark loop.
  CHECK_OK(runner->Execute(args).status());

  for (auto s : state) {
    CHECK_OK(runner->Execute(args).status());
  }
}

static void BM_SimpleModel(benchmark::State& state) {
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 2}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 8}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate()},
                  {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module_kept_var_idx = {0, 1},
  };

  HostBuffer arg0 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  HostBuffer arg1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));

  BM_StableHloModel(state, kSimpleModelPath, stablehlo_spec, {arg0, arg1});
}

BENCHMARK(BM_SimpleModel);

}  // namespace
}  // namespace jsv

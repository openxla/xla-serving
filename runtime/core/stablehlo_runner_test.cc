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

#include "runtime/core/stablehlo_runner.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "xla/tsl/platform/test_benchmark.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/cord.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "llvm/include/llvm/ADT/ArrayRef.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "runtime/core/aot_stablehlo_runner.h"  // from @xla_serving
#include "runtime/core/device_buffer.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_converter.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/context.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "xla/debug_options_flags.h"  // from @xla
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/hlo/ir/tile_assignment.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/framework/test_util/mock_serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

class MockH2DTransferExecutor : public H2DTransferExecutor {
 public:
  explicit MockH2DTransferExecutor(xla::ifrt::Client* client,
                                   xla::ifrt::DeviceListRef device_list,
                                   tsl::thread::ThreadPool* thread_pool)
      : client_(client),
        device_list_(std::move(device_list)),
        thread_pool_(thread_pool) {}

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const xla_serving::HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings,
      absl::Span<const std::optional<xla::ifrt::Shape>> static_shapes)
      const override {
    std::vector<xla::ifrt::ArrayRef> result;
    for (int i = 0; i < inputs.size(); ++i) {
      TF_ASSIGN_OR_RETURN(auto array_result,
                          sharding_internal::MakeArrayFromHostBuffer(
                              *client_, inputs[i], device_list_,
                              input_shardings[i], *thread_pool_));
      result.push_back(std::move(array_result));
    }
    return result;
  }

 private:
  xla::ifrt::Client* client_;
  xla::ifrt::DeviceListRef device_list_;
  tsl::thread::ThreadPool* thread_pool_;
};

class MockH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) override {
    return std::make_unique<MockH2DTransferExecutor>(
        std::move(client), std::move(device_list), thread_pool);
  }
};

class MockErrorH2DTransferExecutor : public H2DTransferExecutor {
 public:
  explicit MockErrorH2DTransferExecutor(
      xla::ifrt::Client* /*client*/,
      xla::ifrt::DeviceListRef /*device_list*/,
      tsl::thread::ThreadPool* /*thread_pool*/) {}

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const xla_serving::HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings,
      absl::Span<const std::optional<xla::ifrt::Shape>> static_shapes)
      const override {
    return absl::InternalError("Simulated Arena Transfer Failure");
  }
};

class MockErrorH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) override {
    return std::make_unique<MockErrorH2DTransferExecutor>(
        std::move(client), std::move(device_list), thread_pool);
  }
};

// Records the (shape, static shape) of every transferred input.
struct RecordedTransfer {
  xla::ifrt::Shape shape;
  std::optional<xla::ifrt::Shape> static_shape;
};

// Emulates dynamic DMA without a backend that supports it: an input bound to a
// static shape is zero-padded on the host to the static shape before the
// transfer, so the resulting array has the static shape and the executable can
// consume it. Real executors allocate the static shape on the device instead
// and transfer only the actual data (see `H2DTransferExecutor`).
class MockPaddingH2DTransferExecutor : public H2DTransferExecutor {
 public:
  MockPaddingH2DTransferExecutor(
      xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool,
      std::shared_ptr<std::vector<RecordedTransfer>> recorded_transfers)
      : client_(client),
        device_list_(std::move(device_list)),
        thread_pool_(thread_pool),
        recorded_transfers_(std::move(recorded_transfers)) {}

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const xla_serving::HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings,
      absl::Span<const std::optional<xla::ifrt::Shape>> static_shapes)
      const override {
    std::vector<xla::ifrt::ArrayRef> result;
    for (int i = 0; i < inputs.size(); ++i) {
      const std::optional<xla::ifrt::Shape> static_shape =
          static_shapes.empty() ? std::nullopt : static_shapes[i];
      recorded_transfers_->push_back(
          {.shape = inputs[i].shape(), .static_shape = static_shape});
      HostBuffer to_transfer = inputs[i];
      if (static_shape.has_value()) {
        HostBuffer padded(inputs[i].dtype(), *static_shape);
        std::memset(padded.base(), 0, padded.total_storage_bytes());
        std::memcpy(padded.base(), inputs[i].raw_data().data(),
                    inputs[i].raw_data().size());
        to_transfer = std::move(padded);
      }
      TF_ASSIGN_OR_RETURN(auto array_result,
                          sharding_internal::MakeArrayFromHostBuffer(
                              *client_, to_transfer, device_list_,
                              input_shardings[i], *thread_pool_));
      result.push_back(std::move(array_result));
    }
    return result;
  }

 private:
  xla::ifrt::Client* client_;
  xla::ifrt::DeviceListRef device_list_;
  tsl::thread::ThreadPool* thread_pool_;
  std::shared_ptr<std::vector<RecordedTransfer>> recorded_transfers_;
};

class MockPaddingH2DTransferExecutorFactory
    : public H2DTransferExecutorFactory {
 public:
  explicit MockPaddingH2DTransferExecutorFactory(
      std::shared_ptr<std::vector<RecordedTransfer>> recorded_transfers)
      : recorded_transfers_(std::move(recorded_transfers)) {}

  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) override {
    return std::make_unique<MockPaddingH2DTransferExecutor>(
        client, std::move(device_list), thread_pool, recorded_transfers_);
  }

 private:
  std::shared_ptr<std::vector<RecordedTransfer>> recorded_transfers_;
};

class MockStablehloOptimizer : public StablehloOptimizer {
 public:
  absl::Status Optimize(const Args& args, mlir::ModuleOp module) override {
    return absl::OkStatus();
  }
};

class FlattenInputsTestOptimizer : public StablehloOptimizer {
 public:
  // Arguments in `nchw_transposed_args` are marked with `kHostTransposeAttr`
  // = [0, 3, 1, 2] (NHWC -> NCHW), as `TpuInputNchwTransposePass` does.
  explicit FlattenInputsTestOptimizer(
      absl::flat_hash_set<int> nchw_transposed_args = {})
      : nchw_transposed_args_(std::move(nchw_transposed_args)) {}

  absl::Status Optimize(const Args& args, mlir::ModuleOp module) override {
    mlir::Operation& func_op = module.getBody()->front();
    mlir::Block& entry_block = func_op.getRegion(0).front();
    std::vector<mlir::Type> new_arg_types;
    new_arg_types.reserve(entry_block.getNumArguments());
    for (mlir::BlockArgument arg : entry_block.getArguments()) {
      mlir::Operation* reshape_op = *arg.getUsers().begin();
      mlir::Type r1_type = reshape_op->getResult(0).getType();
      arg.setType(r1_type);
      reshape_op->getResult(0).replaceAllUsesWith(arg);
      reshape_op->erase();
      new_arg_types.push_back(r1_type);
    }
    auto new_func_type = mlir::FunctionType::get(module.getContext(),
                                                 new_arg_types, new_arg_types);
    func_op.setAttr("function_type", mlir::TypeAttr::get(new_func_type));
    auto main_func = llvm::cast<mlir::func::FuncOp>(func_op);
    for (int idx : nchw_transposed_args_) {
      main_func.setArgAttr(
          idx, kHostTransposeAttr,
          mlir::DenseI64ArrayAttr::get(module.getContext(), {0, 3, 1, 2}));
    }
    return absl::OkStatus();
  }

 private:
  const absl::flat_hash_set<int> nchw_transposed_args_;
};

class NonFlattenTransposeTestOptimizer : public StablehloOptimizer {
 public:
  // Arguments in `nchw_transposed_args` are rewritten from NHWC [N, H, W, C]
  // to NCHW [N, C, H, W] without R1 flattening, marked with
  // `kHostTransposeAttr` = [0, 3, 1, 2], and have their consuming on-device
  // NHWC -> NCHW transpose op eliminated, matching what
  // `TpuInputNchwTransposePass` does.
  explicit NonFlattenTransposeTestOptimizer(
      absl::flat_hash_set<int> nchw_transposed_args = {})
      : nchw_transposed_args_(std::move(nchw_transposed_args)) {}

  absl::Status Optimize(const Args& args, mlir::ModuleOp module) override {
    mlir::Operation& func_op = module.getBody()->front();
    mlir::Block& entry_block = func_op.getRegion(0).front();
    auto main_func = llvm::cast<mlir::func::FuncOp>(func_op);
    mlir::OpBuilder builder(main_func.getContext());
    for (int idx : nchw_transposed_args_) {
      mlir::BlockArgument arg = entry_block.getArgument(idx);
      bool eliminated_transpose = false;
      for (mlir::Operation* user : llvm::make_early_inc_range(arg.getUsers())) {
        if (user->getName().getStringRef() == "stablehlo.transpose") {
          arg.setType(user->getResult(0).getType());
          user->getResult(0).replaceAllUsesWith(arg);
          user->erase();
          eliminated_transpose = true;
        }
      }
      if (!eliminated_transpose) {
        if (auto input_type =
                llvm::dyn_cast<mlir::RankedTensorType>(arg.getType());
            input_type && input_type.getShape().size() == 4) {
          const llvm::ArrayRef<int64_t> shape = input_type.getShape();
          arg.setType(mlir::RankedTensorType::get(
              {shape[0], shape[3], shape[1], shape[2]},
              input_type.getElementType()));
        }
      }
      main_func.setArgAttr(
          idx, kHostTransposeAttr,
          builder.getDenseI64ArrayAttr({0, 3, 1, 2}));
    }
    std::vector<mlir::Type> new_arg_types;
    new_arg_types.reserve(entry_block.getNumArguments());
    for (mlir::BlockArgument arg : entry_block.getArguments()) {
      new_arg_types.push_back(arg.getType());
    }
    auto new_func_type = mlir::FunctionType::get(
        module.getContext(), new_arg_types,
        main_func.getFunctionType().getResults());
    func_op.setAttr("function_type", mlir::TypeAttr::get(new_func_type));
    return absl::OkStatus();
  }

 private:
  const absl::flat_hash_set<int> nchw_transposed_args_;
};

class StableHloRunnerTestPeer {
 public:
  static tsl::Future<StableHloRunner::SharedCachedExecutableBundle>
  LookUpOrCreateExecutable(StableHloRunner* runner,
                           absl::Span<const xla::ifrt::Shape> input_shapes,
                           absl::Span<const xla::ifrt::DType> input_dtypes) {
    return runner->LookUpOrCreateExecutable(input_shapes, input_dtypes);
  }

  static absl::StatusOr<std::string> GetProgramFingerprint(
      const StableHloRunner& runner) {
    std::vector<std::string> program_fingerprints;
    absl::MutexLock lock(runner.mutex_);
    for (const auto& [key, future] : runner.executable_bundles_) {
      TF_ASSIGN_OR_RETURN(auto bundle, future.Await());
      program_fingerprints.push_back(
          bundle->program_fingerprint_with_timestamp);
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

std::shared_ptr<StableHlo> CreateMockStableHloTestHelper(
    std::vector<std::vector<int>> module_kept_var_indices,
    const std::vector<StableHloIOSpec>& io_specs, int num_weights) {
  if (io_specs.empty()) {
    return nullptr;
  }
  std::vector<StableHlo::Spec> specs;
  for (size_t b = 0; b < io_specs.size(); ++b) {
    StableHlo::Spec spec;
    auto& kept_idx = module_kept_var_indices[b];
    std::vector<ArgSpec> padded_inputs = io_specs[b].inputs;

    // We must ensure 'padded_inputs' is large enough to be indexed by the
    // maximum index listed in kept_idx, because StableHlo::Spec natively
    // preserves the full global input variable slots.
    int max_idx = 0;
    for (int idx : kept_idx) {
      if (idx > max_idx) max_idx = idx;
    }

    if (!kept_idx.empty() || num_weights > 0 || !io_specs[b].inputs.empty()) {
      std::vector<ArgSpec> actual_inputs(max_idx + 1, io_specs[b].inputs[0]);
      for (size_t k = 0; k < kept_idx.size(); ++k) {
        if (k < io_specs[b].inputs.size()) {
          actual_inputs[kept_idx[k]] = io_specs[b].inputs[k];
        }
      }
      padded_inputs = std::move(actual_inputs);
    }

    if (kept_idx.empty() && !io_specs[b].inputs.empty()) {
      for (size_t i = 0; i < io_specs[b].inputs.size(); ++i) {
        kept_idx.push_back(num_weights + i);
      }
    }

    spec.mlir_module_kept_var_idx = kept_idx;
    spec.inputs = std::move(padded_inputs);
    spec.outputs = io_specs[b].outputs;
    spec.mlir_module = std::string("mock_mlir");
    specs.push_back(std::move(spec));
  }
  return std::shared_ptr<StableHlo>(
      StableHlo::Create(std::move(specs)).value().release());
}

using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Optional;
using ::testing::Return;
using ::testing::TestWithParam;
using ::testing::Unused;
using ::tsl::monitoring::testing::CellReader;

constexpr const char* kSimpleModelPath =
    "runtime/core/"
    "testdata/simple.mlir";
constexpr const char* kLinearMultiPlatformModelPath =
    "runtime/core/"
    "testdata/linear_multi_platform.mlir";
constexpr const char* kSimpleDynamicShapeModelPath =
    "runtime/core/"
    "testdata/simple_with_dynamic_shape.mlir";
constexpr const char* kSimpleFalseDynamicShapeAttrModelPath =
    "runtime/core/"
    "testdata/simple_false_dynamic_shape_attr.mlir";
constexpr const char* kTupleArgsModelPath =
    "runtime/core/"
    "testdata/tuple_args.mlir";
constexpr const char* kTokenArgsModelPath =
    "runtime/core/"
    "testdata/token_args.mlir";
constexpr const char* kSpmdShardyModelPath =
    "runtime/core/"
    "testdata/spmd_shardy.mlir";
constexpr const char* kSpmdShardyCheckShardingPropagationModelPath =
    "runtime/core/"
    "testdata/spmd_shardy_propagation.mlir";

struct StableHloRunnerCreateTestParams {
  std::string test_name;
  std::string model_path;
  StableHlo::Spec stablehlo_spec;
  std::shared_ptr<const AssignedDevices> assigned_devices;
  absl::Status expected_status;
};

using StableHloRunnerCreateTest =
    TestWithParam<StableHloRunnerCreateTestParams>;

TEST_P(StableHloRunnerCreateTest, RunnerCreateFailure) {
  const StableHloRunnerCreateTestParams& params = GetParam();
  const std::string file_path =
      tsl::GetDataDependencyFilepath(params.model_path);
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  StableHlo::Spec stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module = hlo_str;
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
  std::string model_path;
  StableHlo::Spec stablehlo_spec;
  std::shared_ptr<const AssignedDevices> assigned_devices;
  std::vector<HostBuffer> input_data;
  absl::StatusOr<std::vector<HostBuffer>> expected_outputs;
  bool use_xla_options_from_model = false;
  bool use_shardy_partitioner = false;
  bool use_arena_transfer = false;

  enum class InputFormat {
    kAllHostBuffers,
    kAllDeviceBuffers,
    kMixed,
  };

  InputFormat input_format = InputFormat::kAllHostBuffers;
  std::vector<Placement> placements = {};
};

using enum StableHloRunnerExecuteTestParams::InputFormat;

using StableHloRunnerCreateFailureTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;
using StableHloRunnerExecuteTest =
    TestWithParam<std::tuple<StableHloRunnerExecuteTestParams,
                             /*use_arena_transfer=*/bool>>;
using StableHloRunnerExecuteFailureTest =
    TestWithParam<StableHloRunnerExecuteTestParams>;

xla::CompileOptionsProto GetCompileOptions(
    std::shared_ptr<const AssignedDevices> assigned_devices,
    bool use_shardy_partitioner = false) {
  xla::CompileOptionsProto compile_options;
  *compile_options.mutable_executable_build_options()->mutable_debug_options() =
      xla::GetDebugOptionsFromFlags();
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

TEST_P(StableHloRunnerExecuteTest, Execute) {
  StableHloRunnerExecuteTestParams params = std::get<0>(GetParam());
  params.use_arena_transfer = std::get<1>(GetParam());
  std::string hlo_str;
  if (params.model_path.empty()) {
    hlo_str = params.stablehlo_spec.mlir_module_string();
  } else {
    const std::string file_path =
        tsl::GetDataDependencyFilepath(params.model_path);
    ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  }
  StableHlo::Spec stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module = hlo_str;
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
  std::unique_ptr<H2DTransferExecutorFactory> h2d_factory;
  if (params.use_arena_transfer) {
    h2d_factory = std::make_unique<MockH2DTransferExecutorFactory>();
  }

  tsl::monitoring::testing::CellReader<int64_t> input_transfer_counter(
      "/xla_serving/input_transfer/dispatched");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, params.assigned_devices,
          /*serving_core_selector=*/nullptr, std::move(h2d_factory)));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::Device*> assigned_ifrt_devices,
      GetAssignedIfrtDevices(*ifrt_client,
                             params.assigned_devices->num_replicas,
                             params.assigned_devices->num_cores_per_replica,
                             params.assigned_devices->device_assignment));
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client->MakeDeviceList(assigned_ifrt_devices));

  std::vector<xla_serving::Value> value_inputs;
  if (params.input_format == kMixed) {
    for (int i = 0; i < params.input_data.size(); ++i) {
      if (i % 2 == 0) {
        value_inputs.push_back(xla_serving::Value(params.input_data[i]));
      } else {
        xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
            device_list, xla::ifrt::MemoryKind(),
            params.stablehlo_spec.inputs[i].sharding);
        TF_ASSERT_OK_AND_ASSIGN(auto ifrt_array,
                                sharding_internal::MakeArrayFromHostBuffer(
                                    *ifrt_client, params.input_data[i],
                                    device_list, sharding, *thread_pool));
        value_inputs.push_back(
            xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client,
                                         .ifrt_array = std::move(ifrt_array),
                                         .sharding = sharding,
                                         .device_list = device_list}));
      }
    }
  } else if (params.input_format == kAllHostBuffers) {
    value_inputs.assign(params.input_data.begin(), params.input_data.end());
  } else {
    // kAllDeviceBuffers case
    for (int i = 0; i < params.input_data.size(); ++i) {
      xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
          device_list, xla::ifrt::MemoryKind(),
          params.stablehlo_spec.inputs[i].sharding);
      TF_ASSERT_OK_AND_ASSIGN(
          auto ifrt_array, sharding_internal::MakeArrayFromHostBuffer(
                               *ifrt_client, params.input_data[i], device_list,
                               sharding, *thread_pool));
      value_inputs.push_back(
          xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client,
                                       .ifrt_array = std::move(ifrt_array),
                                       .sharding = sharding,
                                       .device_list = device_list}));
    }
  }

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> value_outputs,
                          runner->Execute(value_inputs,
                                          StableHloRunner::ExecuteOptions{
                                              .placements = params.placements},
                                          RunContext{}));

  for (int i = 0; i < value_outputs.size(); ++i) {
    if (value_outputs[i].IsType<HostBuffer>()) {
      EXPECT_THAT(value_outputs[i].Get<HostBuffer>(),
                  HostBufferEq(params.expected_outputs.value()[i]));
    } else {
      const xla_serving::DeviceBuffer& db = value_outputs[i].Get<xla_serving::DeviceBuffer>();
      TF_ASSERT_OK_AND_ASSIGN(auto host_buffer,
                              sharding_internal::MakeHostBufferFromArray(
                                  *ifrt_client, *db.ifrt_array, db.sharding,
                                  db.device_list, *thread_pool)
                                  .Await());
      EXPECT_THAT(host_buffer,
                  HostBufferEq(params.expected_outputs.value()[i]));
    }
  }

  if (params.use_arena_transfer) {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), 1);
    EXPECT_EQ(input_transfer_counter.Delta("default"), 0);
  } else {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), 0);
    EXPECT_EQ(input_transfer_counter.Delta("default"), 1);
  }
}

std::vector<StableHloRunnerExecuteTestParams> GetExecuteTestParams() {
  return {
      // Test for linear multi-platform model.
      {
          .test_name = "LinearMultiPlatformModel",
          .model_path = kLinearMultiPlatformModelPath,
          .stablehlo_spec =
              {
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
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
                  AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                                      xla::ifrt::Shape({4, 2}))},
          .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
              {5.1, 8.2, 11.1, 16.2, 17.1, 24.2, 23.1, 32.2},
              xla::ifrt::Shape({4, 2}))},
          .use_xla_options_from_model = false,
      },
      {
          .test_name = "LinearMultiPlatformModelMixedInputs",
          .model_path = kLinearMultiPlatformModelPath,
          .stablehlo_spec =
              {
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape(1, 2),
                              .sharding =
                                  xla::HloSharding::PartialTile(
                                      xla::TileAssignment({2, 2},
                                                          {2, 2}, {1, 0}))},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 2}),
                              .sharding = xla::HloSharding::IotaTile({2, 2})},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({4, 2}),
                              .sharding =
                                  xla::HloSharding::PartialTile(
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
                  AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
                                      xla::ifrt::Shape({4, 2}))},
          .expected_outputs = std::vector<HostBuffer>{AsHostBuffer<float>(
              {5.1, 8.2, 11.1, 16.2, 17.1, 24.2, 23.1, 32.2},
              xla::ifrt::Shape({4, 2}))},
          .use_xla_options_from_model = false,
          .input_format = kMixed,
          .placements = {Placement::kDevice},
      },
      {
          .test_name = "LinearMultiPlatformModelUseXlaOptionsFromModel",
          .model_path = kLinearMultiPlatformModelPath,
          .stablehlo_spec{
              .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
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
              .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
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
                  AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
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
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                  .outputs = {{.dtype =
                                   xla::ifrt::DType(xla::ifrt::DType::kF32),
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
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, std::nullopt}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({std::nullopt, 3}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({}),
                              .sharding = xla::HloSharding::Replicate()}},
                  .outputs = {{.dtype =
                                   xla::ifrt::DType(xla::ifrt::DType::kF32),
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
              .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({2, 2}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})}},
              .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
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
              .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 8}),
                          .sharding = xla::HloSharding::IotaTile({2, 2})},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 8}),
                          .sharding = xla::HloSharding::Replicate()},
                         {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                          .shape = ArgShape({8, 16}),
                          .sharding = xla::HloSharding::Replicate()}},
              .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
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
              std::vector<float>(8 * 16, 72.0f), xla::ifrt::Shape({8, 16}))},
          .use_xla_options_from_model = true,
          .use_shardy_partitioner = true,
      },
      // Test for dropped argument.
      {
          .test_name = "DroppedMiddleArgumentModel",
          .model_path = "",
          .stablehlo_spec =
              {
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape(1, 2),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape(1, 2),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape(1, 2),
                              .sharding = xla::HloSharding::Replicate()}},
                  .outputs = {{.dtype =
                                   xla::ifrt::DType(xla::ifrt::DType::kF32),
                               .shape = ArgShape(1, 2),
                               .sharding = xla::HloSharding::Replicate()}},
                  .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<2xf32>, %arg1: tensor<2xf32>) -> tensor<2xf32> {
    %0 = "stablehlo.add"(%arg0, %arg1)
      : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    return %0 : tensor<2xf32>
  }
}
)mlir"),
                  .mlir_module_kept_var_idx = {0, 2},
                  .target_platforms = {"cpu"},
              },
          .assigned_devices =
              std::make_shared<const AssignedDevices>(AssignedDevices{
                  .num_replicas = 1,
                  .num_cores_per_replica = 1,
              }),
          .input_data =
              std::vector<HostBuffer>{
                  AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({2})),
                  AsHostBuffer<float>({99.0, 99.0}, xla::ifrt::Shape({2})),
                  AsHostBuffer<float>({3.0, 4.0}, xla::ifrt::Shape({2}))},
          .expected_outputs = std::vector<HostBuffer>{
              AsHostBuffer<float>({4.0, 6.0}, xla::ifrt::Shape({2}))},
          .use_xla_options_from_model = false,
      },
      // Test for 3D tensor with dot_general output to verify layout handling
      // for batch > 1.
      {
          .test_name = "Transposed3dTensorOutputBatch2",
          .model_path = "",
          .stablehlo_spec =
              {
                  .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({2, 1, 4}),
                              .sharding = xla::HloSharding::Replicate()},
                             {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                              .shape = ArgShape({3, 4, 1}),
                              .sharding = xla::HloSharding::Replicate()}},
                  .outputs = {{.dtype =
                                   xla::ifrt::DType(xla::ifrt::DType::kF32),
                               .shape = ArgShape({2, 3, 1}),
                               .sharding = xla::HloSharding::Replicate()}},
                  .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<2x1x4xf32>,
                  %arg1: tensor<3x4x1xf32>) -> tensor<2x3x1xf32> {
    %0 = "stablehlo.dot_general"(%arg0, %arg1) {
      dot_dimension_numbers =
          #stablehlo.dot<lhs_contracting_dimensions = [2],
                         rhs_contracting_dimensions = [1]>
    } : (tensor<2x1x4xf32>, tensor<3x4x1xf32>) -> tensor<2x1x3x1xf32>
    %1 = stablehlo.transpose %0, dims = [2, 1, 0, 3]
        : (tensor<2x1x3x1xf32>) -> tensor<3x1x2x1xf32>
    %2 = stablehlo.reshape %1 : (tensor<3x1x2x1xf32>) -> tensor<3x1x2xf32>
    %3 = stablehlo.transpose %2, dims = [0, 2, 1]
        : (tensor<3x1x2xf32>) -> tensor<3x2x1xf32>
    %4 = stablehlo.transpose %3, dims = [1, 0, 2]
        : (tensor<3x2x1xf32>) -> tensor<2x3x1xf32>
    return %4 : tensor<2x3x1xf32>
  }
}
)mlir"),
                  .mlir_module_kept_var_idx = {0, 1},
                  .target_platforms = {"cpu"},
              },
          .assigned_devices =
              std::make_shared<const AssignedDevices>(AssignedDevices{
                  .num_replicas = 1,
                  .num_cores_per_replica = 1,
              }),
          .input_data =
              std::vector<HostBuffer>{
                  AsHostBuffer<float>({1.0, 0.0, 0.0, 0.0,
                                       2.0, 0.0, 0.0, 0.0},
                                      xla::ifrt::Shape({2, 1, 4})),
                  AsHostBuffer<float>({1.0, 0.0, 0.0, 0.0,
                                       2.0, 0.0, 0.0, 0.0,
                                       3.0, 0.0, 0.0, 0.0},
                                      xla::ifrt::Shape({3, 4, 1}))},
          .expected_outputs = std::vector<HostBuffer>{
              AsHostBuffer<float>({1.0, 2.0, 3.0,
                                   2.0, 4.0, 6.0},
                                  xla::ifrt::Shape({2, 3, 1}))},
          .use_xla_options_from_model = false,
      },
  };
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerExecuteTests, StableHloRunnerExecuteTest,
    ::testing::Combine(::testing::ValuesIn(GetExecuteTestParams()),
                       ::testing::Bool()),
    [](const ::testing::TestParamInfo<
        std::tuple<StableHloRunnerExecuteTestParams, bool>>& info) {
      std::string name = std::get<0>(info.param).test_name;
      if (std::get<1>(info.param)) {
        name += "Arena";
      }
      return name;
    });

using StableHloRunnerWithCoreSelectionTest =
    TestWithParam<std::tuple</*has_device_assignment=*/bool,
                             /*use_arena_transfer=*/bool>>;

TEST_P(StableHloRunnerWithCoreSelectionTest, CoreSelection) {
  // Set up.
  CellReader<int64_t> core_selection_count_reader(
      "/xla_serving/runtime/core_selection_count");
  const std::string file_path =
      tsl::GetDataDependencyFilepath(kSimpleModelPath);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
  stablehlo_spec.mlir_module = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  auto assigned_devices =
      std::get<0>(GetParam())
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
      device_selector.get(), ifrt_client->addressable_device_count(),
      /*record_core_selection_by_fingerprint=*/true);
  std::unique_ptr<H2DTransferExecutorFactory> h2d_factory;
  bool use_arena_transfer = std::get<1>(GetParam());
  if (use_arena_transfer) {
    h2d_factory = std::make_unique<MockH2DTransferExecutorFactory>();
  }

  tsl::monitoring::testing::CellReader<int64_t> input_transfer_counter(
      "/xla_serving/input_transfer/dispatched");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, assigned_devices,
          /*serving_core_selector=*/serving_core_selector.get(),
          std::move(h2d_factory)));
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
    std::vector<xla_serving::Value> value_inputs(input_data.begin(), input_data.end());
    TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> value_outputs,
                            runner->Execute(value_inputs, RunContext{}));
    for (int j = 0; j < value_outputs.size(); ++j) {
      EXPECT_THAT(value_outputs[j].Get<HostBuffer>(),
                  HostBufferEq(expected_outputs[j]));
    }
    TF_ASSERT_OK_AND_ASSIGN(
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
  std::vector<xla_serving::Value> value_inputs(input_data.begin(), input_data.end());
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> value_outputs,
                          runner->Execute(value_inputs, RunContext{}));
  for (int i = 0; i < value_outputs.size(); ++i) {
    EXPECT_THAT(value_outputs[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      const std::string program_fingerprint,
      StableHloRunnerTestPeer::GetProgramFingerprint(*runner));
  EXPECT_EQ(core_selection_count_reader.Delta(program_fingerprint, "1"), 1);

  int expected_executions = ifrt_client->addressable_device_count() + 1;
  if (use_arena_transfer) {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), expected_executions);
    EXPECT_EQ(input_transfer_counter.Delta("default"), 0);
  } else {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), 0);
    EXPECT_EQ(input_transfer_counter.Delta("default"), expected_executions);
  }
}

INSTANTIATE_TEST_SUITE_P(StableHloRunnerWithCoreSelectionTests,
                         StableHloRunnerWithCoreSelectionTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

using StableHloRunnerConcurrentExecuteTest =
    TestWithParam<std::tuple</*use_xla_options_from_model=*/bool,
                             /*use_arena_transfer=*/bool>>;

TEST_P(StableHloRunnerConcurrentExecuteTest, ConcurrentRuns) {
  const std::string file_path =
      tsl::GetDataDependencyFilepath(kSimpleDynamicShapeModelPath);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
      .mlir_module = hlo_str,
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
  bool use_xla_options_from_model = std::get<0>(GetParam());
  std::optional<xla::CompileOptionsProto> compile_options;
  if (use_xla_options_from_model) {
    compile_options = GetCompileOptions(assigned_devices);
  }
  std::unique_ptr<H2DTransferExecutorFactory> h2d_factory;
  bool use_arena_transfer = std::get<1>(GetParam());
  // Use arena transfer if true
  if (use_arena_transfer) {
    h2d_factory = std::make_unique<MockH2DTransferExecutorFactory>();
  }

  tsl::monitoring::testing::CellReader<int64_t> input_transfer_counter(
      "/xla_serving/input_transfer/dispatched");

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), compile_options, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, assigned_devices,
          /*serving_core_selector=*/nullptr, std::move(h2d_factory)));

  auto [promise1, future1] = tsl::MakePromise<>();

  auto runbase = [&](absl::Span<const HostBuffer> input_data,
                     absl::Span<const HostBuffer> expected_outputs) {
    std::vector<xla_serving::Value> value_inputs(input_data.begin(), input_data.end());
    TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> value_outputs,
                            runner->Execute(value_inputs, RunContext{}));

    EXPECT_EQ(value_outputs.size(), expected_outputs.size());
    for (int i = 0; i < value_outputs.size(); ++i) {
      EXPECT_THAT(value_outputs[i].Get<HostBuffer>(),
                  HostBufferEq(expected_outputs[i]));
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

  if (use_arena_transfer) {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), 2);
    EXPECT_EQ(input_transfer_counter.Delta("default"), 0);
  } else {
    EXPECT_EQ(input_transfer_counter.Delta("arena"), 0);
    EXPECT_EQ(input_transfer_counter.Delta("default"), 2);
  }
}

INSTANTIATE_TEST_SUITE_P(StableHloRunnerConcurrentExecuteTests,
                         StableHloRunnerConcurrentExecuteTest,
                         ::testing::Combine(::testing::Bool(),
                                            ::testing::Bool()));

TEST(StableHloRunnerTest, ArenaTransferFailure) {
  const std::string file_path =
      tsl::GetDataDependencyFilepath(kSimpleModelPath);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
  stablehlo_spec.mlir_module = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));
  auto assigned_devices =
      std::make_shared<const AssignedDevices>(AssignedDevices{
          .device_assignment = std::vector<int>{/*device0*/ 0, 0, 0, 0},
          .num_replicas = 1,
          .num_cores_per_replica = 1});
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices),
          ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, {},
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, assigned_devices,
          /*serving_core_selector=*/nullptr,
          std::make_unique<MockErrorH2DTransferExecutorFactory>()));

  const auto input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<xla_serving::Value> value_inputs(input_data.begin(), input_data.end());
  EXPECT_THAT(runner->Execute(value_inputs, RunContext{}),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Simulated Arena Transfer Failure")));
}

// Creates a runner for `simple.mlir` (`dot(arg0: f32[2,2], arg1: f32[2,8])`,
// returning the product and the product minus one) with both inputs
// replicated, on a single device unless `assigned_devices` says otherwise.
absl::StatusOr<std::unique_ptr<StableHloRunner>> CreateSimpleModelRunner(
    std::shared_ptr<xla::ifrt::Client> ifrt_client,
    IfrtLoadedVariableRegistry* ifrt_loaded_variable_registry,
    tsl::thread::ThreadPool* thread_pool,
    std::unique_ptr<H2DTransferExecutorFactory> h2d_factory,
    std::shared_ptr<const AssignedDevices> assigned_devices = nullptr) {
  const std::string file_path =
      tsl::GetDataDependencyFilepath(kSimpleModelPath);
  std::string hlo_str;
  TF_RETURN_IF_ERROR(
      tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
  stablehlo_spec.mlir_module = hlo_str;
  TF_ASSIGN_OR_RETURN(std::unique_ptr<StableHlo> stablehlo,
                      StableHlo::Create({stablehlo_spec}));
  if (assigned_devices == nullptr) {
    assigned_devices = std::make_shared<const AssignedDevices>(AssignedDevices{
        .device_assignment = std::vector<int>{/*device0*/ 0, 0, 0, 0},
        .num_replicas = 1,
        .num_cores_per_replica = 1});
  }
  return StableHloRunner::Create(
      std::move(stablehlo), GetCompileOptions(assigned_devices),
      std::move(ifrt_client),
      /*persistent_compilation_cache=*/nullptr, ifrt_loaded_variable_registry,
      {},
      /*compilation_environment_proto=*/nullptr, thread_pool,
      /*optimizer=*/nullptr, assigned_devices,
      /*serving_core_selector=*/nullptr, std::move(h2d_factory));
}

TEST(StableHloRunnerTest, DynamicDmaInputUsesExecutableOfStaticShape) {
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  auto recorded_transfers = std::make_shared<std::vector<RecordedTransfer>>();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      CreateSimpleModelRunner(
          ifrt_client, &ifrt_loaded_variable_registry, thread_pool.get(),
          std::make_unique<MockPaddingH2DTransferExecutorFactory>(
              recorded_transfers)));

  // arg0 carries a single row of data but is bound to the executable's static
  // shape f32[2,2] for this call; the second row is padding the model never
  // sees from the host.
  HostBuffer arg0 = AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2}));
  HostBuffer arg1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));
  const StableHloRunner::ExecuteOptions bind_arg0{
      .input_static_shapes = {xla::ifrt::Shape({2, 2}), std::nullopt}};
  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> outputs,
                          runner->Execute({xla_serving::Value(arg0), xla_serving::Value(arg1)},
                                          bind_arg0, RunContext{}));

  // The executor saw the actual data shape together with the static shape.
  ASSERT_EQ(recorded_transfers->size(), 2);
  EXPECT_EQ((*recorded_transfers)[0].shape, xla::ifrt::Shape({1, 2}));
  EXPECT_THAT((*recorded_transfers)[0].static_shape,
              Optional(xla::ifrt::Shape({2, 2})));
  EXPECT_EQ((*recorded_transfers)[1].shape, xla::ifrt::Shape({2, 8}));
  EXPECT_EQ((*recorded_transfers)[1].static_shape, std::nullopt);

  // Outputs have the static batch size; row 0 is the real result and row 1 is
  // whatever the padding produced (zeros, with the padding mock).
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_THAT(outputs[0].Get<HostBuffer>(),
              HostBufferEq(AsHostBuffer<float>(
                  {19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0,  //
                   0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0},
                  xla::ifrt::Shape({2, 8}))));
  EXPECT_THAT(outputs[1].Get<HostBuffer>(),
              HostBufferEq(AsHostBuffer<float>(
                  {18.0, 21.0, 24.0, 27.0, 30.0, 33.0, 36.0, 39.0,  //
                   -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0, -1.0},
                  xla::ifrt::Shape({2, 8}))));

  // A full f32[2,2] input (no static shape) reuses the executable compiled
  // for the static shape.
  HostBuffer full_arg0 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      outputs,
      runner->Execute({xla_serving::Value(full_arg0), xla_serving::Value(arg1)}, RunContext{}));
  EXPECT_THAT(outputs[0].Get<HostBuffer>(),
              HostBufferEq(AsHostBuffer<float>(
                  {19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0,  //
                   39.0, 46.0, 53.0, 60.0, 67.0, 74.0, 81.0, 88.0},
                  xla::ifrt::Shape({2, 8}))));
  // `GetProgramFingerprint` fails unless exactly one executable was cached.
  EXPECT_OK(StableHloRunnerTestPeer::GetProgramFingerprint(*runner));
}

TEST(StableHloRunnerTest, DynamicDmaRejectsInvalidStaticShapeOptions) {
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  auto recorded_transfers = std::make_shared<std::vector<RecordedTransfer>>();
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      CreateSimpleModelRunner(
          ifrt_client, &ifrt_loaded_variable_registry, thread_pool.get(),
          std::make_unique<MockPaddingH2DTransferExecutorFactory>(
              recorded_transfers)));
  HostBuffer arg0 = AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2}));
  HostBuffer arg1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));
  const std::vector<xla_serving::Value> inputs = {xla_serving::Value(arg0), xla_serving::Value(arg1)};

  // Wrong number of entries.
  EXPECT_THAT(
      runner->Execute(inputs,
                      StableHloRunner::ExecuteOptions{
                          .input_static_shapes = {xla::ifrt::Shape({2, 2})}},
                      RunContext{}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("one entry per user input")));

  // Host buffer does not fit the static shape (rank, dim 0, other dims).
  for (const xla::ifrt::Shape& bad_static_shape :
       {xla::ifrt::Shape({2}), xla::ifrt::Shape({0, 2}),
        xla::ifrt::Shape({2, 4})}) {
    EXPECT_THAT(runner->Execute(inputs,
                                StableHloRunner::ExecuteOptions{
                                    .input_static_shapes = {bad_static_shape,
                                                            std::nullopt}},
                                RunContext{}),
                StatusIs(absl::StatusCode::kInvalidArgument))
        << bad_static_shape;
  }

  // A static shape for a device-buffer input is rejected.
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      ifrt_client->addressable_devices().front(), xla::ifrt::MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client->MakeDeviceList(
                              {ifrt_client->addressable_devices().front()}));
  HostBuffer full_arg0 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      auto arg0_array,
      sharding_internal::MakeArrayFromHostBuffer(
          *ifrt_client, full_arg0, device_list, sharding, *thread_pool));
  EXPECT_THAT(
      runner->Execute(
          {xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client,
                                        .ifrt_array = std::move(arg0_array),
                                        .sharding = sharding,
                                        .device_list = device_list}),
           xla_serving::Value(arg1)},
          StableHloRunner::ExecuteOptions{
              .input_static_shapes = {xla::ifrt::Shape({2, 2}), std::nullopt}},
          RunContext{}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("is a device buffer")));
  EXPECT_TRUE(recorded_transfers->empty());
}

TEST(StableHloRunnerTest, DynamicDmaInputRequiresH2DTransferExecutor) {
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      CreateSimpleModelRunner(ifrt_client, &ifrt_loaded_variable_registry,
                              thread_pool.get(),
                              /*h2d_factory=*/nullptr));

  HostBuffer arg0 = AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2}));
  HostBuffer arg1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));
  EXPECT_THAT(
      runner->Execute(
          {xla_serving::Value(arg0), xla_serving::Value(arg1)},
          StableHloRunner::ExecuteOptions{
              .input_static_shapes = {xla::ifrt::Shape({2, 2}), std::nullopt}},
          RunContext{}),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("no H2DTransferExecutor is configured")));
}

TEST(StableHloRunnerTest, DynamicDmaInputRequiresSingleDeviceSharding) {
  TF_ASSERT_OK_AND_ASSIGN(auto ifrt_client, xla::ifrt::test_util::GetClient());
  ASSERT_GE(ifrt_client->addressable_devices().size(), 4);
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  auto recorded_transfers = std::make_shared<std::vector<RecordedTransfer>>();
  // Both inputs are replicated over four devices, so no input lives whole on
  // one device.
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      CreateSimpleModelRunner(
          ifrt_client, &ifrt_loaded_variable_registry, thread_pool.get(),
          std::make_unique<MockPaddingH2DTransferExecutorFactory>(
              recorded_transfers),
          std::make_shared<const AssignedDevices>(AssignedDevices{
              .device_assignment = std::vector<int>{
                  /*device0*/ 0, 0, 0, 0, /*device1*/ 0, 0, 0, 1,
                  /*device2*/ 1, 0, 0, 0, /*device3*/ 1, 0, 0, 1},
              .num_replicas = 1,
              .num_cores_per_replica = 4})));

  HostBuffer arg0 = AsHostBuffer<float>({1.0, 2.0}, xla::ifrt::Shape({1, 2}));
  HostBuffer arg1 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}));
  EXPECT_THAT(
      runner->Execute(
          {xla_serving::Value(arg0), xla_serving::Value(arg1)},
          StableHloRunner::ExecuteOptions{
              .input_static_shapes = {xla::ifrt::Shape({2, 2}), std::nullopt}},
          RunContext{}),
      StatusIs(absl::StatusCode::kUnimplemented,
               HasSubstr("spans more than one device")));
  EXPECT_TRUE(recorded_transfers->empty());

  // Without a static shape the same inputs run as usual on four devices.
  HostBuffer full_arg0 =
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}));
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla_serving::Value> outputs,
      runner->Execute({xla_serving::Value(full_arg0), xla_serving::Value(arg1)}, RunContext{}));
  ASSERT_EQ(outputs.size(), 2);
  EXPECT_THAT(outputs[0].Get<HostBuffer>(),
              HostBufferEq(AsHostBuffer<float>(
                  {19.0, 22.0, 25.0, 28.0, 31.0, 34.0, 37.0, 40.0,  //
                   39.0, 46.0, 53.0, 60.0, 67.0, 74.0, 81.0, 88.0},
                  xla::ifrt::Shape({2, 8}))));
}

class StableHloRunnerExecuteOptionsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    const std::string file_path =
        tsl::GetDataDependencyFilepath(kSimpleModelPath);
    std::string hlo_str;
    ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
    stablehlo_spec.mlir_module = hlo_str;
    TF_ASSERT_OK_AND_ASSIGN(stablehlo_, StableHlo::Create({stablehlo_spec}));

    assigned_devices_ = std::make_shared<const AssignedDevices>(AssignedDevices{
        .device_assignment = std::vector<int>{/*device0*/ 0, 0, 0, 0},
        .num_replicas = 1,
        .num_cores_per_replica = 1});

    TF_ASSERT_OK_AND_ASSIGN(ifrt_client_, xla::ifrt::test_util::GetClient());

    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
  }

  absl::StatusOr<std::unique_ptr<StableHloRunner>> CreateRunner(
      StablehloOptimizer* optimizer = nullptr) {
    return StableHloRunner::Create(
        std::move(stablehlo_), GetCompileOptions(assigned_devices_),
        ifrt_client_, /*persistent_compilation_cache=*/nullptr,
        &ifrt_loaded_variable_registry_, {},
        /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
        optimizer, assigned_devices_,
        /*serving_core_selector=*/nullptr);
  }

  std::unique_ptr<StableHlo> stablehlo_;
  std::shared_ptr<const AssignedDevices> assigned_devices_;
  std::shared_ptr<xla::ifrt::Client> ifrt_client_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteWithOptionsDefaultPass) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));

  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  for (int i = 0; i < output_values.size(); ++i) {
    EXPECT_THAT(output_values[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
}

TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteWithDeviceBufferInputsPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::DeviceListRef device_list,
      xla::ifrt::test_util::GetDevices(
          ifrt_client_.get(), *(assigned_devices_->device_assignment)));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list->devices().front(), xla::ifrt::MemoryKind());

  std::vector<xla_serving::DeviceBuffer> device_inputs;
  device_inputs.reserve(input_data.size());
  for (int i = 0; i < input_data.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto ifrt_array,
                            sharding_internal::MakeArrayFromHostBuffer(
                                *ifrt_client_, input_data[i], device_list,
                                sharding, *thread_pool_));
    device_inputs.push_back(xla_serving::DeviceBuffer{
        .ifrt_client = ifrt_client_,
        .ifrt_array = std::move(ifrt_array),
        .sharding = sharding,
        .device_list = device_list,
    });
  }

  std::vector<Value> input_values(device_inputs.begin(), device_inputs.end());

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));

  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  for (int i = 0; i < output_values.size(); ++i) {
    EXPECT_THAT(output_values[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
}

// A host buffer for one output only: that output is copied into the caller's
// memory, while the other is allocated by the runner as usual. This model is
// compiled on the JIT path and was not exported for input aliasing.
TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteCopiesIntoOutputHostBuffer) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<Value> input_values = {
      Value(
          AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}))),
      Value(AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0,
                                 10.0, 11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                                xla::ifrt::Shape({2, 8})))};

  HostBuffer destination = AsHostBuffer<float>(std::vector<float>(16, -1.0f),
                                               xla::ifrt::Shape({2, 8}));
  StableHloRunner::ExecuteOptions options;
  options.output_host_buffers = {std::nullopt, destination};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values, std::move(options), RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_THAT(output_values[0].Get<HostBuffer>(),
              HostBufferEq(AsHostBuffer<float>({19, 22, 25, 28, 31, 34, 37, 40,
                                                39, 46, 53, 60, 67, 74, 81, 88},
                                               xla::ifrt::Shape({2, 8}))));
  EXPECT_NE(output_values[0].Get<HostBuffer>().base(), destination.base());
  EXPECT_EQ(output_values[1].Get<HostBuffer>().base(), destination.base());
  EXPECT_THAT(destination,
              HostBufferEq(AsHostBuffer<float>({18, 21, 24, 27, 30, 33, 36, 39,
                                                38, 45, 52, 59, 66, 73, 80, 87},
                                               xla::ifrt::Shape({2, 8}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithOptionsAllDevicePlacementsPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values,
                      StableHloRunner::ExecuteOptions::AllDevice(2),
                      RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_EQ(output_values[0].Get<xla_serving::DeviceBuffer>().ifrt_array->shape(),
            xla::ifrt::Shape({2, 8}));
  EXPECT_EQ(output_values[1].Get<xla_serving::DeviceBuffer>().ifrt_array->shape(),
            xla::ifrt::Shape({2, 8}));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithOptionsAllHostPlacementsPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values, StableHloRunner::ExecuteOptions::AllHost(2),
                      RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_TRUE(output_values[0].IsType<HostBuffer>());
  EXPECT_TRUE(output_values[1].IsType<HostBuffer>());
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithManifestDeclaredOutputPlacementsPasses) {
  const std::string file_path =
      tsl::GetDataDependencyFilepath(kSimpleModelPath);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 2}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({2, 8}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate(),
                   .placement = Placement::kDevice},
                  {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({2, 8}),
                   .sharding = xla::HloSharding::Replicate(),
                   .placement = Placement::kHost}},
      .mlir_module_kept_var_idx = {0, 1},
  };
  stablehlo_spec.mlir_module = hlo_str;
  TF_ASSERT_OK_AND_ASSIGN(auto stablehlo, StableHlo::Create({stablehlo_spec}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          /*optimizer=*/nullptr, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};
  std::vector<Value> input_values(input_data.begin(), input_data.end());

  // Pass default ExecuteOptions (empty placements) to defer to manifest
  // placement.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values, StableHloRunner::ExecuteOptions{},
                      RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_TRUE(output_values[0].IsType<xla_serving::DeviceBuffer>());
  EXPECT_TRUE(output_values[1].IsType<HostBuffer>());
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithGranularOutputPlacementsPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  // Request output 0 on Host and output 1 on Device.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values,
                      StableHloRunner::ExecuteOptions{
                          .placements = {Placement::kHost, Placement::kDevice}},
                      RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_TRUE(output_values[0].IsType<HostBuffer>());
  EXPECT_TRUE(output_values[1].IsType<xla_serving::DeviceBuffer>());

  const auto expected_output_0 = AsHostBuffer<float>(
      {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
      xla::ifrt::Shape({2, 8}));
  EXPECT_THAT(output_values[0].Get<HostBuffer>(),
              HostBufferEq(expected_output_0));

  const xla_serving::DeviceBuffer& db1 = output_values[1].Get<xla_serving::DeviceBuffer>();
  TF_ASSERT_OK_AND_ASSIGN(auto host_buf_1,
                          sharding_internal::MakeHostBufferFromArray(
                              *ifrt_client_, *db1.ifrt_array, db1.sharding,
                              db1.device_list, *thread_pool_)
                              .Await());
  const auto expected_output_1 = AsHostBuffer<float>(
      {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
      xla::ifrt::Shape({2, 8}));
  EXPECT_THAT(host_buf_1, HostBufferEq(expected_output_1));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithGranularOutputTransferPoliciesPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  // Request output 0 on Device and output 1 on Host.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<Value> output_values,
      runner->Execute(input_values,
                      StableHloRunner::ExecuteOptions{
                          .placements = {Placement::kDevice, Placement::kHost}},
                      RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  EXPECT_TRUE(output_values[0].IsType<xla_serving::DeviceBuffer>());
  EXPECT_TRUE(output_values[1].IsType<HostBuffer>());

  const auto expected_output_1 = AsHostBuffer<float>(
      {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
      xla::ifrt::Shape({2, 8}));
  EXPECT_THAT(output_values[1].Get<HostBuffer>(),
              HostBufferEq(expected_output_1));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithInvalidOutputPlacementsSizeError) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  std::vector<Value> input_values(input_data.begin(), input_data.end());

  // Mismatched size (1 placement for 2 outputs)
  EXPECT_THAT(runner->Execute(input_values,
                              StableHloRunner::ExecuteOptions{
                                  .placements = {Placement::kHost}},
                              RunContext{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Expected 2 placements, but got 1")));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithDeviceBufferInputsReshapePasses) {
  auto mock_device_selector =
      std::make_unique<tsl::test_util::MockServingDeviceSelector>();
  auto optimizer = std::make_unique<MockStablehloOptimizer>();

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner(optimizer.get()));

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::DeviceListRef device_list,
      xla::ifrt::test_util::GetDevices(
          ifrt_client_.get(), *(assigned_devices_->device_assignment)));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list->devices().front(), xla::ifrt::MemoryKind());

  std::vector<xla_serving::DeviceBuffer> device_inputs;
  device_inputs.reserve(input_data.size());
  for (int i = 0; i < input_data.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(auto ifrt_array,
                            sharding_internal::MakeArrayFromHostBuffer(
                                *ifrt_client_, input_data[i], device_list,
                                sharding, *thread_pool_));
    device_inputs.push_back(xla_serving::DeviceBuffer{
        .ifrt_client = ifrt_client_,
        .ifrt_array = std::move(ifrt_array),
        .sharding = sharding,
        .device_list = device_list,
    });
  }

  std::vector<Value> input_values(device_inputs.begin(), device_inputs.end());

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));

  ASSERT_EQ(output_values.size(), 2);
  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  for (int i = 0; i < output_values.size(); ++i) {
    EXPECT_THAT(output_values[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithFlattened4DHostBufferInputsTransposesNhwcToNchw) {
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 2, 3}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape(1, 12),
                   .sharding = xla::HloSharding::Replicate()},
                  {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape(1, 12),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>, %arg1: tensor<1x2x2x3xf32>)
      -> (tensor<12xf32>, tensor<12xf32>) {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x3x2xf32>) -> tensor<12xf32>
    %1 = stablehlo.reshape %arg1 : (tensor<1x2x2x3xf32>) -> tensor<12xf32>
    return %0, %1 : tensor<12xf32>, tensor<12xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0, 1},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  FlattenInputsTestOptimizer optimizer(/*nchw_transposed_args=*/{0});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  // Input 0 is marked for the NCHW transpose and input 1 is not (zero-copy
  // reshaped).
  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f},
                          xla::ifrt::Shape({1, 2, 3, 2})),
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f},
                          xla::ifrt::Shape({1, 2, 2, 3}))};
  std::vector<Value> input_values(input_data.begin(), input_data.end());

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 2);

  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f,
                                        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f},
                                       xla::ifrt::Shape({12}))));
  EXPECT_THAT(
      output_values[1].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                       xla::ifrt::Shape({12}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithStridedFlattened4DHostBufferInputTransposesNhwcToNchw) {
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape(1, 12),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>) -> tensor<12xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x3x2xf32>) -> tensor<12xf32>
    return %0 : tensor<12xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  FlattenInputsTestOptimizer optimizer(/*nchw_transposed_args=*/{0});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  // Logical NHWC tensor x[h][w][c] = 1 + 6 * h + 2 * w + c, physically stored
  // channel-major (x[c][h][w]) and described by explicit byte strides.
  auto storage = std::make_shared<std::vector<float>>(
      std::vector<float>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 2.0f, 4.0f, 6.0f,
                         8.0f, 10.0f, 12.0f});
  constexpr int64_t kF32Bytes = sizeof(float);
  HostBuffer strided_input(
      storage->data(), xla::ifrt::DType(xla::ifrt::DType::kF32),
      xla::ifrt::Shape({1, 2, 3, 2}),
      absl::InlinedVector<int64_t, 4>{12 * kF32Bytes, 3 * kF32Bytes, kF32Bytes,
                                      6 * kF32Bytes},
      storage);
  std::vector<Value> input_values = {Value(std::move(strided_input))};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 1);

  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f,
                                        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f},
                                       xla::ifrt::Shape({12}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithFlattened4DHostBufferInputDoesNotTransposeWhenDisabled) {
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape(1, 12),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>) -> tensor<12xf32> {
    %0 = stablehlo.reshape %arg0 : (tensor<1x2x3x2xf32>) -> tensor<12xf32>
    return %0 : tensor<12xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  FlattenInputsTestOptimizer optimizer;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  // Without the NCHW input transpose, the 4D input is zero-copy reshaped to 1D
  // and stays in NHWC order.
  std::vector<Value> input_values = {
      Value(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f},
                                xla::ifrt::Shape({1, 2, 3, 2})))};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 1);

  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                       xla::ifrt::Shape({12}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithNchw4DHostBufferInputTransposesNhwcToNchw) {
  // The optimized module takes the input as 4D [N, C, H, W] (no R1
  // flattening), so the transposed host buffer matches the target shape
  // directly without reshaping.
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({1, 2, 2, 3}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    return %0 : tensor<1x2x2x3xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  NonFlattenTransposeTestOptimizer optimizer(/*nchw_transposed_args=*/{0});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  std::vector<Value> input_values = {
      Value(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f,
                                 9.0f, 10.0f, 11.0f, 12.0f},
                                xla::ifrt::Shape({1, 2, 3, 2})))};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 1);

  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f,
                                        2.0f, 4.0f, 6.0f, 8.0f, 10.0f, 12.0f},
                                       xla::ifrt::Shape({1, 2, 2, 3}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithNonFlattened4DHostBufferInputTransposesNhwcToNchw) {
  // Input 0 is 4D [1, 2, 3, 2] (NHWC) and gets transposed to [1, 2, 2, 3]
  // (NCHW) on the host before transfer, without R1 flattening. On the device,
  // the module transposes it back to NHWC [1, 2, 3, 2], matching what
  // `TpuInputNchwTransposePass` does.
  // Input 1 is 4D [1, 2, 2, 3] and is not transposed on the host.
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()},
                 {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 2, 3}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({1, 2, 3, 2}),
                   .sharding = xla::HloSharding::Replicate()},
                  {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({1, 2, 2, 3}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>, %arg1: tensor<1x2x2x3xf32>)
      -> (tensor<1x2x3x2xf32>, tensor<1x2x2x3xf32>) {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 3, 1] : (tensor<1x2x2x3xf32>) -> tensor<1x2x3x2xf32>
    return %1, %arg1 : tensor<1x2x3x2xf32>, tensor<1x2x2x3xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0, 1},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  NonFlattenTransposeTestOptimizer optimizer(/*nchw_transposed_args=*/{0});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f, 9.0f,
                           10.0f, 11.0f, 12.0f},
                          xla::ifrt::Shape({1, 2, 3, 2})),
      AsHostBuffer<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f, 60.0f, 70.0f,
                           80.0f, 90.0f, 100.0f, 110.0f, 120.0f},
                          xla::ifrt::Shape({1, 2, 2, 3}))};
  std::vector<Value> input_values(input_data.begin(), input_data.end());

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 2);

  // Input 0 was transposed NHWC -> NCHW on the host and NCHW -> NHWC on the
  // device, round-tripping the user's data exactly.
  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                       xla::ifrt::Shape({1, 2, 3, 2}))));
  // Input 1 was not transposed.
  EXPECT_THAT(
      output_values[1].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({10.0f, 20.0f, 30.0f, 40.0f, 50.0f,
                                        60.0f, 70.0f, 80.0f, 90.0f, 100.0f,
                                        110.0f, 120.0f},
                                       xla::ifrt::Shape({1, 2, 2, 3}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteWithStridedNonFlattened4DHostBufferInputTransposesNhwcToNchw) {
  StableHlo::Spec stablehlo_spec = {
      .inputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                  .shape = ArgShape({1, 2, 3, 2}),
                  .sharding = xla::HloSharding::Replicate()}},
      .outputs = {{.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                   .shape = ArgShape({1, 2, 3, 2}),
                   .sharding = xla::HloSharding::Replicate()}},
      .mlir_module = std::string(R"mlir(
module @main {
  func.func @main(%arg0: tensor<1x2x3x2xf32>) -> tensor<1x2x3x2xf32> {
    %0 = stablehlo.transpose %arg0, dims = [0, 3, 1, 2] : (tensor<1x2x3x2xf32>) -> tensor<1x2x2x3xf32>
    %1 = stablehlo.transpose %0, dims = [0, 2, 3, 1] : (tensor<1x2x2x3xf32>) -> tensor<1x2x3x2xf32>
    return %1 : tensor<1x2x3x2xf32>
  }
}
)mlir"),
      .mlir_module_kept_var_idx = {0},
  };
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHlo> stablehlo,
                          StableHlo::Create({stablehlo_spec}));

  NonFlattenTransposeTestOptimizer optimizer(/*nchw_transposed_args=*/{0});
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<StableHloRunner> runner,
      StableHloRunner::Create(
          std::move(stablehlo), GetCompileOptions(assigned_devices_),
          ifrt_client_, /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry_, {},
          /*compilation_environment_proto=*/nullptr, thread_pool_.get(),
          &optimizer, assigned_devices_,
          /*serving_core_selector=*/nullptr));

  // Logical NHWC tensor x[h][w][c] = 1 + 6 * h + 2 * w + c, physically stored
  // channel-major (x[c][h][w]) and described by explicit byte strides.
  auto storage = std::make_shared<std::vector<float>>(
      std::vector<float>{1.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 2.0f, 4.0f, 6.0f,
                         8.0f, 10.0f, 12.0f});
  constexpr int64_t kF32Bytes = sizeof(float);
  HostBuffer strided_input(
      storage->data(), xla::ifrt::DType(xla::ifrt::DType::kF32),
      xla::ifrt::Shape({1, 2, 3, 2}),
      absl::InlinedVector<int64_t, 4>{12 * kF32Bytes, 3 * kF32Bytes, kF32Bytes,
                                      6 * kF32Bytes},
      storage);
  std::vector<Value> input_values = {Value(std::move(strided_input))};

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));
  ASSERT_EQ(output_values.size(), 1);

  EXPECT_THAT(
      output_values[0].Get<HostBuffer>(),
      HostBufferEq(AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f,
                                        7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f},
                                       xla::ifrt::Shape({1, 2, 3, 2}))));
}

TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteWithMixedInputsPasses) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<HostBuffer> input_data = {
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})),
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))};

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::DeviceListRef device_list,
      xla::ifrt::test_util::GetDevices(
          ifrt_client_.get(), *(assigned_devices_->device_assignment)));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list->devices().front(), xla::ifrt::MemoryKind());

  std::vector<xla_serving::Value> input_values;

  // First input is HostBuffer.
  input_values.emplace_back(input_data[0]);

  // Second input is DeviceBuffer.
  TF_ASSERT_OK_AND_ASSIGN(
      auto ifrt_array,
      sharding_internal::MakeArrayFromHostBuffer(
          *ifrt_client_, input_data[1], device_list, sharding, *thread_pool_));
  input_values.emplace_back(
      xla_serving::DeviceBuffer{.ifrt_client = ifrt_client_,
                        .ifrt_array = std::move(ifrt_array),
                        .sharding = sharding,
                        .device_list = device_list});

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla_serving::Value> output_values,
                          runner->Execute(input_values, RunContext{}));

  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  for (int i = 0; i < output_values.size(); ++i) {
    EXPECT_THAT(output_values[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
}

// A host value type the runner doesn't know, standing in for a framework
// tensor (e.g. `tensorflow::Tensor`) whose support library registers a
// `HostBuffer` converter.
struct FakeHostTensor {
  HostBuffer buffer;
};

absl::StatusOr<HostBuffer> ConvertFakeHostTensor(const FakeHostTensor& tensor) {
  return tensor.buffer;
}

TEST_F(StableHloRunnerExecuteOptionsTest,
       ExecuteConvertsInputsWithRegisteredConverter) {
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedHostBufferConverterRegistration registration,
      ScopedHostBufferConverterRegistration::Create<FakeHostTensor>(
          "fake", ConvertFakeHostTensor));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<Value> input_values;
  input_values.emplace_back(FakeHostTensor{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}))});
  input_values.emplace_back(
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8})));

  TF_ASSERT_OK_AND_ASSIGN(std::vector<Value> output_values,
                          runner->Execute(input_values, RunContext{}));

  const auto expected_outputs = std::vector<HostBuffer>{
      AsHostBuffer<float>(
          {19, 22, 25, 28, 31, 34, 37, 40, 39, 46, 53, 60, 67, 74, 81, 88},
          xla::ifrt::Shape({2, 8})),
      AsHostBuffer<float>(
          {18, 21, 24, 27, 30, 33, 36, 39, 38, 45, 52, 59, 66, 73, 80, 87},
          xla::ifrt::Shape({2, 8}))};
  ASSERT_EQ(output_values.size(), expected_outputs.size());
  for (int i = 0; i < output_values.size(); ++i) {
    EXPECT_THAT(output_values[i].Get<HostBuffer>(),
                HostBufferEq(expected_outputs[i]));
  }
  // The caller's input is left as it was.
  EXPECT_TRUE(input_values[0].IsType<FakeHostTensor>());
}

TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteRejectsInputWithoutConverter) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<Value> input_values;
  input_values.emplace_back(
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2})));
  input_values.emplace_back(FakeHostTensor{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8}))});

  EXPECT_THAT(
      runner->Execute(input_values, RunContext{}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          "StableHloRunner inputs must be HostBuffer or xla_serving::DeviceBuffer. "
          "Input #1 is neither, and no HostBuffer converter is registered for "
          "its type. To pass tensorflow::Tensor inputs, link "
          "//third_party/neptune/runtime/tensorflow:"
          "interpreter_tensorflow_support (it registers the TF converter), or "
          "convert the input to a HostBuffer before calling."));
}

TEST_F(StableHloRunnerExecuteOptionsTest, ExecuteReportsConverterError) {
  TF_ASSERT_OK_AND_ASSIGN(
      ScopedHostBufferConverterRegistration registration,
      ScopedHostBufferConverterRegistration::Create<FakeHostTensor>(
          "broken_fake",
          [](const FakeHostTensor&) -> absl::StatusOr<HostBuffer> {
            absl::Status status =
                absl::FailedPreconditionError("unsupported fake dtype");
            status.SetPayload("type.googleapis.com/test.Detail",
                              absl::Cord("detail"));
            return status;
          }));
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<StableHloRunner> runner,
                          CreateRunner());

  std::vector<Value> input_values;
  input_values.emplace_back(FakeHostTensor{
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0}, xla::ifrt::Shape({2, 2}))});
  input_values.emplace_back(
      AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0,
                           11.0, 12.0, 13.0, 14.0, 15.0, 16.0},
                          xla::ifrt::Shape({2, 8})));

  // The converter's code and payloads are kept, with context added.
  const auto outputs = runner->Execute(input_values, RunContext{});
  EXPECT_THAT(outputs,
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       "Failed to convert StableHloRunner input #0 to "
                       "HostBuffer (converter 'broken_fake'): unsupported "
                       "fake dtype"));
  EXPECT_THAT(outputs.status().GetPayload("type.googleapis.com/test.Detail"),
              Optional(Eq(absl::Cord("detail"))));
}

TEST_P(StableHloRunnerCreateFailureTest, RunnerCreationFailure) {
  const StableHloRunnerExecuteTestParams& params = GetParam();

  const std::string file_path =
      tsl::GetDataDependencyFilepath(params.model_path);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  auto stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module = hlo_str;
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
                  /*optimizer=*/nullptr, params.assigned_devices,
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
            {
                .test_name = "ExecuteOutputValidationMismatch",
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
                              .shape = ArgShape(
                                  {2, 4}),  // Expect mismatch with {2, 8}
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
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0},
                                            xla::ifrt::Shape({2, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                             14.0, 15.0, 16.0},
                                            xla::ifrt::Shape({2, 8}))},
                .expected_outputs = absl::FailedPreconditionError(
                    "StableHlo expected arg#0 to have ArgSpec: "
                    "{dtype: F32, shape: {2, 8}, sharding: {replicated}}, "
                    "but got {dtype: F32, shape: {2, 4}, sharding: "
                    "{replicated}} "
                    "from model IO spec arg#0."),
            },
        }));

TEST_P(StableHloRunnerExecuteFailureTest, RunnerExecuteFailure) {
  const StableHloRunnerExecuteTestParams& params = GetParam();

  const std::string file_path =
      tsl::GetDataDependencyFilepath(params.model_path);
  std::string hlo_str;
  ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
  auto stablehlo_spec = params.stablehlo_spec;
  stablehlo_spec.mlir_module = hlo_str;
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

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::Device*> assigned_ifrt_devices,
      GetAssignedIfrtDevices(*ifrt_client,
                             params.assigned_devices->num_replicas,
                             params.assigned_devices->num_cores_per_replica,
                             params.assigned_devices->device_assignment));
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client->MakeDeviceList(assigned_ifrt_devices));

  std::vector<xla_serving::Value> values;
  if (params.input_format == kMixed) {
    for (int i = 0; i < params.input_data.size(); ++i) {
      if (i % 2 == 0) {
        values.push_back(xla_serving::Value(params.input_data[i]));
      } else {
        xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
            device_list, xla::ifrt::MemoryKind(),
            params.stablehlo_spec.inputs[i].sharding);
        TF_ASSERT_OK_AND_ASSIGN(auto ifrt_array,
                                sharding_internal::MakeArrayFromHostBuffer(
                                    *ifrt_client, params.input_data[i],
                                    device_list, sharding, *thread_pool));
        values.push_back(
            xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client,
                                         .ifrt_array = std::move(ifrt_array),
                                         .sharding = sharding,
                                         .device_list = device_list}));
      }
    }
  } else if (params.input_format == kAllHostBuffers) {
    values.assign(params.input_data.begin(), params.input_data.end());
  } else {
    // kAllDeviceBuffers case
    for (int i = 0; i < params.input_data.size(); ++i) {
      xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
          device_list, xla::ifrt::MemoryKind(),
          params.stablehlo_spec.inputs[i].sharding);
      TF_ASSERT_OK_AND_ASSIGN(
          auto ifrt_array, sharding_internal::MakeArrayFromHostBuffer(
                               *ifrt_client, params.input_data[i], device_list,
                               sharding, *thread_pool));
      values.push_back(
          xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client,
                                       .ifrt_array = std::move(ifrt_array),
                                       .sharding = sharding,
                                       .device_list = device_list}));
    }
  }
  EXPECT_THAT(runner->Execute(values,
                              StableHloRunner::ExecuteOptions{
                                  .placements = params.placements},
                              RunContext{}),
              StatusIs(params.expected_outputs.status().code(),
                       HasSubstr(params.expected_outputs.status().message())));
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerExecuteFailureTests, StableHloRunnerExecuteFailureTest,
    testing::ValuesIn<StableHloRunnerExecuteTestParams>(
        {
            {
                .test_name = "DeviceBufferInputDTypeMismatch",
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
                        AsHostBuffer<double>({1.0, 2.0, 3.0, 4.0},
                                             xla::ifrt::Shape({2, 2})),
                        AsHostBuffer<float>({1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0,
                                             8.0, 9.0, 10.0, 11.0, 12.0, 13.0,
                                             14.0, 15.0, 16.0},
                                            xla::ifrt::Shape({2, 8}))},
                .expected_outputs = absl::InvalidArgumentError(
                    "Input DType mismatch for device buffer at index 0. "
                    "Expected F32, but got F64"),
                .input_format = kAllDeviceBuffers,
            },
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
        tsl::GetDataDependencyFilepath(kSimpleDynamicShapeModelPath);
    std::string hlo_str;
    ASSERT_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &hlo_str));
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
        .mlir_module = hlo_str,
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
    std::vector<xla_serving::Value> value_inputs(input_data.begin(), input_data.end());
    TF_ASSIGN_OR_RETURN(std::vector<xla_serving::Value> value_outputs,
                        runner_->Execute(value_inputs, RunContext{}));
    EXPECT_EQ(value_outputs.size(), expected_outputs.size());
    for (int i = 0; i < value_outputs.size(); ++i) {
      EXPECT_THAT(value_outputs[i].Get<HostBuffer>(),
                  HostBufferEq(expected_outputs[i]));
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
    ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  }
  ASSERT_OK(runner_->MarkFullyInitialized());
  {
    ASSERT_OK(RunAndCheck(input_data, expected_outputs));
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

  ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  ASSERT_OK(runner_->MarkFullyInitialized());
  {
    EXPECT_THAT(
        RunAndCheck(new_input_data, expected_outputs),
        StatusIs(
            absl::StatusCode::kFailedPrecondition,
            HasSubstr(
                "Got new input shapes requiring a new executable, but can't "
                "create it after full initialization. Please warm up the model "
                "with all the input shapes that are possible during "
                "serving.")));
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
  ASSERT_OK(RunAndCheck(input_data, expected_outputs));
  ASSERT_OK(RunAndCheck(new_input_data, new_expected_outputs));
  ASSERT_OK(runner_->MarkFullyInitialized());
}

class StableHloRunnerAotTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(ifrt_client, xla::ifrt::test_util::GetClient());
    thread_pool = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "IfrtWorkerThreadPool", 16);
    assigned_devices = std::make_shared<const AssignedDevices>(
        AssignedDevices{.num_replicas = 1, .num_cores_per_replica = 1});
  }

  std::shared_ptr<xla::ifrt::Client> ifrt_client;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool;
  std::shared_ptr<const AssignedDevices> assigned_devices;
};

TEST_F(StableHloRunnerAotTest, AotCreateEmptyExecutablesFails) {
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = {},
              .assigned_devices = assigned_devices,
          },
      .executables = {},
      .stablehlo = CreateMockStableHloTestHelper({}, {}, 0),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No AOT executables provided.")));
}

TEST_F(StableHloRunnerAotTest, AotExecuteUnseenShapeCacheMissFails) {
  std::vector<std::string> weights = {"weight_0"};

  auto mock_exec = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec, name()).WillRepeatedly(Return("mock_exec"));
  EXPECT_CALL(*mock_exec, devices()).WillRepeatedly(Return(std::nullopt));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec)));

  std::vector<std::vector<int>> module_kept_var_indices = {{1}};

  std::vector<ArgSpec> inputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs),
                                     .outputs = std::move(outputs)});

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          AotStableHloRunner::Create(std::move(options)));

  std::vector<xla_serving::Value> values;
  auto host_buffer = AsHostBuffer<float>(std::vector<float>(99 * 99, 0.0f),
                                         xla::ifrt::Shape({99, 99}));
  values.push_back(xla_serving::Value(std::move(host_buffer)));

  EXPECT_THAT(runner->Execute(values, RunContext{}).status(),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       HasSubstr("AOT cache miss: unseen input shape.")));
}

struct AotExecutionTestParams {
  bool is_portable;
  int num_replicas;
  int num_cores_per_replica;
};

class StableHloRunnerAotExecuteTest
    : public StableHloRunnerAotTest,
      public ::testing::WithParamInterface<AotExecutionTestParams> {};

TEST_P(StableHloRunnerAotExecuteTest, ExecuteMatrix) {
  auto params = GetParam();
  std::vector<std::string> weights = {};

  auto mock_exec = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec, name()).WillRepeatedly(Return("mock_exec"));

  if (params.is_portable) {
    EXPECT_CALL(*mock_exec, devices()).WillRepeatedly(Return(std::nullopt));
    EXPECT_CALL(*mock_exec,
                Execute(testing::_, testing::_, testing::Optional(testing::_)))
        .WillOnce(
            testing::Return(absl::InternalError("Mock Execution Reached")));
  } else {
    TF_ASSERT_OK_AND_ASSIGN(
        auto device_list, ifrt_client->MakeDeviceList(ifrt_client->devices()));
    EXPECT_CALL(*mock_exec, devices())
        .WillRepeatedly(testing::Return(device_list));
    EXPECT_CALL(*mock_exec,
                Execute(testing::_, testing::_, testing::Eq(std::nullopt)))
        .WillOnce(
            testing::Return(absl::InternalError("Mock Execution Reached")));
  }

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0}};

  std::vector<ArgSpec> inputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs),
                                     .outputs = std::move(outputs)});

  const int total_devices = params.num_replicas * params.num_cores_per_replica;

  // TPU architecture requires physical device placement mapped to a 4D
  // coordinate space: [x, y, z, core].
  constexpr int kTopologyRank = 4;
  constexpr int kCoreIndex = 3;

  std::vector<int> device_attributes(total_devices * kTopologyRank, 0);
  for (int i = 0; i < total_devices; ++i) {
    device_attributes[i * kTopologyRank + kCoreIndex] = i;
  }

  auto dynamic_assigned_devices =
      std::make_shared<const AssignedDevices>(AssignedDevices{
          .device_assignment = std::move(device_attributes),
          .num_replicas = params.num_replicas,
          .num_cores_per_replica = params.num_cores_per_replica,
      });

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = dynamic_assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          AotStableHloRunner::Create(std::move(options)));

  std::vector<xla_serving::Value> values;
  auto host_buffer = AsHostBuffer<float>(std::vector<float>(2 * 8, 0.0f),
                                         xla::ifrt::Shape({2, 8}));
  values.push_back(xla_serving::Value(std::move(host_buffer)));

  EXPECT_THAT(runner->Execute(values, RunContext{}).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Mock Execution Reached")));
}

INSTANTIATE_TEST_SUITE_P(
    StableHloRunnerAotExecuteTests, StableHloRunnerAotExecuteTest,
    ::testing::Values(
        // Single-Device Portable.
        AotExecutionTestParams{/*is_portable=*/true, /*num_replicas=*/1,
                               /*num_cores_per_replica=*/1},
        // Single-Device Bound.
        AotExecutionTestParams{/*is_portable=*/false, /*num_replicas=*/1,
                               /*num_cores_per_replica=*/1},
        // Multi-Device Bound.
        AotExecutionTestParams{/*is_portable=*/false, /*num_replicas=*/1,
                               /*num_cores_per_replica=*/2}));

TEST_F(StableHloRunnerAotTest, AotCreateDroppedArguments) {
  // Simulate a model with 3 pre-DCE weights
  std::vector<std::string> weights = {"weight_0", "weight_1", "weight_2"};

  auto mock_exec = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec, name()).WillRepeatedly(Return("mock_exec"));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec)));

  // Simulate DCE dropping weight_1 and weight_2. So kept args are:
  // weight_0 (original idx 0) and user_input_0 (original idx 3)
  std::vector<std::vector<int>> module_kept_var_indices = {{0, 3}};

  std::vector<ArgSpec> inputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .ifrt_shape = xla::ifrt::Shape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .ifrt_shape = xla::ifrt::Shape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs),
                                     .outputs = std::move(outputs)});

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          AotStableHloRunner::Create(std::move(options)));
  EXPECT_EQ(runner->GetNumInputs(), 1);     // 1 user input
  EXPECT_EQ(runner->GetNumOutputs(), 1);    // 1 output
  EXPECT_EQ(runner->num_executables(), 1);  // 1 cached executable bundle
}

TEST_F(StableHloRunnerAotTest, AotExecuteMultiBucketRoutingSuccess) {
  std::vector<std::string> weights = {};

  auto mock_exec_0 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec_0, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec_0, name()).WillRepeatedly(Return("mock_exec_0"));
  EXPECT_CALL(*mock_exec_0, devices())
      .WillRepeatedly(testing::Return(std::nullopt));

  auto mock_exec_1 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec_1, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec_1, name()).WillRepeatedly(Return("mock_exec_1"));
  EXPECT_CALL(*mock_exec_1, devices())
      .WillRepeatedly(testing::Return(std::nullopt));

  EXPECT_CALL(*mock_exec_1,
              Execute(testing::_, testing::_, testing::Optional(testing::_)))
      .WillOnce(testing::Return(absl::InternalError("Mock Bucket 1 Reached")));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_0)));
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_1)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0}, {0}};

  std::vector<ArgSpec> inputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({1, 8}),
       .ifrt_shape = xla::ifrt::Shape({1, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({1, 8}),
       .ifrt_shape = xla::ifrt::Shape({1, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> inputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({4, 4}),
       .ifrt_shape = xla::ifrt::Shape({4, 4}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({4, 4}),
       .ifrt_shape = xla::ifrt::Shape({4, 4}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_0),
                                     .outputs = std::move(outputs_0)});
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_1),
                                     .outputs = std::move(outputs_1)});

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  TF_ASSERT_OK_AND_ASSIGN(auto runner,
                          AotStableHloRunner::Create(std::move(options)));

  std::vector<xla_serving::Value> values;
  auto host_buffer = AsHostBuffer<float>(std::vector<float>(4 * 4, 0.0f),
                                         xla::ifrt::Shape({4, 4}));
  values.push_back(xla_serving::Value(std::move(host_buffer)));

  EXPECT_THAT(runner->Execute(values, RunContext{}).status(),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Mock Bucket 1 Reached")));
}

TEST_F(StableHloRunnerAotTest, AotCreateMultiBucketMismatchOutputsFails) {
  std::vector<std::string> weights = {"weight_0"};

  auto mock_exec_0 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  auto mock_exec_1 = std::make_unique<xla::ifrt::MockLoadedExecutable>();

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_0)));
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_1)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0, 1}, {0, 1}};

  std::vector<ArgSpec> inputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> inputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_0),
                                     .outputs = std::move(outputs_0)});
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_1),
                                     .outputs = std::move(outputs_1)});

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched number of outputs across "
                                 "buckets")));
}

TEST_F(StableHloRunnerAotTest, AotCreateMultiBucketMismatchUserInputsFails) {
  std::vector<std::string> weights = {"weight_0"};

  auto mock_exec_0 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  auto mock_exec_1 = std::make_unique<xla::ifrt::MockLoadedExecutable>();

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_0)));
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec_1)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0, 1}, {0, 1, 2}};

  std::vector<ArgSpec> inputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> inputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs_1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_0),
                                     .outputs = std::move(outputs_0)});
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs_1),
                                     .outputs = std::move(outputs_1)});

  int num_weights = weights.size();
  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = std::move(weights),
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), num_weights),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched number of user inputs across "
                                 "buckets")));
}

TEST_F(StableHloRunnerAotTest, AotWeightSpecMismatchAcrossBucketsFails) {
  auto mock_exec0 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec0, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec0, name()).WillRepeatedly(Return("mock_exec0"));
  EXPECT_CALL(*mock_exec0, devices()).WillRepeatedly(Return(std::nullopt));

  auto mock_exec1 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec1, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec1, name()).WillRepeatedly(Return("mock_exec1"));
  EXPECT_CALL(*mock_exec1, devices()).WillRepeatedly(Return(std::nullopt));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec0)));
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec1)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0}, {0}};

  std::vector<ArgSpec> inputs0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> inputs1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 3}),  // Mismatch here
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs0),
                                     .outputs = std::move(outputs0)});
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs1),
                                     .outputs = std::move(outputs1)});

  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = {"weight_0"},
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), 1),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Weights should have the same dtype, shape "
                                 "and sharding among StableHLOs")));
}

TEST_F(StableHloRunnerAotTest, AotMultiBucketKeptWeightsInvariant) {
  auto mock_exec0 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec0, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec0, name()).WillRepeatedly(Return("mock_exec0"));
  EXPECT_CALL(*mock_exec0, devices()).WillRepeatedly(Return(std::nullopt));

  auto mock_exec1 = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec1, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec1, name()).WillRepeatedly(Return("mock_exec1"));
  EXPECT_CALL(*mock_exec1, devices()).WillRepeatedly(Return(std::nullopt));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec0)));
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec1)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0, 1, 2}, {0, 2}};

  std::vector<ArgSpec> inputs0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({4, 4}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs0 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> inputs1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()},
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({4, 4}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<ArgSpec> outputs1 = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};

  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs0),
                                     .outputs = std::move(outputs0)});
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs1),
                                     .outputs = std::move(outputs1)});

  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = {"weight_0", "weight_1"},
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), 2),
      .aot_memory_regions = {},
  };

  EXPECT_OK(AotStableHloRunner::Create(std::move(options)).status());
}

TEST_F(StableHloRunnerAotTest, AotCreateMismatchedArraySizesFails) {
  auto mock_exec = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec, name()).WillRepeatedly(Return("mock_exec"));
  EXPECT_CALL(*mock_exec, devices()).WillRepeatedly(Return(std::nullopt));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0}, {1}};
  std::vector<ArgSpec> input_dummy = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 8}),
       .sharding = xla::HloSharding::Replicate()}};
  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = input_dummy, .outputs = {}});
  io_specs.push_back(StableHloIOSpec{.inputs = input_dummy, .outputs = {}});

  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = {"weight_0"},
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), 1),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Mismatched array sizes in AOT options")));
}

TEST_F(StableHloRunnerAotTest, AotCreateIoSpecInputsSmallerThanWeightsFails) {
  auto mock_exec = std::make_unique<xla::ifrt::MockLoadedExecutable>();
  EXPECT_CALL(*mock_exec, client()).WillRepeatedly(Return(ifrt_client.get()));
  EXPECT_CALL(*mock_exec, name()).WillRepeatedly(Return("mock_exec"));
  EXPECT_CALL(*mock_exec, devices()).WillRepeatedly(Return(std::nullopt));

  std::vector<xla::ifrt::LoadedExecutableRef> executables;
  executables.push_back(xla::ifrt::LoadedExecutableRef(std::move(mock_exec)));

  std::vector<std::vector<int>> module_kept_var_indices = {{0}};
  std::vector<ArgSpec> inputs = {
      {.dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
       .shape = ArgShape({2, 2}),
       .sharding = xla::HloSharding::Replicate()}};
  std::vector<ArgSpec> outputs = {};
  std::vector<StableHloIOSpec> io_specs;
  io_specs.push_back(StableHloIOSpec{.inputs = std::move(inputs),
                                     .outputs = std::move(outputs)});

  StableHloRunnerAotOptions options{
      .base_options =
          {
              .client = ifrt_client,
              .thread_pool = thread_pool.get(),
              .ifrt_loaded_variable_registry = &ifrt_loaded_variable_registry,
              .weights = {"weight_0", "weight_1"},
              .assigned_devices = assigned_devices,
          },
      .executables = std::move(executables),
      .stablehlo = CreateMockStableHloTestHelper(
          std::move(module_kept_var_indices), std::move(io_specs), 2),
      .aot_memory_regions = {},
  };

  EXPECT_THAT(AotStableHloRunner::Create(std::move(options)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("StableHLO module has fewer inputs")));
}

//===----------------------------------------------------------------------===//
// Performance benchmarks below.
//===----------------------------------------------------------------------===//

static void BM_StableHloModel(benchmark::State& state,
                              const std::string& model_path,
                              StableHlo::Spec stablehlo_spec,
                              std::vector<HostBuffer> weights,
                              std::vector<HostBuffer> inputs) {
  auto thread_pool = std::make_unique<tsl::thread::ThreadPool>(
      tsl::Env::Default(), tsl::ThreadOptions(), "benchmark", 8);

  const std::string file_path =
      tsl::GetDataDependencyFilepath(model_path);
  auto ifrt_client = xla::ifrt::test_util::GetClient().value();

  std::string str;
  CHECK_OK(tsl::ReadFileToString(tsl::Env::Default(), file_path, &str));
  stablehlo_spec.mlir_module = str;

  std::unique_ptr<StableHlo> stablehlo =
      StableHlo::Create({stablehlo_spec}).value();

  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry;
  std::vector<std::string> weight_names;
  std::vector<int> device_ids = {0};  // Single device

  for (int i = 0; i < weights.size(); ++i) {
    std::string weight_name = absl::StrCat("weight_", i);
    weight_names.push_back(weight_name);
    auto weight_buffer = weights[i];
    auto hlo_sharding = stablehlo_spec.inputs[i].sharding;

    CHECK_OK(ifrt_loaded_variable_registry.TryRegisterLoadedVariable(
        {.device_ids = device_ids,
         .input_name = weight_name,
         .hlo_sharding = hlo_sharding},
        [ifrt_client, weight_buffer, device_ids, hlo_sharding, &thread_pool]()
            -> absl::StatusOr<IfrtLoadedVariableRegistry::LoadedVariable> {
          TF_ASSIGN_OR_RETURN(
              xla::ifrt::DeviceListRef device_list,
              xla::ifrt::test_util::GetDevices(ifrt_client.get(), device_ids));
          xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
              device_list, xla::ifrt::MemoryKind(), hlo_sharding);
          TF_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef array,
                              xla_serving::sharding_internal::MakeArrayFromHostBuffer(
                                  *ifrt_client, weight_buffer, device_ids,
                                  std::move(sharding), *thread_pool));
          auto [promise, future] = tsl::MakePromise<xla::ifrt::ArrayRef>();
          promise.Set(std::move(array));
          return IfrtLoadedVariableRegistry::LoadedVariable{
              .value = Value(std::move(future))};
        }));
  }

  std::unique_ptr<StableHloRunner> runner =
      StableHloRunner::Create(
          std::move(stablehlo),
          /*compile_options=*/std::nullopt, ifrt_client,
          /*persistent_compilation_cache=*/nullptr,
          &ifrt_loaded_variable_registry, weight_names,
          /*compilation_environment_proto=*/nullptr, thread_pool.get(),
          /*optimizer=*/nullptr, std::make_shared<const AssignedDevices>(),
          /*serving_core_selector=*/nullptr)
          .value();

  std::vector<xla_serving::Value> value_inputs(inputs.begin(), inputs.end());

  // Warm up the model to avoid compilation inside the benchmark loop.
  CHECK_OK(runner->Execute(value_inputs, RunContext{}).status());

  for (auto s : state) {
    CHECK_OK(runner->Execute(value_inputs, RunContext{}).status());
  }
}

static void BM_SimpleModel(benchmark::State& state) {
  int num_weights = state.range(0);
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

  std::vector<HostBuffer> weights;
  std::vector<HostBuffer> inputs;
  if (num_weights == 1) {
    weights.push_back(arg0);
    inputs.push_back(arg1);
  } else {
    inputs.push_back(arg0);
    inputs.push_back(arg1);
  }

  BM_StableHloModel(state, kSimpleModelPath, stablehlo_spec, weights, inputs);
}

BENCHMARK(BM_SimpleModel)->Arg(0)->Arg(1);

}  // namespace
}  // namespace xla_serving

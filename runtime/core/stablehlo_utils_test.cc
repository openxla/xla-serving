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

#include "runtime/core/stablehlo_utils.h"  // from @xla_serving

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/log_severity.h"  // from @abseil-cpp
#include "absl/log/globals.h"  // from @abseil-cpp
#include "absl/log/scoped_mock_log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tests/test_utils.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {
namespace {
using ::absl::ScopedMockLog;
using ::absl_testing::StatusIs;
using ::testing::_;
using ::testing::AtLeast;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;
using ::testing::TestWithParam;
using ::xla::ifrt::DType::Kind::kF16;
using ::xla::ifrt::DType::Kind::kF32;
using ::xla::ifrt::DType::Kind::kInvalid;
using ::xla::ifrt::DType::Kind::kOpaque;
using ::xla::ifrt::DType::Kind::kPred;
using ::xla::ifrt::DType::Kind::kString;
using ::xla::ifrt::DType::Kind::kToken;

constexpr const char* kMlirPathWithDynamicShape =
    "runtime/core/testdata/"
    "simple_with_dynamic_shape.mlir";
constexpr const char* kTupleArgsModelPath =
    "runtime/core/"
    "testdata/tuple_args.mlir";
constexpr const char* kTokenArgsModelPath =
    "runtime/core/"
    "testdata/token_args.mlir";

TEST(GetStableHloIOSpecTest, ModuleWithoutSpmd) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/core/testdata/"
      "simple.mlir");

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));
  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();

  EXPECT_EQ(spec.inputs.size(), 2);
  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 2}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = default_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = default_sharding,
      },
  };
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(GetStableHloIOSpecTest, SucceedsWhenModuleHasSpmd) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/core/testdata/"
      "spmd.mlir");

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));

  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();

  const std::string kOtherSharding = R"pb(
    type: OTHER
    tile_assignment_dimensions: 2
    tile_assignment_dimensions: 1
    iota_reshape_dims: 2
    iota_transpose_perm: 0
  )pb";

  TF_ASSERT_OK_AND_ASSIGN(
      xla::HloSharding other_sharding,
      xla::HloSharding::FromProto(
          *xla::ParseTextProto<xla::OpSharding>(kOtherSharding)));

  EXPECT_EQ(spec.inputs.size(), 2);

  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 2}),
          .sharding = other_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = other_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 8}),
          .sharding = default_sharding,
      },
  };
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(GetStableHloIOSpecTest, FailsWhenInputArgHasTupleType) {
  StableHloLoader loader;
  auto module = loader
                    .LoadFromFile(tsl::GetDataDependencyFilepath(
                        kTupleArgsModelPath))
                    .value();
  ASSERT_TRUE(module);

  EXPECT_THAT(GetStableHloIOSpec(*module, nullptr),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid XLA PrimitiveType: 13")));
}

TEST(GetStableHloIOSpecTest, SucceedsWhenInputArgHasTokenType) {
  StableHloLoader loader;
  auto module = loader
                    .LoadFromFile(tsl::GetDataDependencyFilepath(
                        kTokenArgsModelPath))
                    .value();
  ASSERT_TRUE(module);

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));
  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();

  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kToken),
          .shape = ArgShape({}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({}),
          .sharding = default_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kToken),
          .shape = ArgShape({}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({}),
          .sharding = default_sharding,
      },
  };
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(RefineShapesIfNeededTest, SkipWhenNoAttribute) {
  const int old_log_level = absl::SetVLogLevel("stablehlo_utils", 2);
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_jax_model_fn {
    func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
      return %0 : tensor<f32>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);
  ScopedMockLog log;
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, _,
          HasSubstr(
              "Module does not have attribute: jax.uses_shape_polymorphism")))
      .Times(AtLeast(1));
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, _,
          HasSubstr(
              "Skipping shape refinement because attribute "
              "jax.uses_shape_polymorphism doesn't exist or set to false.")))
      .Times(AtLeast(1));
  log.StartCapturingLogs();

  EXPECT_OK(RefineShapesIfNeeded(
      *module, /*weights=*/{}, /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
      /*input_dtypes=*/{xla::ifrt::DType(kF32)}, /*module_kept_var_idx=*/{0}));

  absl::SetVLogLevel("stablehlo_utils", old_log_level);
}

TEST(RefineShapesIfNeededTest, SkipWhenAttributeFalse) {
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_fn attributes {jax.uses_shape_polymorphism = false} {
    func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
      return %0 : tensor<f32>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);
  ScopedMockLog log;
  EXPECT_CALL(
      log,
      Log(absl::LogSeverity::kInfo, _,
          HasSubstr(
              "Skipping shape refinement because attribute "
              "jax.uses_shape_polymorphism doesn't exist or set to false.")))
      .Times(AtLeast(1));
  log.StartCapturingLogs();

  EXPECT_OK(RefineShapesIfNeeded(
      *module, /*weights=*/{}, /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
      /*input_dtypes=*/{xla::ifrt::DType(kF32)}, /*module_kept_var_idx=*/{0}));
}

TEST(RefineShapesIfNeededTest, FailsWhenAttributeNotBoolean) {
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_fn attributes {jax.uses_shape_polymorphism = 1} {
    func.func @main(%arg0: tensor<f32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
      return %0 : tensor<f32>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                           /*input_dtypes=*/{xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("jax.uses_shape_polymorphism value is not a boolean")));
}

TEST(RefineShapesIfNeededTest, FailsWhenNoMainFunction) {
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_fn attributes {jax.uses_shape_polymorphism = true} {
    func.func @not_main(%arg0: tensor<f32>) -> tensor<f32> {
      %0 = stablehlo.add %arg0, %arg0 : tensor<f32>
      return %0 : tensor<f32>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(RefineShapesIfNeeded(*module, /*weights=*/{},
                                   /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                                   /*input_dtypes=*/{xla::ifrt::DType(kF32)},
                                   /*module_kept_var_idx=*/{0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("No main function found in the module")));
}

TEST(RefineShapesIfNeededTest, Succeeds) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);
  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  EXPECT_OK(RefineShapesIfNeeded(
      *module, /*weights=*/{},
      /*input_shapes=*/
      {xla::ifrt::Shape({3, 2}), xla::ifrt::Shape({2, 3}),
       xla::ifrt::Shape({})},
      /*input_dtypes=*/
      {xla::ifrt::DType(kF32), xla::ifrt::DType(kF32), xla::ifrt::DType(kF32)},
      /*module_kept_var_idx=*/{0, 1, 2}));

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));
  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();
  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({3, 2}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({2, 3}),
          .sharding = default_sharding,
      },
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({}),
          .sharding = default_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kF32),
          .shape = ArgShape({3, 3}),
          .sharding = default_sharding,
      },
  };
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(RefineShapesIfNeededTest, SuccessWithConcurrentRuns) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);
  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  // Create a thread pool after the module is loaded to avoid data race on MLIR
  // context at destruction time.
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), "test_thread_pool",
                                      2);

  // Run two concurrent runs with different input shapes.
  auto [promise1, future1] = tsl::MakePromise<>();
  auto run1 = [&]() {
    mlir::OwningOpRef<mlir::ModuleOp> module_copy =
        mlir::OwningOpRef<mlir::ModuleOp>(module->clone());
    TF_ASSERT_OK(
        RefineShapesIfNeeded(*module_copy, /*weights=*/{},
                             /*input_shapes=*/
                             {xla::ifrt::Shape({3, 2}),
                              xla::ifrt::Shape({2, 3}), xla::ifrt::Shape({})},
                             /*input_dtypes=*/
                             {xla::ifrt::DType(kF32), xla::ifrt::DType(kF32),
                              xla::ifrt::DType(kF32)},
                             /*module_kept_var_idx=*/{0, 1, 2}));
    TF_ASSERT_OK_AND_ASSIGN(auto spec,
                            GetStableHloIOSpec(*module_copy, nullptr));
    const xla::HloSharding default_sharding = xla::HloSharding::Replicate();
    const std::vector<ArgSpec> expected_inputs = {
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({3, 2}),
            .sharding = default_sharding,
        },
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({2, 3}),
            .sharding = default_sharding,
        },
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({}),
            .sharding = default_sharding,
        },
    };
    const std::vector<ArgSpec> expected_outputs = {
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({3, 3}),
            .sharding = default_sharding,
        },
    };
    EXPECT_EQ(spec.inputs.size(), 3);
    EXPECT_EQ(spec.outputs.size(), 1);
    EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
    EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
    promise1.Set(absl::OkStatus());
  };
  auto [promise2, future2] = tsl::MakePromise<>();
  auto run2 = [&]() {
    mlir::OwningOpRef<mlir::ModuleOp> module_copy =
        mlir::OwningOpRef<mlir::ModuleOp>(module->clone());
    TF_ASSERT_OK(
        RefineShapesIfNeeded(*module_copy, /*weights=*/{},
                             /*input_shapes=*/
                             {xla::ifrt::Shape({4, 2}),
                              xla::ifrt::Shape({2, 3}), xla::ifrt::Shape({})},
                             /*input_dtypes=*/
                             {xla::ifrt::DType(kF32), xla::ifrt::DType(kF32),
                              xla::ifrt::DType(kF32)},
                             /*module_kept_var_idx=*/{0, 1, 2}));
    TF_ASSERT_OK_AND_ASSIGN(auto spec,
                            GetStableHloIOSpec(*module_copy, nullptr));
    const xla::HloSharding default_sharding = xla::HloSharding::Replicate();
    const std::vector<ArgSpec> expected_inputs = {
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({4, 2}),
            .sharding = default_sharding,
        },
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({2, 3}),
            .sharding = default_sharding,
        },
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({}),
            .sharding = default_sharding,
        },
    };
    const std::vector<ArgSpec> expected_outputs = {
        ArgSpec{
            .dtype = xla::ifrt::DType(kF32),
            .shape = ArgShape({4, 3}),
            .sharding = default_sharding,
        },
    };
    EXPECT_EQ(spec.inputs.size(), 3);
    EXPECT_EQ(spec.outputs.size(), 1);
    EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
    EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
    promise2.Set(absl::OkStatus());
  };

  thread_pool.Schedule(run1);
  thread_pool.Schedule(run2);
  EXPECT_OK(future1.Await());
  EXPECT_OK(future2.Await());
}

TEST(RefineShapesIfNeededTest, SuccessWithBooleanArgs) {
  // The stablehlo main is exported by a Jax function with a boolean argument
  // (dtypenp=np.bool_).
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {
    func.func public @main(%arg0: tensor<?x?xi1>) -> tensor<?x?xi1> {
      return %arg0 : tensor<?x?xi1>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);

  EXPECT_OK(RefineShapesIfNeeded(
      *module, /*weights=*/{}, /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
      /*input_dtypes=*/{xla::ifrt::DType(kPred)}, /*module_kept_var_idx=*/{0}));

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));
  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();
  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kPred),
          .shape = ArgShape({2, 2}),
          .sharding = default_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kPred),
          .shape = ArgShape({2, 2}),
          .sharding = default_sharding,
      },
  };
  EXPECT_EQ(spec.inputs.size(), 1);
  EXPECT_EQ(spec.outputs.size(), 1);
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(RefineShapesIfNeededTest, SucceedsWhenNotAllInputsUsed) {
  // Test the case where not all the inputs are required by the StableHlo
  // "main" function.

  // The stablehlo main is exported by a Jax function with a boolean argument
  // (dtypenp=np.bool_).
  constexpr absl::string_view stablehlo_str = R"(
    module @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {
      func.func public @main(%arg0: tensor<?x?xi1>) -> tensor<?x?xi1> {
        return %arg0 : tensor<?x?xi1>
      }
    })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);

  EXPECT_OK(
      RefineShapesIfNeeded(*module,
                           /*weights=*/{},
                           /*input_shapes=*/
                           {xla::ifrt::Shape({2, 2}), xla::ifrt::Shape({2, 2})},
                           /*input_dtypes=*/
                           {xla::ifrt::DType(kPred), xla::ifrt::DType(kPred)},
                           /*module_kept_var_idx=*/{0}));

  TF_ASSERT_OK_AND_ASSIGN(auto spec, GetStableHloIOSpec(*module, nullptr));
  const xla::HloSharding default_sharding = xla::HloSharding::Replicate();
  const std::vector<ArgSpec> expected_inputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kPred),
          .shape = ArgShape({2, 2}),
          .sharding = default_sharding,
      },
  };
  const std::vector<ArgSpec> expected_outputs = {
      ArgSpec{
          .dtype = xla::ifrt::DType(kPred),
          .shape = ArgShape({2, 2}),
          .sharding = default_sharding,
      },
  };
  EXPECT_EQ(spec.inputs.size(), 1);
  EXPECT_EQ(spec.outputs.size(), 1);
  EXPECT_THAT(spec.inputs, ElementsAreArray(expected_inputs));
  EXPECT_THAT(spec.outputs, ElementsAreArray(expected_outputs));
}

TEST(RefineShapesIfNeededTest, FailsWhenInputTypeUnsupported) {
  constexpr absl::string_view stablehlo_str = R"(
  module @jit_f.0 attributes {jax.uses_shape_polymorphism = true} {
    func.func public @main(%arg0: tensor<?x?xf32>) -> tensor<?x?xf32> {
      return %arg0 : tensor<?x?xf32>
    }
  })";
  StableHloLoader loader;
  auto module = loader.Load(stablehlo_str).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                           /*input_dtypes=*/{xla::ifrt::DType(kInvalid)},
                           /*module_kept_var_idx=*/{0}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Type: INVALID is not supported.")));
  EXPECT_THAT(RefineShapesIfNeeded(*module, /*weights=*/{},
                                   /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                                   /*input_dtypes=*/{xla::ifrt::DType(kOpaque)},
                                   /*module_kept_var_idx=*/{0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type: OPAQUE is not supported.")));
  EXPECT_THAT(RefineShapesIfNeeded(*module, /*weights=*/{},
                                   /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                                   /*input_dtypes=*/{xla::ifrt::DType(kToken)},
                                   /*module_kept_var_idx=*/{0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type: TOKEN is not supported.")));
  EXPECT_THAT(RefineShapesIfNeeded(*module, /*weights=*/{},
                                   /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                                   /*input_dtypes=*/{xla::ifrt::DType(kString)},
                                   /*module_kept_var_idx=*/{0}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type: STRING is not supported.")));
}

TEST(RefineShapesIfNeededTest, FailsWhenInputNumLessThanModuleKeptVarIdx) {
  // Validates that invalid argument failure gets returned when the number of
  // inputs is less than the number of module kept variable indices.

  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  // Return failures since the number of inputs is 1, but the number of module
  // kept variable indices is 3, which should never happen.
  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                           /*input_dtypes=*/{xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0, 1, 2}),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("The total number of weights and inputs should be equal "
                    "or more than the number of module kept var indices: ")));
}

TEST(RefineShapesIfNeededTest, FailsWhenNumOfInputLessThanMainFuncArgs) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/{xla::ifrt::Shape({2, 2})},
                           /*input_dtypes=*/{xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Number of main function arguments does not match "
                         "number of inputs: 3 vs 1")));
}

TEST(RefineShapesIfNeededTest, FailsWhenInputDTypeMismatch) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/
                           {xla::ifrt::Shape({3, 2}), xla::ifrt::Shape({2, 3}),
                            xla::ifrt::Shape({})},
                           /*input_dtypes=*/
                           {xla::ifrt::DType(kF32), xla::ifrt::DType(kF16),
                            xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0, 1, 2}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("refinement element types must match in "
                         "tensor<?x3xf32> -> tensor<2x3xf16>")));
}

TEST(RefineShapesIfNeededTest, RefineShapeInputRankMismatch) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/
                           {xla::ifrt::Shape({3, 2}), xla::ifrt::Shape({3}),
                            xla::ifrt::Shape({})},
                           /*input_dtypes=*/
                           {xla::ifrt::DType(kF32), xla::ifrt::DType(kF32),
                            xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0, 1, 2}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("refinement rank must match operand rank in "
                         "tensor<?x3xf32> -> tensor<3xf32>")));
}

TEST(RefineShapesIfNeededTest, FailsWhenInputDimensionSizeMismatch) {
  // The main function argument types are: tensor<?x?xf32>, tensor<?x3xf32> and
  // tensor<f32>.
  const std::string kMlirPath =
      tsl::GetDataDependencyFilepath(kMlirPathWithDynamicShape);

  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);

  EXPECT_THAT(
      RefineShapesIfNeeded(*module, /*weights=*/{},
                           /*input_shapes=*/
                           {xla::ifrt::Shape({2, 3}), xla::ifrt::Shape({3, 2}),
                            xla::ifrt::Shape({})},
                           /*input_dtypes=*/
                           {xla::ifrt::DType(kF32), xla::ifrt::DType(kF32),
                            xla::ifrt::DType(kF32)},
                           /*module_kept_var_idx=*/{0, 1, 2}),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("refinement dimension sizes must match for static "
                         "dimensions in tensor<?x3xf32> -> tensor<3x2xf32>")));
}

TEST(StableHloUtilsTest, SetPlatformIndex) {
  const std::string kMlirPath = tsl::GetDataDependencyFilepath(
      "runtime/core/testdata/"
      "linear_multi_platform.mlir");
  StableHloLoader loader;
  auto module = loader.LoadFromFile(kMlirPath).value();
  ASSERT_TRUE(module);
  auto main = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main);
  ASSERT_EQ(main.getNumArguments(), 4);

  // Save the input types before the platform index is set.
  std::vector<mlir::Type> input_types;
  for (int i = 1; i < main.getNumArguments(); ++i) {
    input_types.push_back(main.getArgument(i).getType());
  }

  TF_ASSERT_OK(SetPlatformIndex(module.get(), {"cpu", "gpu"}, "gpu"));

  // Argument 0 is now removed.
  EXPECT_EQ(main.getNumArguments(), 3);

  // The previous arguments are shifted by 1.
  for (int i = 0; i < main.getNumArguments(); ++i) {
    EXPECT_EQ(main.getArgument(i).getType(), input_types[i]);
  }

  // The first operation is a const op.
  auto first_op = main.getBody().getOps().begin();
  EXPECT_TRUE(llvm::isa<mlir::stablehlo::ConstantOp>(*first_op));

  // The const op has the correct value.
  auto dense_attr = first_op->getAttrOfType<mlir::DenseElementsAttr>("value");
  auto range = dense_attr.getValues<int32_t>();
  ASSERT_EQ(range.size(), 1);
  EXPECT_EQ(range[0], 1);
}

struct ValidateIOSpecTestParam {
  std::string test_name;
  std::vector<ArgSpec> stablehlo_arg_specs;
  std::vector<ArgSpec> model_arg_specs;
  std::vector<int> module_kept_var_idx = {};
  absl::Status expected_status;
};

using ValidateIOSpecTest = ::testing::TestWithParam<ValidateIOSpecTestParam>;

TEST_P(ValidateIOSpecTest, ValidateIOSpec) {
  const ValidateIOSpecTestParam& param = GetParam();
  absl::Status status =
      ValidateIOSpecs(param.stablehlo_arg_specs, param.model_arg_specs,
                      param.module_kept_var_idx);
  EXPECT_THAT(status, StatusIs(param.expected_status.code()));
  if (!status.ok()) {
    EXPECT_THAT(status.message(), HasSubstr(param.expected_status.message()));
  }
}

INSTANTIATE_TEST_SUITE_P(
    ValidateIOSpecTests, ValidateIOSpecTest,
    testing::ValuesIn<ValidateIOSpecTestParam>(
        {
            {.test_name = "IOArgSpecMatchAndSupported",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .expected_status = absl::OkStatus()},
            {.test_name = "IOArgSpecMatchAndSupportedWithModuleKeptVarIdx",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({3, 5}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .module_kept_var_idx = {0, 1},
             .expected_status = absl::OkStatus()},
            {.test_name = "IOArgSpecDTypeUnsupported",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kToken),
                         .shape = ArgShape({}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kToken),
                         .shape = ArgShape({}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .expected_status =
                 absl::InvalidArgumentError("Type: TOKEN is not supported.")},
            {.test_name = "IOArgSpecSizeMismatch",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .expected_status = absl::FailedPreconditionError(
                 "StableHlo expected 1 args, but got 2 from model IO spec")},
            {.test_name = "IOArgSpecDTypeMismatch",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF16),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .expected_status = absl::FailedPreconditionError(
                 "StableHlo expected arg#0 to have ArgSpec: {dtype: F32, "
                 "shape: {2, 2}, sharding: {replicated}}, but got {dtype: F16, "
                 "shape: {2, 2}, sharding: {replicated}} from model IO spec")},
            {.test_name = "IOArgSpecShapeMismatch",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, std::nullopt}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .expected_status = absl::FailedPreconditionError(
                 "StableHlo expected arg#0 to have ArgSpec: {dtype: F32, "
                 "shape: {2, 2}, sharding: {replicated}}, but got {dtype: F32, "
                 "shape: {2, nullopt}, sharding: {replicated}} from model IO "
                 "spec")},
            {.test_name = "IOArgSpecShardingMismatch",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Unknown(),
                     },
                 },
             .expected_status = absl::FailedPreconditionError(
                 "StableHlo expected arg#0 to have ArgSpec: {dtype: F32, "
                 "shape: {2, 2}, sharding: {replicated}}, but got {dtype: F32, "
                 "shape: {2, 2}, sharding: {unknown}} from model IO spec")},
            {.test_name = "ModuleKeptVarIdxSizeMismatch",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 8}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .module_kept_var_idx = {1},
             .expected_status = absl::FailedPreconditionError(
                 "StableHlo expected 2 args, but got 1 from model IO spec.")},
            {.test_name = "ModuleKeptVarIdxOutOfRange",
             .stablehlo_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .model_arg_specs =
                 {
                     ArgSpec{
                         .dtype = xla::ifrt::DType(kF32),
                         .shape = ArgShape({2, 2}),
                         .sharding = xla::HloSharding::Replicate(),
                     },
                 },
             .module_kept_var_idx = {1},
             .expected_status = absl::InvalidArgumentError(
                 "Module kept var idx 1 is out of range [0, 1).")},
        }),
    [](const testing::TestParamInfo<ValidateIOSpecTestParam>& info) {
      return info.param.test_name;
    });

struct InputShapeEqualTestParam {
  std::string test_name;
  ArgShape stablehlo_input_shape;
  xla::ifrt::Shape input_shape;
  absl::Status expected_status;
};

using InputShapeEqualTest = ::testing::TestWithParam<InputShapeEqualTestParam>;

TEST_P(InputShapeEqualTest, CheckShapesEqual) {
  const InputShapeEqualTestParam& param = GetParam();
  absl::Status status =
      InputShapeEqual(param.stablehlo_input_shape, param.input_shape);
  EXPECT_THAT(status, StatusIs(param.expected_status.code()));
  if (!status.ok()) {
    EXPECT_THAT(status.message(), HasSubstr(param.expected_status.message()));
  }
}

INSTANTIATE_TEST_SUITE_P(
    InputShapeEqualTests, InputShapeEqualTest,
    testing::ValuesIn<InputShapeEqualTestParam>({
        {.test_name = "InputMatch",
         .stablehlo_input_shape = ArgShape({2, 2}),
         .input_shape = xla::ifrt::Shape({2, 2}),
         .expected_status = absl::OkStatus()},
        {.test_name = "InputMatchWithDynamicShape",
         .stablehlo_input_shape = ArgShape({2, std::nullopt}),
         .input_shape = xla::ifrt::Shape({2, 2}),
         .expected_status = absl::OkStatus()},
        {.test_name = "InputShapeRankMismatch",
         .stablehlo_input_shape = ArgShape({2, 2}),
         .input_shape = xla::ifrt::Shape({2, 2, 3}),
         .expected_status = absl::FailedPreconditionError(
             "StableHlo expected input shape rank: 2, but got: 3 from the "
             "actual input.")},
        {.test_name = "InputShapeMismatch",
         .stablehlo_input_shape = ArgShape({2, 2}),
         .input_shape = xla::ifrt::Shape({2, 3}),
         .expected_status = absl::FailedPreconditionError(
             "StableHlo expected input shape dimension #1: 2, but got: 3 from "
             "the actual input.")},
    }),
    [](const testing::TestParamInfo<InputShapeEqualTestParam>& info) {
      return info.param.test_name;
    });

TEST(MaybeToIfrtShapeTest, StaticAndDynamicShapes) {
  EXPECT_EQ(MaybeToIfrtShape(ArgShape({})), xla::ifrt::Shape({}));
  EXPECT_EQ(MaybeToIfrtShape(ArgShape({2, 4})), xla::ifrt::Shape({2, 4}));
  EXPECT_EQ(MaybeToIfrtShape(ArgShape({2, std::nullopt})), std::nullopt);
  EXPECT_EQ(ToIfrtShape(ArgShape({2, 4})), xla::ifrt::Shape({2, 4}));
  EXPECT_DEATH(ToIfrtShape(ArgShape({2, std::nullopt})), "dynamic");
}

}  // namespace
}  // namespace xla_serving

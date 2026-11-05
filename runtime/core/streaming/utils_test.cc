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

#include "runtime/core/streaming/utils.h"  // from @xla_serving

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Eq;
using ::testing::HasSubstr;

TEST(StableHloStreamResultsBatchingTest, BasicPropagation) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) {
        stablehlo.custom_call @xla_serving_stream(%arg0) {
          mhlo.backend_config = {callback_id = 123 : i64, output_names = "out"}
        } : (tensor<4xf32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/true));

  // Verify main function signature changes
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));

  auto step_id_arg = main_func.getArgument(1);
  auto step_id_type =
      llvm::dyn_cast<mlir::RankedTensorType>(step_id_arg.getType());
  ASSERT_TRUE(step_id_type);
  EXPECT_THAT(step_id_type.getRank(), Eq(1));
  EXPECT_THAT(step_id_type.getShape()[0], Eq(4));
  EXPECT_TRUE(step_id_type.getElementType().isSignlessInteger(32));

  // Verify custom call changes
  mlir::stablehlo::CustomCallOp custom_call;
  main_func.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() == "xla_serving_stream") {
      custom_call = op;
    }
  });
  ASSERT_TRUE(custom_call);
  ASSERT_THAT(custom_call.getInputs().size(), Eq(2));
  EXPECT_THAT(custom_call.getInputs()[1], Eq(step_id_arg));

  // Verify backend_config changes
  auto backend_config =
      custom_call->getAttrOfType<mlir::DictionaryAttr>("mhlo.backend_config");
  ASSERT_TRUE(backend_config);
  auto output_names_attr =
      backend_config.getAs<mlir::StringAttr>("output_names");
  ASSERT_TRUE(output_names_attr);
  EXPECT_THAT(output_names_attr.getValue().str(),
              Eq("out,__streamed_result_batched_step_id"));
  auto callback_id_attr =
      backend_config.getAs<mlir::IntegerAttr>("callback_id");
  ASSERT_TRUE(callback_id_attr);
  EXPECT_THAT(callback_id_attr.getInt(), Eq(123));
}

TEST(StableHloStreamResultsBatchingTest, ScalarDefaultToBatchSizeOne) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<f32>) {
        stablehlo.custom_call @xla_serving_stream(%arg0) {
          mhlo.backend_config = {callback_id = 123 : i64, output_names = "out"}
        } : (tensor<f32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/true));

  // Verify main function signature changes (should default to batch size 1)
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));

  auto step_id_arg = main_func.getArgument(1);
  auto step_id_type =
      llvm::dyn_cast<mlir::RankedTensorType>(step_id_arg.getType());
  ASSERT_TRUE(step_id_type);
  EXPECT_THAT(step_id_type.getRank(), Eq(1));
  EXPECT_THAT(step_id_type.getShape()[0], Eq(1));  // Defaulted to 1!
}

TEST(StableHloStreamResultsBatchingTest, UnbatchedScalarPropagation) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) {
        stablehlo.custom_call @xla_serving_stream(%arg0) {
          mhlo.backend_config = {callback_id = 123 : i64, output_names = "out"}
        } : (tensor<4xf32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/false));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));

  auto step_id_arg = main_func.getArgument(1);
  auto step_id_type =
      llvm::dyn_cast<mlir::RankedTensorType>(step_id_arg.getType());
  ASSERT_TRUE(step_id_type);
  EXPECT_THAT(step_id_type.getRank(), Eq(0));
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughIf) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>, %pred: tensor<i1>) {
        "stablehlo.if"(%pred) ({
          stablehlo.custom_call @xla_serving_stream(%arg0) {
            mhlo.backend_config = {
              callback_id = 123 : i64, output_names = "out"
            }
          } : (tensor<4xf32>) -> ()
          stablehlo.return
        }, {
          stablehlo.return
        }) : (tensor<i1>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  // %arg0 (tensor<4xf32>), %pred (tensor<i1>), plus step_id makes it 3
  // arguments!
  ASSERT_THAT(main_func.getNumArguments(), Eq(3));
  auto step_id_arg = main_func.getArgument(2);

  // Find custom call inside the IF block
  mlir::stablehlo::CustomCallOp custom_call;
  main_func.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() == "xla_serving_stream") {
      custom_call = op;
    }
  });
  ASSERT_TRUE(custom_call);
  ASSERT_THAT(custom_call.getInputs().size(), Eq(2));
  EXPECT_THAT(custom_call.getInputs()[1],
              Eq(step_id_arg));  // Uses the step_id from main!
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughWhile) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) {
        %c0 = stablehlo.constant dense<0> : tensor<i32>
        %c10 = stablehlo.constant dense<10> : tensor<i32>
        %result:2 = stablehlo.while(
            %iterArg0 = %c0, %iterArg1 = %arg0) : tensor<i32>, tensor<4xf32>
          cond {
            %cond = stablehlo.compare LT, %iterArg0, %c10
                : (tensor<i32>, tensor<i32>) -> tensor<i1>
            stablehlo.return %cond : tensor<i1>
          } do {
            %c1 = stablehlo.constant dense<1> : tensor<i32>
            %next_i = stablehlo.add %iterArg0, %c1 : tensor<i32>
            // Stream inside loop
            stablehlo.custom_call @xla_serving_stream(%iterArg1) {
              mhlo.backend_config = {
                callback_id = 123 : i64, output_names = "out"
              }
            } : (tensor<4xf32>) -> ()
            stablehlo.return %next_i, %iterArg1 : tensor<i32>, tensor<4xf32>
          }
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/true));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));  // %arg0 + step_id
  auto step_id_arg = main_func.getArgument(1);

  // Verify the while op has been updated
  mlir::stablehlo::WhileOp while_op;
  main_func.walk([&](mlir::stablehlo::WhileOp op) { while_op = op; });
  ASSERT_TRUE(while_op);

  // Should have 3 loop carriers now (i, x, step_id)
  ASSERT_THAT(while_op.getOperands().size(), Eq(3));
  EXPECT_THAT(while_op.getOperands()[2], Eq(step_id_arg));

  // Verify cond block signature
  mlir::Block& cond_block = while_op.getCond().front();
  ASSERT_THAT(cond_block.getNumArguments(), Eq(3));
  auto cond_step_id_type = llvm::dyn_cast<mlir::RankedTensorType>(
      cond_block.getArgument(2).getType());
  ASSERT_TRUE(cond_step_id_type);
  EXPECT_TRUE(cond_step_id_type.getElementType().isSignlessInteger(32));
  EXPECT_THAT(cond_step_id_type.getShape()[0], Eq(4));

  // Verify body block signature and custom call usage
  mlir::Block& body_block = while_op.getBody().front();
  ASSERT_THAT(body_block.getNumArguments(), Eq(3));
  auto body_step_id = body_block.getArgument(2);

  mlir::stablehlo::CustomCallOp custom_call;
  body_block.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() == "xla_serving_stream") {
      custom_call = op;
    }
  });
  ASSERT_TRUE(custom_call);
  ASSERT_THAT(custom_call.getInputs().size(), Eq(2));
  EXPECT_THAT(custom_call.getInputs()[1],
              Eq(body_step_id));  // Uses the loop carrier step_id!

  // Verify body terminator returns the step_id
  auto return_op =
      llvm::cast<mlir::stablehlo::ReturnOp>(body_block.getTerminator());
  ASSERT_THAT(return_op.getNumOperands(), Eq(3));
  EXPECT_THAT(return_op.getOperand(2), Eq(body_step_id));
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughWhileUnbatched) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>) {
        %c0 = stablehlo.constant dense<0> : tensor<i32>
        %c10 = stablehlo.constant dense<10> : tensor<i32>
        %result:2 = stablehlo.while(
            %iterArg0 = %c0, %iterArg1 = %arg0) : tensor<i32>, tensor<4xf32>
          cond {
            %cond = stablehlo.compare LT, %iterArg0, %c10
                : (tensor<i32>, tensor<i32>) -> tensor<i1>
            stablehlo.return %cond : tensor<i1>
          } do {
            %c1 = stablehlo.constant dense<1> : tensor<i32>
            %next_i = stablehlo.add %iterArg0, %c1 : tensor<i32>
            // Stream inside loop
            stablehlo.custom_call @xla_serving_stream(%iterArg1) {
              mhlo.backend_config = {
                callback_id = 123 : i64, output_names = "out"
              }
            } : (tensor<4xf32>) -> ()
            stablehlo.return %next_i, %iterArg1 : tensor<i32>, tensor<4xf32>
          }
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/false));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));  // %arg0 + step_id
  auto step_id_arg = main_func.getArgument(1);

  // Verify the while op has been updated
  mlir::stablehlo::WhileOp while_op;
  main_func.walk([&](mlir::stablehlo::WhileOp op) { while_op = op; });
  ASSERT_TRUE(while_op);

  // Should have 3 loop carriers now (i, x, step_id)
  ASSERT_THAT(while_op.getOperands().size(), Eq(3));
  EXPECT_THAT(while_op.getOperands()[2], Eq(step_id_arg));

  // Verify cond block signature (scalar step_id rank = 0)
  mlir::Block& cond_block = while_op.getCond().front();
  ASSERT_THAT(cond_block.getNumArguments(), Eq(3));
  auto cond_step_id_type = llvm::dyn_cast<mlir::RankedTensorType>(
      cond_block.getArgument(2).getType());
  ASSERT_TRUE(cond_step_id_type);
  EXPECT_TRUE(cond_step_id_type.getElementType().isSignlessInteger(32));
  EXPECT_THAT(cond_step_id_type.getRank(), Eq(0));
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughCase) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @main(%arg0: tensor<4xf32>, %index: tensor<i32>) {
        "stablehlo.case"(%index) ({
          stablehlo.custom_call @xla_serving_stream(%arg0) {
            mhlo.backend_config = {
              callback_id = 123 : i64, output_names = "out"
            }
          } : (tensor<4xf32>) -> ()
          stablehlo.return
        }, {
          stablehlo.return
        }) : (tensor<i32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module));

  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  // %arg0 (tensor<4xf32>), %index (tensor<i32>), plus step_id -> 3 arguments
  ASSERT_THAT(main_func.getNumArguments(), Eq(3));
  auto step_id_arg = main_func.getArgument(2);

  // Find custom call inside the CASE block
  mlir::stablehlo::CustomCallOp custom_call;
  main_func.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() == "xla_serving_stream") {
      custom_call = op;
    }
  });
  ASSERT_TRUE(custom_call);
  ASSERT_THAT(custom_call.getInputs().size(), Eq(2));
  EXPECT_THAT(custom_call.getInputs()[1], Eq(step_id_arg));
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughCall) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @called_fn(%arg0: tensor<4xf32>) {
        stablehlo.custom_call @xla_serving_stream(%arg0) {
          mhlo.backend_config = {
            callback_id = 123 : i64, output_names = "out"
          }
        } : (tensor<4xf32>) -> ()
        return
      }
      func.func @main(%arg0: tensor<4xf32>) {
        func.call @called_fn(%arg0) : (tensor<4xf32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module));

  // Verify main function signature changes
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));
  auto main_step_id_arg = main_func.getArgument(1);

  // Verify called_fn signature changes
  auto called_func = module->lookupSymbol<mlir::func::FuncOp>("called_fn");
  ASSERT_TRUE(called_func);
  ASSERT_THAT(called_func.getNumArguments(), Eq(2));
  auto called_step_id_arg = called_func.getArgument(1);

  // Verify the call op in main passes the step_id
  mlir::func::CallOp call_op;
  main_func.walk([&](mlir::func::CallOp op) {
    call_op = op;
  });
  ASSERT_TRUE(call_op);
  ASSERT_THAT(call_op.getOperands().size(), Eq(2));
  EXPECT_THAT(call_op.getOperand(1), Eq(main_step_id_arg));

  // Verify the custom call in called_fn uses the called_fn's step_id argument
  mlir::stablehlo::CustomCallOp custom_call;
  called_func.walk([&](mlir::stablehlo::CustomCallOp op) {
    if (op.getCallTargetName() == "xla_serving_stream") {
      custom_call = op;
    }
  });
  ASSERT_TRUE(custom_call);
  ASSERT_THAT(custom_call.getInputs().size(), Eq(2));
  EXPECT_THAT(custom_call.getInputs()[1], Eq(called_step_id_arg));
}

TEST(StableHloStreamResultsBatchingTest, PropagateThroughRecursiveCall) {
  const std::string mlir_src = R"mlir(
    module {
      func.func @recursive_fn(%arg0: tensor<4xf32>, %depth: tensor<i32>) {
        %c0 = stablehlo.constant dense<0> : tensor<i32>
        %cond = stablehlo.compare GT, %depth, %c0
            : (tensor<i32>, tensor<i32>) -> tensor<i1>
        "stablehlo.if"(%cond) ({
          func.call @recursive_fn(%arg0, %depth)
              : (tensor<4xf32>, tensor<i32>) -> ()
          stablehlo.return
        }, {
          stablehlo.custom_call @xla_serving_stream(%arg0) {
            mhlo.backend_config = {
              callback_id = 123 : i64, output_names = "out"
            }
          } : (tensor<4xf32>) -> ()
          stablehlo.return
        }) : (tensor<i1>) -> ()
        return
      }
      func.func @main(%arg0: tensor<4xf32>, %depth: tensor<i32>) {
        func.call @recursive_fn(%arg0, %depth)
            : (tensor<4xf32>, tensor<i32>) -> ()
        return
      }
    }
  )mlir";

  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(auto module, loader.Load(mlir_src));
  ASSERT_TRUE(module);

  TF_ASSERT_OK(PropagateStreamingStepId(*module));

  // Verify main function signature changes
  auto main_func = module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(3));  // %arg0, %depth, %step_id
  auto main_step_id_arg = main_func.getArgument(2);

  // Verify recursive_fn signature changes
  auto recursive_func =
      module->lookupSymbol<mlir::func::FuncOp>("recursive_fn");
  ASSERT_TRUE(recursive_func);
  ASSERT_THAT(recursive_func.getNumArguments(), Eq(3));  // arg0, depth, step_id
  auto recursive_step_id_arg = recursive_func.getArgument(2);

  // Verify the call in main passes step_id
  mlir::func::CallOp main_call;
  main_func.walk([&](mlir::func::CallOp op) {
    if (op.getCallee() == "recursive_fn") {
      main_call = op;
    }
  });
  ASSERT_TRUE(main_call);
  ASSERT_THAT(main_call.getOperands().size(), Eq(3));
  EXPECT_THAT(main_call.getOperand(2), Eq(main_step_id_arg));

  // Verify the recursive call inside recursive_fn passes its own step_id arg
  mlir::func::CallOp recursive_call;
  recursive_func.walk([&](mlir::func::CallOp op) {
    if (op.getCallee() == "recursive_fn" && op != main_call) {
      recursive_call = op;
    }
  });
  ASSERT_TRUE(recursive_call);
  ASSERT_THAT(recursive_call.getOperands().size(), Eq(3));
  EXPECT_THAT(recursive_call.getOperand(2), Eq(recursive_step_id_arg));
}

TEST(CreateRequestIdsBufferTest, Success) {
  int64_t batch_size = 4;
  int32_t request_id = 42;
  TF_ASSERT_OK_AND_ASSIGN(xla_serving::HostBuffer buffer,
                          CreateRequestIdsBuffer(batch_size, request_id));

  EXPECT_THAT(buffer.dtype().kind(), Eq(xla::ifrt::DType::Kind::kS32));
  EXPECT_THAT(buffer.shape().dims(), ::testing::ElementsAre(batch_size));

  const int32_t* data = reinterpret_cast<const int32_t*>(buffer.base());
  for (int i = 0; i < batch_size; ++i) {
    EXPECT_THAT(data[i], Eq(request_id));
  }
}

constexpr absl::string_view kFlattenedR1StreamingMlir = R"mlir(
  module {
    func.func @main(%arg0: tensor<12xf32>) {
      stablehlo.custom_call @xla_serving_stream(%arg0) {
        mhlo.backend_config = {callback_id = 123 : i64, output_names = "out"}
      } : (tensor<12xf32>) -> ()
      return
    }
  }
)mlir";

TEST(StableHloStreamResultsBatchingTest,
     PropagateStreamingStepIdUsesExplicitBatchSizeForFlattenedR1Input) {
  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          loader.Load(kFlattenedR1StreamingMlir));
  ASSERT_TRUE(module);

  ASSERT_OK(PropagateStreamingStepId(*module, /*is_batched=*/true,
                                     /*batch_size=*/3));

  mlir::func::FuncOp main_func =
      module->lookupSymbol<mlir::func::FuncOp>("main");
  ASSERT_TRUE(main_func);
  ASSERT_THAT(main_func.getNumArguments(), Eq(2));

  auto step_id_type = llvm::dyn_cast<mlir::RankedTensorType>(
      main_func.getArgument(1).getType());
  ASSERT_TRUE(step_id_type);
  EXPECT_THAT(step_id_type.getRank(), Eq(1));
  EXPECT_THAT(step_id_type.getShape()[0], Eq(3));
  EXPECT_TRUE(step_id_type.getElementType().isSignlessInteger(32));
}

TEST(StableHloStreamResultsBatchingTest,
     PropagateStreamingStepIdReturnsErrorWhenExplicitBatchSizeIsNonPositive) {
  xla_serving::StableHloLoader loader;
  TF_ASSERT_OK_AND_ASSIGN(mlir::OwningOpRef<mlir::ModuleOp> module,
                          loader.Load(kFlattenedR1StreamingMlir));
  ASSERT_TRUE(module);

  EXPECT_THAT(PropagateStreamingStepId(*module, /*is_batched=*/true,
                                       /*batch_size=*/0),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Streaming batch_size must be positive")));
}

}  // namespace
}  // namespace xla_serving

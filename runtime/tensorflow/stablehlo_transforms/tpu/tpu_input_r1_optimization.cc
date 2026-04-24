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

#include "runtime/tensorflow/stablehlo_transforms/tpu/tpu_input_r1_optimization.h"  // from @neptune

#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "llvm/include/llvm/ADT/STLExtras.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/ADT/StringRef.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "llvm/include/llvm/Support/CommandLine.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/Pass.h"  // from @llvm-project
#include "mlir/include/mlir/Pass/PassRegistry.h"  // from @llvm-project
#include "mlir/include/mlir/Support/TypeID.h"  // from @llvm-project
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @neptune
#include "runtime/libs/mlir/utils.h"  // from @neptune
#include "shardy/dialect/sdy/ir/dialect.h"  // from @shardy
#include "shardy/dialect/sdy/ir/utils.h"  // from @shardy
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/hlo/parser/hlo_parser.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"  // from @xla
#include "xla/service/spmd/shardy/utils.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

namespace {

absl::StatusOr<xla::OpSharding> ParseOpSharding(
    mlir::StringAttr sharding_attr) {
  xla::OpSharding op_sharding;
  if (!op_sharding.ParseFromString(sharding_attr.getValue())) {
    absl::StatusOr<xla::HloSharding> hlo_sharding =
        xla::ParseSharding(sharding_attr.getValue().str());
    if (!hlo_sharding.ok()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to parse sharding: ", sharding_attr.getValue().str()));
    }
    op_sharding = hlo_sharding->ToProto();
  }
  return op_sharding;
}

xla::OpSharding GetShardingFromShardyAttr(
    mlir::sdy::TensorShardingAttr sharding_attr, mlir::func::FuncOp func) {
  if (sharding_attr) {
    xla::HloSharding hloSharding = xla::sdy::convertToHloSharding(
        sharding_attr,
        [&](mlir::sdy::TensorShardingAttr sharding) {
          return sharding.getMesh(func);
        },
        /*manualAxes=*/{});
    return hloSharding.ToProto();
  } else {
    return xla::HloSharding::Replicate().ToProto();
  }
}

absl::StatusOr<bool> ShouldRunTpuInputR1Optimization(
    mlir::func::FuncOp main_func,
    const xla::CompileOptionsProto* compile_options) {
  for (int i = 0; i < main_func.getNumArguments(); ++i) {
    TF_ASSIGN_OR_RETURN(auto shape, neptune::sharding_internal::ToMlirShape(
                                        main_func.getArgumentTypes()[i]));
    xla::OpSharding op_sharding;
    if (mlir::StringAttr sharding_attr =
            main_func.getArgAttrOfType<mlir::StringAttr>(i, "mhlo.sharding")) {
      TF_ASSIGN_OR_RETURN(op_sharding, ParseOpSharding(sharding_attr));
      if (op_sharding.type() != xla::OpSharding::OTHER ||
          op_sharding.tile_assignment_dimensions().empty()) {
        continue;
      }
    } else if (compile_options && compile_options->executable_build_options()
                                      .use_shardy_partitioner()) {
      mlir::sdy::TensorShardingAttr sharding_attr =
          mlir::sdy::getSharding(main_func.getArgument(i));
      xla::sdy::adjustInputSharding(
          main_func, i, sharding_attr, shape.size(),
          compile_options->executable_build_options()
              .allow_spmd_sharding_propagation_to_parameters());
      op_sharding = GetShardingFromShardyAttr(sharding_attr, main_func);
    }
    // R1 optimization can only be applied if the input is sharded on the
    // first dimension.
    // Consider if `replicate_on_last_tile_dim` should be supported as well.
    for (int j = 1; j < op_sharding.tile_assignment_dimensions().size(); ++j) {
      if (op_sharding.tile_assignment_dimensions(j) > 1) {
        LOG(INFO) << "Skipping R1 optimization because the input " << i
                  << " is sharded on dimension: " << j
                  << ". Only sharding on dimension 0 is supported.";
        return false;
      }
    }
  }
  return true;
}

absl::Status MaybeUpdateArgumentSharding(
    mlir::func::FuncOp main_func, int arg_idx, mlir::OpBuilder& builder,
    bool enable_serialize_sharding,
    const xla::CompileOptionsProto* compile_options) {
  TF_ASSIGN_OR_RETURN(auto shape, neptune::sharding_internal::ToMlirShape(
                                      main_func.getArgumentTypes()[arg_idx]));
  if (mlir::StringAttr sharding_attr =
          main_func.getArgAttrOfType<mlir::StringAttr>(arg_idx,
                                                       "mhlo.sharding")) {
    TF_ASSIGN_OR_RETURN(xla::OpSharding op_sharding,
                        ParseOpSharding(sharding_attr));
    if (op_sharding.type() != xla::OpSharding::OTHER ||
        op_sharding.tile_assignment_dimensions().empty()) {
      return absl::OkStatus();
    }
    // R1 optimization can only be applied if the input is sharded on the first
    // dimension.
    int dim = op_sharding.tile_assignment_dimensions().Get(0);
    for (int i = 1; i < op_sharding.tile_assignment_dimensions().size(); ++i) {
      if (op_sharding.tile_assignment_dimensions(i) > 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("sharding on dimension: ", i,
                         " is not supported when R1 optimization is enabled."));
      }
    }
    op_sharding.clear_tile_assignment_dimensions();
    op_sharding.mutable_tile_assignment_dimensions()->Add(dim);
    if (enable_serialize_sharding) {
      main_func.setArgAttr(
          arg_idx, "mhlo.sharding",
          builder.getStringAttr(op_sharding.SerializeAsString()));
    } else {
      TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                          xla::HloSharding::FromProto(op_sharding));
      main_func.setArgAttr(arg_idx, "mhlo.sharding",
                           builder.getStringAttr(hlo_sharding.ToString()));
    }
  } else if (compile_options && compile_options->executable_build_options()
                                    .use_shardy_partitioner()) {
    mlir::sdy::TensorShardingAttr sharding =
        mlir::sdy::getSharding(main_func.getArgument(arg_idx));
    xla::sdy::adjustInputSharding(
        main_func, arg_idx, sharding, shape.size(),
        compile_options->executable_build_options()
            .allow_spmd_sharding_propagation_to_parameters());
    if (sharding.getDimShardings().empty()) {
      return absl::OkStatus();
    }
    // R1 optimization can only be applied if the input is sharded on the first
    // dimension.
    for (const auto& [idx, dim_sharding] :
         llvm::enumerate(sharding.getDimShardings().drop_front())) {
      if (dim_sharding.getAxes().size() > 1) {
        return absl::InvalidArgumentError(
            absl::StrCat("sharding on dimension: ", idx,
                         " is not supported when R1 optimization is enabled."));
      }
    }
    mlir::sdy::TensorShardingAttr sharding_updated =
        mlir::sdy::TensorShardingAttr::get(
            sharding.getContext(), sharding.getMesh(main_func),
            sharding.getDimShardings()[0], sharding.getReplicatedAxes(),
            sharding.getUnreducedAxes());
    mlir::sdy::setSharding(main_func.getArgument(arg_idx), sharding_updated);
  }
  return absl::OkStatus();
}

absl::Status OptimizeTpuInputShape(
    mlir::ModuleOp module, const absl::flat_hash_set<int64_t>& variable_indices,
    bool enable_serialize_sharding,
    const xla::CompileOptionsProto* compile_options) {
  auto main_func = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) {
    return absl::InvalidArgumentError("Main function not found.");
  }
  TF_ASSIGN_OR_RETURN(
      bool should_run_tpu_input_r1_optimization,
      ShouldRunTpuInputR1Optimization(main_func, compile_options));
  if (!should_run_tpu_input_r1_optimization) {
    return absl::OkStatus();
  }
  mlir::OpBuilder builder(&main_func.getBody());
  llvm::SmallVector<mlir::Type, 4> new_arg_types;
  for (auto& arg : main_func.getArguments()) {
    int idx = arg.getArgNumber();
    if (variable_indices.contains(idx)) {
      new_arg_types.push_back(arg.getType());
      continue;
    }

    auto input_type = llvm::cast<mlir::TensorType>(arg.getType());
    if (input_type && (!input_type.hasRank() || !input_type.hasStaticShape())) {
      return absl::InvalidArgumentError(
          "Input type must have static shape and rank.");
    }
    if (!input_type || input_type.getRank() <= 1) {
      new_arg_types.push_back(input_type);
      continue;
    }
    new_arg_types.push_back(mlir::RankedTensorType::get(
        input_type.getNumElements(), input_type.getElementType()));
    arg.setType(new_arg_types.back());
    TF_RETURN_IF_ERROR(MaybeUpdateArgumentSharding(
        main_func, idx, builder, enable_serialize_sharding, compile_options));
    auto reshape_op = mlir::stablehlo::ReshapeOp::create(
        builder,
        mlir::NameLoc::get(
            mlir::StringAttr::get(builder.getContext(), "reshape_op"),
            arg.getLoc()),
        input_type, arg);
    arg.replaceAllUsesExcept(reshape_op.getResult(), reshape_op);
  }
  main_func.setType(
      mlir::FunctionType::get(main_func.getContext(), new_arg_types,
                              main_func.getFunctionType().getResults()));

  if (VLOG_IS_ON(1)) {
    ::neptune::DumpMlirOpToFile("after_reshape_optimization", module);
  }

  return absl::OkStatus();
}

struct TpuInputR1OptimizationPass
    : public mlir::PassWrapper<TpuInputR1OptimizationPass,
                               mlir::OperationPass<mlir::ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TpuInputR1OptimizationPass)

  TpuInputR1OptimizationPass() = default;
  TpuInputR1OptimizationPass(const TpuInputR1OptimizationPass&) {}

  explicit TpuInputR1OptimizationPass(
      absl::Span<const int64_t> variable_indices,
      const xla::CompileOptionsProto* compile_options) {
    for (int64_t idx : variable_indices) {
      variable_arg_indices_list.push_back(idx);
    }
    this->compile_options = compile_options;
  }

  void runOnOperation() override {
    mlir::ModuleOp module = getOperation();
    variable_arg_indices_set = {variable_arg_indices_list.begin(),
                                variable_arg_indices_list.end()};
    absl::Status status = OptimizeTpuInputShape(
        module, variable_arg_indices_set, serialize_sharding.getValue(),
        this->compile_options);
    if (!status.ok()) {
      module.emitError(status.message());
      return signalPassFailure();
    }
  }

  llvm::StringRef getArgument() const final {
    return "tpu-input-r1-optimization-pass";
  }
  llvm::StringRef getDescription() const final {
    return "Reshape input tensors to R1 for TPU models.";
  }

  ListOption<int> variable_arg_indices_list{
      *this, "variable-arg-indices-list",
      llvm::cl::desc("the arguments of the main function that are variables.")};

  // Test only option. If false, the modified OpSharding proto will be converted
  // to Hlosharding human readable string.
  Option<bool> serialize_sharding{*this, "enable_serialize_sharding",
                                  llvm::cl::init(false),
                                  llvm::cl::desc("Test only flag to enable "
                                                 "serialize sharding.")};
  absl::flat_hash_set<int64_t> variable_arg_indices_set;
  const xla::CompileOptionsProto* compile_options = nullptr;
};

}  // namespace

std::unique_ptr<mlir::Pass> CreateTpuInputR1OptimizationPass(
    absl::Span<const int64_t> variable_indices,
    const xla::CompileOptionsProto* absl_nullable compile_options) {
  return std::make_unique<TpuInputR1OptimizationPass>(variable_indices,
                                                      compile_options);
}

static mlir::PassRegistration<TpuInputR1OptimizationPass> pass;

}  // namespace neptune

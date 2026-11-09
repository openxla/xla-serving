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
#include <optional>
#include <string>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "llvm/include/llvm/ADT/SmallVector.h"  // from @llvm-project
#include "llvm/include/llvm/Support/Casting.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinTypes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Location.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @xla_serving
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

constexpr absl::string_view kXlaServingStreamOpName = "xla_serving_stream";
constexpr absl::string_view kBackendConfigAttr = "mhlo.backend_config";
constexpr absl::string_view kOutputNamesAttr = "output_names";

bool ContainsStreamingOp(mlir::Operation* op);

bool ContainsStreamingOpImpl(mlir::Operation* op,
                             absl::flat_hash_set<mlir::Operation*>& visited) {
  if (visited.contains(op)) return false;
  visited.insert(op);

  // Direct check for stablehlo.custom_call target name
  if (auto custom_call = llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(op)) {
    if (custom_call.getCallTargetName().str() == kXlaServingStreamOpName) {
      return true;
    }
  }

  // Direct check for func.call
  if (auto call_op = llvm::dyn_cast<mlir::func::CallOp>(op)) {
    mlir::ModuleOp module = call_op->getParentOfType<mlir::ModuleOp>();
    auto called_func =
        module.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
    if (called_func && ContainsStreamingOpImpl(called_func, visited)) {
      return true;
    }
  }

  // General walk for nested operations (e.g. inside functions, loops, etc.)
  bool found = false;
  op->walk([&](mlir::Operation* nested_op) {
    if (found) return;

    if (auto custom_call =
            llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(nested_op)) {
      if (custom_call.getCallTargetName().str() == kXlaServingStreamOpName) {
        found = true;
        return;
      }
    } else if (auto call_op = llvm::dyn_cast<mlir::func::CallOp>(nested_op)) {
      mlir::ModuleOp module = call_op->getParentOfType<mlir::ModuleOp>();
      auto called_func =
          module.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
      if (called_func && ContainsStreamingOpImpl(called_func, visited)) {
        found = true;
        return;
      }
    }
  });

  return found;
}

absl::Status AddStepIdToCustomCall(mlir::OpBuilder& builder,
                                   mlir::stablehlo::CustomCallOp custom_call,
                                   mlir::Value step_id) {
  // 1. Append step_id to inputs operand group
  custom_call.getInputsMutable().append({step_id});

  // 2. Parse, modify and write back the 'output_names' in 'mhlo.backend_config'
  mlir::DictionaryAttr backend_config =
      custom_call->getAttrOfType<mlir::DictionaryAttr>(kBackendConfigAttr);
  llvm::SmallVector<mlir::NamedAttribute, 4> new_attrs;
  std::string output_names;
  if (backend_config) {
    for (mlir::NamedAttribute attr : backend_config) {
      if (attr.getName().str() == kOutputNamesAttr) {
        output_names =
            llvm::cast<mlir::StringAttr>(attr.getValue()).getValue().str();
      } else {
        new_attrs.push_back(attr);
      }
    }
  }

  if (output_names.empty()) {
    // If output_names is not configured, FFI falls back to generating
    // "output_i". Since we appended step_id, the number of user outputs was
    // custom_call.getInputs().size() - 1.
    std::vector<std::string> default_names;
    for (int i = 0; i < custom_call.getInputs().size() - 1; ++i) {
      default_names.push_back(absl::StrCat("output_", i));
    }
    output_names = absl::StrJoin(default_names, ",");
  }

  // Append the reserved request ID name
  absl::StrAppend(&output_names, ",", kStreamedResultBatchedStepIdName);

  mlir::Builder mlir_builder(custom_call->getContext());
  new_attrs.push_back({mlir_builder.getStringAttr(kOutputNamesAttr),
                       mlir_builder.getStringAttr(output_names)});
  custom_call->setAttr(kBackendConfigAttr,
                       mlir_builder.getDictionaryAttr(new_attrs));

  // 3. Update 'operand_layouts' attribute if present to match the new operands
  // count.
  mlir::ArrayAttr operand_layouts = custom_call.getOperandLayoutsAttr();
  if (operand_layouts) {
    llvm::SmallVector<mlir::Attribute, 4> new_layouts(operand_layouts.begin(),
                                                      operand_layouts.end());
    mlir::DenseIntElementsAttr step_id_layout;
    if (llvm::cast<mlir::RankedTensorType>(step_id.getType()).getRank() == 0) {
      step_id_layout = mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get({0}, mlir_builder.getIndexType()),
          llvm::ArrayRef<int64_t>{});
    } else {
      llvm::SmallVector<int64_t, 1> step_id_layout_values = {0};
      step_id_layout = mlir::DenseIntElementsAttr::get(
          mlir::RankedTensorType::get({1}, mlir_builder.getIndexType()),
          step_id_layout_values);
    }
    new_layouts.push_back(step_id_layout);
    custom_call.setOperandLayoutsAttr(mlir_builder.getArrayAttr(new_layouts));
  }

  return absl::OkStatus();
}

// Declarations of blocks/operations recursively traversal.
absl::Status PropagateBlock(
    mlir::OpBuilder& builder, mlir::Block* block, mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs);

absl::Status PropagateCall(
    mlir::OpBuilder& builder, mlir::func::CallOp call_op, mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs) {
  // 1. Find the called function
  mlir::ModuleOp module = call_op->getParentOfType<mlir::ModuleOp>();
  auto called_func =
      module.lookupSymbol<mlir::func::FuncOp>(call_op.getCallee());
  if (!called_func) return absl::OkStatus();

  // 2. If the function hasn't been mutated yet, mutate it!
  mlir::Value func_step_id;
  auto it = mutated_funcs.find(called_func.getOperation());
  if (it == mutated_funcs.end()) {
    mlir::Type step_id_type = step_id.getType();
    if (mlir::failed(called_func.insertArgument(
            called_func.getNumArguments(), step_id_type,
            /*argAttrs=*/{}, call_op.getLoc()))) {
      return absl::InternalError(absl::StrCat(
          "Failed to insert step ID argument into called function signature: ",
          called_func.getName().str()));
    }
    func_step_id = called_func.getArgument(called_func.getNumArguments() - 1);
    mutated_funcs[called_func.getOperation()] = func_step_id;

    // Propagate inside the called function's entry block!
    mlir::Block* entry_block = &called_func.getBody().front();
    mlir::OpBuilder func_builder(called_func.getContext());
    func_builder.setInsertionPointToStart(entry_block);
    TF_RETURN_IF_ERROR(PropagateBlock(func_builder, entry_block, func_step_id,
                                      value_in_block, mutated_funcs));
  } else {
    func_step_id = it->second;
  }

  // 3. Re-create or update the CallOp to pass the step_id as the last operand!
  llvm::SmallVector<mlir::Value, 4> operands(call_op.getOperands());
  operands.push_back(step_id);

  builder.setInsertionPoint(call_op);
  auto new_call_op = builder.create<mlir::func::CallOp>(
      call_op.getLoc(), call_op.getResultTypes(), call_op.getCallee(),
      operands);

  // Replace uses of the old call's results with the new call's results
  for (int i = 0; i < call_op.getNumResults(); ++i) {
    call_op.getResult(i).replaceAllUsesWith(new_call_op.getResult(i));
  }
  call_op.erase();

  return absl::OkStatus();
}

absl::StatusOr<mlir::stablehlo::WhileOp> PropagateThroughWhile(
    mlir::OpBuilder& builder, mlir::stablehlo::WhileOp while_op,
    mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs) {
  // 1. Get old operands and add step_id
  llvm::SmallVector<mlir::Value, 4> operands(while_op.getOperands());
  operands.push_back(step_id);

  // 2. Get old results and add step_id type
  llvm::SmallVector<mlir::Type, 4> result_types(while_op.getResultTypes());
  result_types.push_back(step_id.getType());

  builder.setInsertionPoint(while_op);
  mlir::Location loc = while_op.getLoc();

  // 3. Create the new while_op with updated signature
  auto new_while_op =
      builder.create<mlir::stablehlo::WhileOp>(loc, result_types, operands);

  // 4. Move cond region blocks
  mlir::Region& new_cond = new_while_op.getCond();
  new_cond.takeBody(while_op.getCond());
  mlir::Block& cond_block = new_cond.front();
  cond_block.addArgument(step_id.getType(), loc);

  // 5. Move body region blocks
  mlir::Region& new_body = new_while_op.getBody();
  new_body.takeBody(while_op.getBody());
  mlir::Block& body_block = new_body.front();
  mlir::Value body_step_id = body_block.addArgument(step_id.getType(), loc);

  // 6. Update body block's return op (terminator) to include step_id as a
  // passthrough
  auto return_op =
      llvm::cast<mlir::stablehlo::ReturnOp>(body_block.getTerminator());
  llvm::SmallVector<mlir::Value, 4> return_operands(return_op.getOperands());
  return_operands.push_back(body_step_id);

  mlir::OpBuilder return_builder(return_op);
  return_builder.create<mlir::stablehlo::ReturnOp>(return_op.getLoc(),
                                                   return_operands);
  return_op.erase();

  // 7. Replace results usage for original outputs
  for (int i = 0; i < while_op.getNumResults(); ++i) {
    while_op.getResult(i).replaceAllUsesWith(new_while_op.getResult(i));
  }

  while_op.erase();

  // 8. Recursively traverse inside the cond/body blocks using top-down step ID
  // mapping
  TF_RETURN_IF_ERROR(PropagateBlock(builder, &cond_block,
                                    cond_block.getArguments().back(),
                                    value_in_block, mutated_funcs));
  TF_RETURN_IF_ERROR(PropagateBlock(builder, &body_block, body_step_id,
                                    value_in_block, mutated_funcs));

  return new_while_op;
}

absl::Status PropagateThroughIf(
    mlir::OpBuilder& builder, mlir::stablehlo::IfOp if_op, mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs) {
  // stablehlo.if regions do not have arguments. Nested ops can directly access
  // parent block's variables via SSA dominance. We just need to recurse.
  mlir::Block& then_block = if_op.getTrueBranch().front();
  TF_RETURN_IF_ERROR(PropagateBlock(builder, &then_block, step_id,
                                    value_in_block, mutated_funcs));

  mlir::Block& else_block = if_op.getFalseBranch().front();
  TF_RETURN_IF_ERROR(PropagateBlock(builder, &else_block, step_id,
                                    value_in_block, mutated_funcs));

  return absl::OkStatus();
}

absl::Status PropagateThroughCase(
    mlir::OpBuilder& builder, mlir::stablehlo::CaseOp case_op,
    mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs) {
  for (mlir::Region& region : case_op.getBranches()) {
    mlir::Block& block = region.front();
    TF_RETURN_IF_ERROR(PropagateBlock(builder, &block, step_id, value_in_block,
                                      mutated_funcs));
  }

  return absl::OkStatus();
}

absl::Status PropagateBlock(
    mlir::OpBuilder& builder, mlir::Block* block, mlir::Value step_id,
    absl::flat_hash_map<mlir::Block*, mlir::Value>& value_in_block,
    absl::flat_hash_map<mlir::Operation*, mlir::Value>& mutated_funcs) {
  value_in_block[block] = step_id;

  mlir::Operation* current_op = &block->front();
  while (current_op != nullptr) {
    mlir::Operation* next_op = current_op->getNextNode();

    if (auto custom_call =
            llvm::dyn_cast<mlir::stablehlo::CustomCallOp>(current_op)) {
      if (custom_call.getCallTargetName().str() == kXlaServingStreamOpName) {
        TF_RETURN_IF_ERROR(
            AddStepIdToCustomCall(builder, custom_call, value_in_block[block]));
      }
    } else if (auto while_op =
                   llvm::dyn_cast<mlir::stablehlo::WhileOp>(current_op)) {
      if (ContainsStreamingOp(while_op)) {
        TF_ASSIGN_OR_RETURN(
            auto new_while_op,
            PropagateThroughWhile(builder, while_op, value_in_block[block],
                                  value_in_block, mutated_funcs));
        next_op = new_while_op->getNextNode();
      }
    } else if (auto if_op = llvm::dyn_cast<mlir::stablehlo::IfOp>(current_op)) {
      if (ContainsStreamingOp(if_op)) {
        TF_RETURN_IF_ERROR(PropagateThroughIf(builder, if_op,
                                              value_in_block[block],
                                              value_in_block, mutated_funcs));
      }
    } else if (auto case_op =
                   llvm::dyn_cast<mlir::stablehlo::CaseOp>(current_op)) {
      if (ContainsStreamingOp(case_op)) {
        TF_RETURN_IF_ERROR(PropagateThroughCase(builder, case_op,
                                                value_in_block[block],
                                                value_in_block, mutated_funcs));
      }
    } else if (auto call_op = llvm::dyn_cast<mlir::func::CallOp>(current_op)) {
      if (ContainsStreamingOp(call_op)) {
        TF_RETURN_IF_ERROR(PropagateCall(builder, call_op,
                                         value_in_block[block], value_in_block,
                                         mutated_funcs));
      }
    }

    current_op = next_op;
  }

  return absl::OkStatus();
}

bool ContainsStreamingOp(mlir::Operation* op) {
  absl::flat_hash_set<mlir::Operation*> visited;
  return ContainsStreamingOpImpl(op, visited);
}

}  // namespace

absl::Status PropagateStreamingStepId(mlir::ModuleOp module, bool is_batched,
                                      std::optional<int64_t> batch_size) {
  if (VLOG_IS_ON(1)) {
    xla_serving::DumpMlirOpToFile("module_before_step_id_propagation", module);
  }
  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) {
    // If no main function is found, this is an unsupported module structure.
    return absl::InternalError(
        "Could not find entry 'main' function in StableHLO module");
  }

  // Check if xla_serving_stream exists at all before mutating the entry
  // function.
  if (!ContainsStreamingOp(main_func)) {
    LOG(INFO)
        << "[PropagateStreamingStepId] Does not contain xla_serving_stream!";
    return absl::OkStatus();
  }

  mlir::OpBuilder builder(module.getContext());
  mlir::Type step_id_type;
  if (is_batched) {
    int64_t batch_dim = 1;
    if (batch_size.has_value()) {
      if (*batch_size <= 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Streaming batch_size must be positive, got ", *batch_size));
      }
      batch_dim = *batch_size;
    } else {
      mlir::RankedTensorType ranked_type;
      for (int i = main_func.getNumArguments() - 1; i >= 0; --i) {
        bool is_const_arg = false;
        if (auto arg_attr = main_func.getArgAttrOfType<mlir::DictionaryAttr>(
                i, "mhlo.frontend_attributes")) {
          if (auto is_const = arg_attr.get("is_constant")) {
            if (auto str = llvm::dyn_cast<mlir::StringAttr>(is_const)) {
              if (str.getValue() == "true") is_const_arg = true;
            } else if (auto b = llvm::dyn_cast<mlir::BoolAttr>(is_const)) {
              if (b.getValue()) is_const_arg = true;
            }
          }
        }
        if (is_const_arg) continue;
        if (auto type = llvm::dyn_cast<mlir::RankedTensorType>(
                main_func.getArgument(i).getType())) {
          if (type.getRank() >= 1) {
            ranked_type = type;
            break;
          }
        }
      }
      if (ranked_type) {
        batch_dim = ranked_type.getShape()[0];
      }
    }
    step_id_type =
        mlir::RankedTensorType::get({batch_dim}, builder.getI32Type());
  } else {
    step_id_type = mlir::RankedTensorType::get({}, builder.getI32Type());
  }

  // 1. Insert step_id argument into the main entry function signature.
  if (mlir::failed(
          main_func.insertArgument(main_func.getNumArguments(), step_id_type,
                                   /*argAttrs=*/{}, main_func.getLoc()))) {
    return absl::InternalError(
        "Failed to insert dynamic request ID argument into main entry function "
        "signature");
  }
  mlir::Value main_step_id =
      main_func.getArgument(main_func.getNumArguments() - 1);

  // 2. Traversal block top-down to map dominance and rewrite ops
  absl::flat_hash_map<mlir::Block*, mlir::Value> value_in_block;
  absl::flat_hash_map<mlir::Operation*, mlir::Value> mutated_funcs;
  mlir::Block* entry_block = &main_func.getBody().front();
  builder.setInsertionPointToStart(entry_block);

  TF_RETURN_IF_ERROR(PropagateBlock(builder, entry_block, main_step_id,
                                    value_in_block, mutated_funcs));

  if (VLOG_IS_ON(1)) {
    xla_serving::DumpMlirOpToFile("module_after_step_id_propagation", module);
  }

  return absl::OkStatus();
}

absl::StatusOr<xla_serving::HostBuffer> CreateRequestIdsBuffer(int64_t batch_size,
                                                       int32_t request_id) {
  xla::ifrt::Shape shape({batch_size});
  xla::ifrt::DType dtype{xla::ifrt::DType::Kind::kS32};

  xla_serving::HostBuffer host_buffer(dtype, shape);
  int32_t* data = reinterpret_cast<int32_t*>(host_buffer.base());

  for (int64_t i = 0; i < batch_size; ++i) {
    data[i] = request_id;
  }

  return host_buffer;
}

absl::StatusOr<xla_serving::HostBuffer> CreateScalarRequestIdBuffer(
    int32_t request_id) {
  xla::ifrt::Shape shape({});
  xla::ifrt::DType dtype{xla::ifrt::DType::Kind::kS32};

  xla_serving::HostBuffer host_buffer(dtype, shape);
  *reinterpret_cast<int32_t*>(host_buffer.base()) = request_id;

  return host_buffer;
}

absl::StatusOr<xla_serving::HostBuffer> CreateRequestIdsBuffer(
    absl::Span<const int32_t> request_ids) {
  int64_t batch_size = request_ids.size();
  xla::ifrt::Shape shape({batch_size});
  xla::ifrt::DType dtype{xla::ifrt::DType::Kind::kS32};

  xla_serving::HostBuffer host_buffer(dtype, shape);
  int32_t* data = reinterpret_cast<int32_t*>(host_buffer.base());

  for (int64_t i = 0; i < batch_size; ++i) {
    data[i] = request_ids[i];
  }

  return host_buffer;
}

}  // namespace xla_serving

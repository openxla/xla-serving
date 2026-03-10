#include "runtime/core/stablehlo_utils.h"  // from @neptune

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @com_google_absl
#include "absl/container/inlined_vector.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/strings/str_join.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "llvm/include/llvm/Support/LogicalResult.h"  // from @llvm-project
#include "mlir/include/mlir/Dialect/Func/IR/FuncOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Attributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Block.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Builders.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Types.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Value.h"  // from @llvm-project
#include "mlir/include/mlir/Support/DebugStringHelper.h"  // from @llvm-project
#include "mlir/include/mlir/Support/LLVM.h"  // from @llvm-project
#include "runtime/libs/mlir/utils.h"  // from @neptune
#include "third_party/openxla/shardy/src/shardy/dialect/sdy/ir/dialect.h"
#include "third_party/openxla/shardy/src/shardy/dialect/sdy/ir/utils.h"
#include "stablehlo/dialect/StablehloOps.h"  // from @stablehlo
#include "stablehlo/transforms/StablehloRefineShapes.h"  // from @stablehlo
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/hlo/parser/hlo_parser.h"  // from @xla
#include "xla/mlir/utils/error_util.h"  // from @xla
#include "xla/mlir/utils/type_util.h"  // from @xla
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/refine_polymorphic_shapes.h"  // from @xla
#include "xla/service/spmd/shardy/stablehlo_round_trip/export_shardings.h"  // from @xla
#include "xla/service/spmd/shardy/utils.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace jsv {
namespace {

constexpr absl::string_view kPolymorphicShapeAttrName =
    "jax.uses_shape_polymorphism";

// Gets sharding from an MLIR string attribute.
absl::StatusOr<xla::OpSharding> GetSharding(
    mlir::StringAttr sharding_attr_str) {
  if (!sharding_attr_str) {
    return xla::HloSharding::Replicate().ToProto();
  } else {
    absl::StatusOr<xla::HloSharding> hlo_sharding =
        xla::ParseSharding(sharding_attr_str.getValue());
    if (hlo_sharding.ok()) {
      VLOG(2) << "Input Arg " << index
              << " hlo sharding: " << hlo_sharding->ToString();
      return hlo_sharding->ToProto();
    }
    // Try parse serialized sharding.
    xla::OpSharding op_sharding;
    if (op_sharding.ParseFromString(sharding_attr_str.getValue().str())) {
      VLOG(2) << "Input Arg " << index << " xla op sharding: " << op_sharding;
      return op_sharding;
    }
    return absl::InvalidArgumentError(absl::StrCat(
        "Failed to parse sharding: ", sharding_attr_str.getValue().str()));
  }
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

// Builds ArgSpec from an MLIR function argument.
absl::StatusOr<ArgSpec> BuildArgSpecFromArg(
    mlir::func::FuncOp func, int idx,
    xla::CompileOptionsProto* compile_options) {
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::DType dtype,
      jsv::sharding_internal::ToIfrtDType(func.getArgumentTypes()[idx]));
  TF_ASSIGN_OR_RETURN(auto shape, jsv::sharding_internal::ToMlirShape(
                                      func.getArgumentTypes()[idx]));

  xla::OpSharding sharding;
  if (compile_options &&
      compile_options->executable_build_options().use_shardy_partitioner()) {
    mlir::sdy::TensorShardingAttr sharding_attr =
        mlir::sdy::getSharding(func.getArgument(idx));
    // Enforce allow_spmd_sharding_propagation_to_parameters flag.
    xla::sdy::adjustInputSharding(
        func, idx, sharding_attr, shape.size(),
        compile_options->executable_build_options()
            .allow_spmd_sharding_propagation_to_parameters());
    sharding = GetShardingFromShardyAttr(sharding_attr, func);
  } else {
    auto sharding_attr_str =
        func.getArgAttrOfType<mlir::StringAttr>(idx, "mhlo.sharding");
    TF_ASSIGN_OR_RETURN(sharding, GetSharding(sharding_attr_str));
  }

  TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                      xla::HloSharding::FromProto(sharding));

  return ArgSpec{.dtype = dtype,
                 .shape = std::move(shape),
                 .sharding = std::move(hlo_sharding)};
}

// Builds ArgSpec from an MLIR function result.
absl::StatusOr<ArgSpec> BuildArgSpecFromResult(
    mlir::func::FuncOp func, int idx,
    xla::CompileOptionsProto* compile_options) {
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::DType dtype,
      jsv::sharding_internal::ToIfrtDType(func.getResultTypes()[idx]));
  TF_ASSIGN_OR_RETURN(auto shape, jsv::sharding_internal::ToMlirShape(
                                      func.getResultTypes()[idx]));

  xla::OpSharding sharding;
  if (compile_options &&
      compile_options->executable_build_options().use_shardy_partitioner()) {
    mlir::sdy::TensorShardingAttr sharding_attr =
        mlir::sdy::getFuncResultSharding(func, idx);
    // Enforce allow_spmd_sharding_propagation_to_output flag.
    xla::sdy::adjustOutputSharding(
        func, idx, sharding_attr, shape.size(),
        compile_options->executable_build_options()
            .allow_spmd_sharding_propagation_to_output());
    sharding = GetShardingFromShardyAttr(sharding_attr, func);
  } else {
    auto sharding_attr_str =
        func.getResultAttrOfType<mlir::StringAttr>(idx, "mhlo.sharding");
    TF_ASSIGN_OR_RETURN(sharding, GetSharding(sharding_attr_str));
  }

  TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                      xla::HloSharding::FromProto(sharding));

  return ArgSpec{.dtype = dtype,
                 .shape = std::move(shape),
                 .sharding = std::move(hlo_sharding)};
}

// Creates a list of MLIR types based on the input concrete and types. It is
// potentially used to replace the dynamic shape of input args. This function
// assumes that the platform index has already been removed by a prior call to
// SetPlatformIndex.
absl::StatusOr<std::vector<mlir::Type>> GetMlirType(
    mlir::Block::BlockArgListType args, mlir::Builder builder,
    absl::Span<const xla::ifrt::DType> input_dtypes,
    absl::Span<const xla::ifrt::Shape> input_shapes) {
  int num_inputs = input_shapes.size();
  if (args.size() != num_inputs) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of main function arguments does not match number of inputs: ",
        args.size(), " vs ", num_inputs));
  }

  std::vector<mlir::Type> input_mlir_types(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    TF_RETURN_IF_ERROR(ValidateIOType(input_dtypes[i]));
    TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                        xla::ifrt::ToPrimitiveType(input_dtypes[i]));
    std::vector<int64_t> xla_dimensions(input_shapes[i].dims().begin(),
                                        input_shapes[i].dims().end());
    TF_ASSIGN_OR_RETURN(
        mlir::Type mlir_type,
        xla::ConvertPrimitiveTypeToMlirType(primitive_type, builder));
    mlir::RankedTensorType ranked_tensor_type =
        mlir::RankedTensorType::get(xla_dimensions, mlir_type);

    input_mlir_types[i] = ranked_tensor_type;
    VLOG(2) << absl::StrCat(
        "The module main function arg#", i,
        " type is: ", mlir::debugString(args[i].getType()),
        ", the input mlir type is: ", mlir::debugString(input_mlir_types[i]));
  }

  return input_mlir_types;
}

}  // namespace

// Checks if the jax.uses_shape_polymorphism attribute is set to true in an MLIR
// module.
absl::StatusOr<bool> UsesPolymorphicShape(mlir::ModuleOp module) {
  mlir::Attribute uses_polymorphic_shape_attr =
      module->getAttr(kPolymorphicShapeAttrName);
  if (!uses_polymorphic_shape_attr) {
    VLOG(2) << absl::StrCat("Module does not have attribute: ",
                            kPolymorphicShapeAttrName);
    return false;
  }
  mlir::BoolAttr uses_polymorphic_shape_bool_attr =
      module->getAttrOfType<mlir::BoolAttr>(kPolymorphicShapeAttrName);
  if (!uses_polymorphic_shape_bool_attr) {
    return absl::InvalidArgumentError(
        absl::StrCat(kPolymorphicShapeAttrName, " value is not a boolean: ",
                     mlir::debugString(uses_polymorphic_shape_attr)));
  }
  return uses_polymorphic_shape_bool_attr.getValue();
}

absl::StatusOr<StableHloIOSpec> GetStableHloIOSpec(
    mlir::ModuleOp module,
    xla::CompileOptionsProto* absl_nullable compile_options) {
  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) {
    return absl::InvalidArgumentError("No main function found in the module");
  }

  StableHloIOSpec spec;

  for (int i = 0; i < main_func.getNumArguments(); ++i) {
    TF_ASSIGN_OR_RETURN(ArgSpec arg_spec,
                        BuildArgSpecFromArg(main_func, i, compile_options));
    spec.inputs.push_back(std::move(arg_spec));
  }

  for (int i = 0; i < main_func.getNumResults(); ++i) {
    TF_ASSIGN_OR_RETURN(ArgSpec arg_spec,
                        BuildArgSpecFromResult(main_func, i, compile_options));
    spec.outputs.push_back(std::move(arg_spec));
  }
  return spec;
}

absl::Status SetPlatformIndex(mlir::ModuleOp module,
                              absl::Span<const std::string> supported_platforms,
                              absl::string_view platform_name) {
  int platform_index = -1;
  if (!supported_platforms.empty()) {
    auto found_platform = std::find(supported_platforms.begin(),
                                    supported_platforms.end(), platform_name);
    if (found_platform == supported_platforms.end()) {
      return absl::InvalidArgumentError(
          absl::StrCat("The current platform ", platform_name,
                       " is not among the platforms required by the module: [",
                       absl::StrJoin(supported_platforms, ", "), "]"));

    } else {
      // We only use a platform index argument if we support at least 2
      // platforms.
      if (supported_platforms.size() > 1) {
        platform_index = found_platform - supported_platforms.begin();
      }
    }
  }

  if (platform_index < 0) return absl::OkStatus();
  VLOG(3) << "XlaCallModule setting the platform_index to " << platform_index
          << " for platform " << platform_name << ".";

  auto main = module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main) {
    return absl::InvalidArgumentError("Cannot find 'main' in module");
  }

  mlir::Block& main_body = main.front();

  if (main.getNumArguments() < 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "The module should have a platform index argument but it has no ",
        "arguments"));
  }
  mlir::OpBuilder op_builder(main);
  op_builder.setInsertionPointToStart(&main_body);
  mlir::BlockArgument platform_index_arg = main_body.getArgument(0);
  mlir::RankedTensorType arg_ranked_type =
      mlir::dyn_cast<mlir::RankedTensorType>(platform_index_arg.getType());
  if (!arg_ranked_type || arg_ranked_type.getRank() != 0 ||
      !(arg_ranked_type.getElementType().isSignlessInteger(32) ||
        arg_ranked_type.getElementType().isSignlessInteger(64))) {
    return absl::InvalidArgumentError(
        absl::StrCat("Module argument at index 0 should be a 0-dimensional "
                     "32-bit or 64-bit integer-tensor platform index argument "
                     "but has type ",
                     mlir::debugString(platform_index_arg.getType())));
  }
  bool is_32_bit = arg_ranked_type.getElementType().isSignlessInteger(32);
  auto const_attr = is_32_bit ? op_builder.getI32IntegerAttr(platform_index)
                              : op_builder.getI64IntegerAttr(platform_index);
  auto platform_index_op = mlir::stablehlo::ConstantOp::create(
      op_builder, platform_index_arg.getLoc(), const_attr);
  platform_index_arg.replaceAllUsesWith(platform_index_op);

  CHECK(succeeded(main.eraseArgument(0)));
  return absl::OkStatus();
}

absl::Status RefineShapesIfNeeded(
    mlir::ModuleOp module,
    absl::Span<const std::optional<xla::ifrt::ArrayRef>> weights,
    absl::Span<const xla::ifrt::Shape> input_shapes,
    absl::Span<const xla::ifrt::DType> input_dtypes,
    absl::Span<const int> module_kept_var_idx) {
  const int num_weights = weights.size();

  TF_ASSIGN_OR_RETURN(bool use_shape_polymorphism,
                      UsesPolymorphicShape(module));
  if (!use_shape_polymorphism) {
    LOG(INFO) << absl::StrCat("Skipping shape refinement because attribute ",
                              kPolymorphicShapeAttrName,
                              " doesn't exist or set to false.");
    return absl::OkStatus();
  }

  if (input_dtypes.size() != input_shapes.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Number of input dtypes does not match number of input shapes: ",
        input_dtypes.size(), " vs ", input_shapes.size()));
  }

  if (input_shapes.size() + num_weights < module_kept_var_idx.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("The total number of weights and inputs should be equal "
                     "or more than the number of module kept var indices: ",
                     input_shapes.size(), " vs ", module_kept_var_idx.size()));
  }

  std::vector<xla::ifrt::Shape> args_shapes;
  std::vector<xla::ifrt::DType> args_dtypes;
  args_shapes.reserve(module_kept_var_idx.size());
  args_dtypes.reserve(module_kept_var_idx.size());

  for (int idx : module_kept_var_idx) {
    if (idx < num_weights) {
      const auto& weight_spec = weights[idx];
      if (!weight_spec.has_value()) {
        // When this error occurs, it likely indicates that the weight was
        // incorrectly treated as unused weights, thus never being loaded,
        // resulting in a nullopt weight_spec.
        return absl::InvalidArgumentError(
            absl::StrCat("Weight #", idx, " is required, but not loaded."));
      }
      args_shapes.push_back(weight_spec.value()->shape());
      args_dtypes.push_back(weight_spec.value()->dtype());
    } else {
      int64_t index = idx - num_weights;
      args_shapes.push_back(input_shapes[index]);
      args_dtypes.push_back(input_dtypes[index]);
    }
  }
  mlir::func::FuncOp main_func =
      module.lookupSymbol<mlir::func::FuncOp>("main");
  if (!main_func) {
    return absl::InvalidArgumentError(
        "No main function found in the module. Shape refinement is only "
        "supported for main functions.");
  }

  mlir::Block& main_block = main_func.front();
  mlir::Builder builder(module.getContext());
  mlir::Block::BlockArgListType main_args = main_block.getArguments();

  TF_ASSIGN_OR_RETURN(
      std::vector<mlir::Type> input_mlir_types,
      GetMlirType(main_args, builder, args_dtypes, args_shapes));

  // Insert @stablehlo.shape_refinement_operand_wrapper custom_call ops as shims
  // to maintain the validity of the module when main's input types are changed
  // later. This is a workaround to allow shape refinement to be applied; the
  // custom_calls are removed before returning. More details at
  // https://source.corp.google.com/piper///depot/google3/third_party/tensorflow/compiler/tf2xla/kernels/xla_call_module_loader.cc;l=285-325;rcl=720675664
  //
  // Example:
  //
  // The original main function with dynamic shapes:
  //
  // func.func public @main(%arg0: tensor<?x?xf32>) -> (tensor<?x?xf32>
  // {jax.result_info = ""}) {
  //     return %0 : tensor<?x?xf32>
  // }
  //
  // The main function after mlir::stablehlo::refineArguments call with input
  // type "tensor<2x2xf32>":
  //
  // func.func public @main(%arg0: tensor<2x2xf32>) ->
  // (tensor<?x?xf32> {jax.result_info = ""}) {
  //     %c = stablehlo.constant dense<2> : tensor<2xi64>
  //     %0 = stablehlo.custom_call
  //     @stablehlo.shape_refinement_operand_wrapper(%arg0, %c)
  //     {indices_of_shape_operands = dense<1> : tensor<1xi64>} :
  //     (tensor<2x2xf32>, tensor<2xi64>) -> tensor<?x?xf32>
  //     return %0 :tensor<?x?xf32>
  //   }
  {
    mlir::BaseScopedDiagnosticHandler diag_handler(module.getContext());
    if (failed(mlir::stablehlo::refineArguments(main_func, input_mlir_types))) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Failed to wrap static input types with "
          "@stablehlo.shape_refinement_operand_wrapper custom call: ",
          diag_handler.ConsumeStatus().message()));
    }
  }

  // Runs shape refinement pass and removes the custom calls. The main function
  // after shape refinement:
  //
  // func.func public @main(%arg0: tensor<2x2xf32>) -> (tensor<2x2xf32>
  // {jax.result_info = ""}) {
  //     return %arg0 : tensor<2x2xf32>
  //   }
  // Enable the shape assertion check. Revisit this if StableHloRunner adds
  // support for skipping shape assertion checks.
  TF_RETURN_IF_ERROR(
      xla::RefinePolymorphicShapes(module, /*enable_shape_assertions=*/true));

  return absl::OkStatus();
}

absl::Status ValidateIOType(xla::ifrt::DType dtype) {
  switch (dtype.kind()) {
    case xla::ifrt::DType::kInvalid:
    case xla::ifrt::DType::kOpaque:
    case xla::ifrt::DType::kToken:
    case xla::ifrt::DType::kString:
      return absl::InvalidArgumentError(
          absl::StrCat("Type: ", dtype, " is not supported."));
    default:
      return absl::OkStatus();
  }
}

absl::Status ValidateIOSpecs(absl::Span<const ArgSpec> stablehlo_arg_specs,
                             absl::Span<const ArgSpec> model_arg_specs,
                             absl::Span<const int> module_kept_var_idx) {
  for (int i = 0; i < stablehlo_arg_specs.size(); ++i) {
    TF_RETURN_IF_ERROR(ValidateIOType(stablehlo_arg_specs[i].dtype));
  }

  size_t arg_count_from_model_io_spec = module_kept_var_idx.empty()
                                            ? model_arg_specs.size()
                                            : module_kept_var_idx.size();
  if (stablehlo_arg_specs.size() != arg_count_from_model_io_spec) {
    return absl::FailedPreconditionError(absl::StrCat(
        "StableHlo expected ", stablehlo_arg_specs.size(), " args, but got ",
        arg_count_from_model_io_spec, " from model IO spec."));
  }
  for (int i = 0; i < stablehlo_arg_specs.size(); ++i) {
    int idx_in_model_arg_specs =
        module_kept_var_idx.empty() ? i : module_kept_var_idx[i];
    if (idx_in_model_arg_specs >= model_arg_specs.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Module kept var idx ", idx_in_model_arg_specs,
                       " is out of range [0, ", model_arg_specs.size(), ")."));
    }
    if (stablehlo_arg_specs[i] != model_arg_specs[idx_in_model_arg_specs]) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected arg#", i, " to have ArgSpec: {",
          stablehlo_arg_specs[i].DebugString(), "}, but got {",
          model_arg_specs[idx_in_model_arg_specs].DebugString(),
          "} from model IO spec arg#", idx_in_model_arg_specs, "."));
    }
  }
  return absl::OkStatus();
}

absl::Status InputShapeEqual(ArgShape stablehlo_input_shape,
                             xla::ifrt::Shape input_shape) {
  if (stablehlo_input_shape.size() != input_shape.dims().size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "StableHlo expected input shape rank: ", stablehlo_input_shape.size(),
        ", but got: ", input_shape.dims().size(), " from the actual input."));
  }
  for (int i = 0; i < stablehlo_input_shape.size(); ++i) {
    if (!stablehlo_input_shape[i].has_value()) continue;
    if (*stablehlo_input_shape[i] != input_shape.dims()[i]) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected input shape dimension #", i, ": ",
          *stablehlo_input_shape[i], ", but got: ", input_shape.dims()[i],
          " from the actual input."));
    }
  }
  return absl::OkStatus();
}

xla::ifrt::Shape ToIfrtShape(const ArgShape& shape) {
  absl::InlinedVector<int64_t, 6> dimensions;
  dimensions.reserve(shape.size());
  for (std::optional<int64_t> dim : shape) {
    // If a dimension is not set, it must be an input and won't actually affect
    // checkpoint loading. We use an invalid value here to make sure that the
    // shape is not used for anything else.
    //
    // If this ever causes problems, we should probably make it so the Orbax
    // checkpoint loader doesn't require argument types to be passed in.
    dimensions.push_back(dim.value_or(-1));
  }
  return xla::ifrt::Shape(dimensions);
}

}  // namespace jsv

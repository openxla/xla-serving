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

#include "runtime/core/loaded_stablehlo.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinAttributes.h"  // from @llvm-project
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/Operation.h"  // from @llvm-project
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @xla_serving
#include "runtime/libs/mlir/dump_mlir_util.h"  // from @xla_serving
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

// TODO: Add unit tests for this file.
namespace xla_serving {

namespace {

// Returns the `ArgSpec`s for model weights. The `ArgSpec` will be std::nullopt
// if the weight is unused by any of the StableHLOs.
// Args:
//  - stablehlo: the StableHLO that holds one or more StableHLOs (i.e.
//    concretized StableHLOs from a polymorphic StableHLO).
//  - num_weights: the total number of weights irrespective of whether they are
//    used or not (i.e., passed to StableHLO during the execution)
// Returns:
//  - A vector of `ArgSpec`s for model weights. The `ArgSpec` will be
//    std::nullopt if the weight is unused by any of the StableHLOs.
absl::StatusOr<std::vector<std::optional<ArgSpec>>> GetWeightSpecs(
    absl_nonnull std::shared_ptr<StableHlo> stablehlo, int num_weights) {
  // The following checks are defensive and should never fail, since some of
  // them are already checked in `StableHloRunner::Create`
  if (!stablehlo) {
    return absl::InvalidArgumentError(
        "`StableHlo` should be non-null when getting weight specs");
  }
  if (stablehlo->specs().empty()) {
    return absl::InvalidArgumentError(
        "`StableHlo` should have at least one spec when getting weight specs");
  }
  if (num_weights < 0) {
    return absl::InvalidArgumentError(
        "`num_weights` should be non-negative when getting weight specs");
  }

  // Get specs for weights that are ever used by any of the StableHLOs.
  // When a weight spec is used, verify that its spec is consistent across all
  // StableHLOs that require this weight.
  // When a weight spec is not used, it means the weight is unused by any of
  // the StableHLOs, thus setting spec to std::nullopt.
  std::vector<std::optional<ArgSpec>> weight_specs(num_weights, std::nullopt);
  for (int spec_idx = 0; spec_idx < stablehlo->specs().size(); ++spec_idx) {
    const StableHlo::Spec& cur_spec = stablehlo->specs()[spec_idx];
    for (int i : cur_spec.mlir_module_kept_var_idx) {
      if (i >= num_weights) {
        break;
      }
      const auto& cur_weight_spec = cur_spec.inputs[i];
      if (!weight_specs[i].has_value()) {
        weight_specs[i] = cur_weight_spec;
      } else if (*weight_specs[i] != cur_weight_spec) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Weights should have the same dtype, shape and sharding among"
            " StableHLOs, but found mismatch for weight #",
            i, " : ", cur_weight_spec.DebugString(), " vs ",
            weight_specs[i].value().DebugString()));
      }
    }
  }

  // Verify that all weights have concrete shapes. There shouldn't be a case
  // where a weight is polymorphic.
  for (const auto& spec : weight_specs) {
    if (!spec.has_value()) {
      continue;
    }
    for (const auto& dim : spec.value().shape) {
      if (!dim.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Weight should have concrete shape, but found "
                         "weight with polymorphic shape: ",
                         spec.value().DebugString()));
      }
    }
  }
  return weight_specs;
}

// Returns the offset of user inputs in the main function arguments of the
// MLIR module. The offset is the number of USED weights in the main function
// arguments, since this function returns the number of used weights.
int GetInputOffset(absl::Span<const int> mlir_module_kept_var_idx,
                   int num_weights) {
  int num_used_weights = 0;
  for (int idx : mlir_module_kept_var_idx) {
    if (idx < num_weights) {
      num_used_weights++;
    }
  }
  return num_used_weights;
}

// Returns true if any of the inputs has polymorphic shape.
bool HasPolymorphicShape(absl::Span<const ArgSpec> inputs) {
  for (const auto& input : inputs) {
    for (const auto& dim : input.shape) {
      if (!dim.has_value()) {
        return true;
      }
    }
  }
  return false;
}

bool HasStreaming(mlir::ModuleOp module) {
  bool found = false;
  module.walk([&](mlir::Operation* op) {
    if (op->getName().getStringRef() == "stablehlo.custom_call") {
      if (auto target_attr =
              op->getAttrOfType<mlir::StringAttr>("call_target_name")) {
        if (target_attr.getValue() == "xla_serving_stream") {
          found = true;
        }
      }
    }
  });
  return found;
}

// Verifies that all StableHLO modules (shape buckets) have a consistent
// number of user inputs and outputs. This ensures that dynamic shape
// bucketing is structurally sound before execution.
absl::Status VerifyMultiBucketConsistency(
    const std::shared_ptr<StableHlo>& stablehlo, int num_weights,
    int* num_user_inputs_out, int* num_outputs_out) {
  if (stablehlo->specs().empty()) {
    return absl::InvalidArgumentError("StableHLO has no specs.");
  }

  if (stablehlo->specs()[0].inputs.size() < num_weights) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "StableHLO module has fewer inputs (%d) than the number of "
        "weights (%d)",
        stablehlo->specs()[0].inputs.size(), num_weights));
  }
  int num_user_inputs = stablehlo->specs()[0].inputs.size() - num_weights;
  int num_outputs = stablehlo->specs()[0].outputs.size();

  for (int i = 1; i < stablehlo->specs().size(); ++i) {
    if (stablehlo->specs()[i].inputs.size() < num_weights) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "StableHLO module has fewer inputs (%d) than the number of "
          "weights (%d)",
          stablehlo->specs()[i].inputs.size(), num_weights));
    }
    int cur_user_inputs = stablehlo->specs()[i].inputs.size() - num_weights;
    int cur_outputs = stablehlo->specs()[i].outputs.size();

    if (cur_user_inputs != num_user_inputs) {
      return absl::InvalidArgumentError(
          "Mismatched number of user inputs across buckets");
    }
    if (cur_outputs != num_outputs) {
      return absl::InvalidArgumentError(
          "Mismatched number of outputs across buckets");
    }
    for (int j = 0; j < num_outputs; ++j) {
      if (stablehlo->specs()[i].outputs[j].placement !=
          stablehlo->specs()[0].outputs[j].placement) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Mismatched output placement across buckets for output #", j,
            ": bucket 0 vs bucket ", i));
      }
    }
  }

  if (num_user_inputs_out) *num_user_inputs_out = num_user_inputs;
  if (num_outputs_out) *num_outputs_out = num_outputs;
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<LoadedStableHlo>> LoadedStableHlo::CreateForAot(
    absl_nonnull std::shared_ptr<StableHlo> stablehlo, int num_weights) {
  int num_user_inputs = 0;
  int num_outputs = 0;
  TF_RETURN_IF_ERROR(VerifyMultiBucketConsistency(
      stablehlo, num_weights, &num_user_inputs, &num_outputs));

  TF_ASSIGN_OR_RETURN(std::vector<std::optional<ArgSpec>> weight_specs,
                      GetWeightSpecs(stablehlo, num_weights));

  return absl::WrapUnique(new LoadedStableHlo(
      /*concrete_module_bundles=*/std::nullopt,
      /*symbolic_module_bundle=*/std::nullopt, std::move(weight_specs),
      /*output_specs=*/stablehlo->specs()[0].outputs, num_user_inputs,
      num_outputs,
      // TODO: Set has_streaming based on AOT executable metadata
      // when AOT + streaming execution is formally supported.
      /*has_streaming=*/false));
}

absl::StatusOr<std::unique_ptr<LoadedStableHlo>> LoadedStableHlo::Create(
    absl_nonnull std::shared_ptr<StableHlo> stablehlo, int num_weights,
    absl::string_view platform_name,
    std::optional<xla::CompileOptionsProto> compile_options) {
  if (!stablehlo) {
    return absl::InvalidArgumentError(
        "`StableHlo` should be non-null when creating `LoadedStableHlo`");
  }
  if (stablehlo->specs().empty()) {
    return absl::InvalidArgumentError(
        "`StableHlo` should have at least one spec when creating "
        "`LoadedStableHlo`");
  }
  if (num_weights < 0) {
    return absl::InvalidArgumentError(
        "`num_weights` should be non-negative when creating "
        "`LoadedStableHlo`");
  }

  TF_ASSIGN_OR_RETURN(std::vector<std::optional<ArgSpec>> weight_specs,
                      GetWeightSpecs(stablehlo, num_weights));

  std::optional<absl::flat_hash_map<
      std::vector<xla::ifrt::Shape>,
      std::unique_ptr<LoadedStableHlo::ModuleBundleWithLoader>>>
      concrete_module_bundles = std::nullopt;
  std::optional<std::unique_ptr<LoadedStableHlo::ModuleBundleWithLoader>>
      symbolic_module_bundle = std::nullopt;

  for (const auto& spec : stablehlo->specs()) {
    if (spec.mlir_module_string().empty()) {
      return absl::InvalidArgumentError(
          "StableHlo spec should have a non-empty MLIR module string");
    }

    // TODO: Check if loader can be shared across multiple
    // StableHLOs.
    //
    // Deserialize the string to the MLIR module.
    auto loader = std::make_unique<StableHloLoader>();
    TF_ASSIGN_OR_RETURN(auto module, loader->Load(spec.mlir_module_string()));
    if (!module) {
      return absl::FailedPreconditionError("Failed to load StableHlo module");
    }
    if (VLOG_IS_ON(1)) {
      DumpMlirOpToFile("stablehlo_module_origin", module.get());
    }

    // If the StableHLO module is exported for multiple platforms, set the
    // platform index to the given platform name.
    if (spec.target_platforms.size() > 1) {
      // The StableHLO module is exported for multiple platforms.
      TF_RETURN_IF_ERROR(
          SetPlatformIndex(module.get(), spec.target_platforms, platform_name));
    }

    // Validate JAX function USED input and output are consistent with that in
    // the MLIR module.
    TF_ASSIGN_OR_RETURN(
        StableHloIOSpec stablehlo_io_spec,
        GetStableHloIOSpec(module.get(), compile_options.has_value()
                                             ? &compile_options.value()
                                             : nullptr));
    TF_RETURN_IF_ERROR(ValidateIOSpecs(stablehlo_io_spec.inputs, spec.inputs,
                                       spec.mlir_module_kept_var_idx));
    TF_RETURN_IF_ERROR(ValidateIOSpecs(stablehlo_io_spec.outputs, spec.outputs,
                                       /*module_kept_var_indices=*/{}));

    // Create bundle for polymorphic or concrete StableHLO(s).
    //
    // Use MLIR module input argument shapes to determine whether the module
    // is polymorphic or not, rather than relying on the shape polymorphism
    // attribute (i.e. jax.uses_shape_polymorphism = True), since the latter
    // can be True when the JAX model is exported for multiple native
    // serialization platforms while all input shapes are concrete.
    if (HasPolymorphicShape(stablehlo_io_spec.inputs)) {
      TF_ASSIGN_OR_RETURN(bool use_shape_polymorphism,
                          UsesPolymorphicShape(module.get()));
      // jax.uses_shape_polymorphism is expected to be true if the model has
      // polymorphic input shapes, but may not be the case the other way around
      // as mentioned above.
      if (!use_shape_polymorphism) {
        return absl::InvalidArgumentError(
            "StableHLO module has polymorphic shape in the input spec, but "
            "jax.uses_shape_polymorphism is false");
      }
      if (symbolic_module_bundle.has_value()) {
        return absl::InvalidArgumentError(
            "Unexpected multiple polymorphic StableHLO modules");
      }
      symbolic_module_bundle =
          std::make_unique<LoadedStableHlo::ModuleBundleWithLoader>(
              LoadedStableHlo::ModuleBundleWithLoader{
                  .loader = std::move(loader),
                  .bundle =
                      LoadedStableHlo::ModuleBundle{
                          .module = std::move(module),
                          .module_kept_var_indices =
                              spec.mlir_module_kept_var_idx,
                          .input_offset = GetInputOffset(
                              spec.mlir_module_kept_var_idx, num_weights),
                      },
              });
    } else {
      std::vector<xla::ifrt::Shape> user_input_shapes;
      if (spec.inputs.size() < num_weights) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "StableHLO module has fewer inputs (%d) than the number of "
            "weights (%d)",
            spec.inputs.size(), num_weights));
      }
      user_input_shapes.reserve(spec.inputs.size() - num_weights);
      for (int i = num_weights; i < spec.inputs.size(); ++i) {
        user_input_shapes.push_back(ToIfrtShape(spec.inputs[i].shape));
      }
      if (!concrete_module_bundles.has_value()) {
        concrete_module_bundles = absl::flat_hash_map<
            std::vector<xla::ifrt::Shape>,
            std::unique_ptr<LoadedStableHlo::ModuleBundleWithLoader>>();
      }
      auto [_, inserted] = concrete_module_bundles->emplace(
          user_input_shapes,
          std::make_unique<LoadedStableHlo::ModuleBundleWithLoader>(
              LoadedStableHlo::ModuleBundleWithLoader{
                  .loader = std::move(loader),
                  .bundle =
                      LoadedStableHlo::ModuleBundle{
                          .module = std::move(module),
                          .module_kept_var_indices =
                              spec.mlir_module_kept_var_idx,
                          .input_offset = GetInputOffset(
                              spec.mlir_module_kept_var_idx, num_weights),
                      },
              }));
      if (!inserted) {
        return absl::InvalidArgumentError(
            "Unexpected multiple StableHLO modules with the same user input "
            "shapes");
      }
    }
  }

  // Verify that all StableHLO modules have the same number of user inputs and
  // outputs.
  int num_user_inputs = 0;
  int num_outputs = 0;
  TF_RETURN_IF_ERROR(VerifyMultiBucketConsistency(
      stablehlo, num_weights, &num_user_inputs, &num_outputs));

  bool has_streaming = false;
  if (symbolic_module_bundle.has_value()) {
    has_streaming =
        HasStreaming((*symbolic_module_bundle)->bundle.module.get());
  } else if (concrete_module_bundles.has_value() &&
             !concrete_module_bundles->empty()) {
    // All concrete modules should have the same stablehlo ops. They have
    // different input and output shapes only. So we can check any one of them
    // for streaming.
    has_streaming = HasStreaming(
        concrete_module_bundles->begin()->second->bundle.module.get());
  }

  return absl::WrapUnique(new LoadedStableHlo(
      std::move(concrete_module_bundles), std::move(symbolic_module_bundle),
      std::move(weight_specs), /*output_specs=*/stablehlo->specs()[0].outputs,
      num_user_inputs, num_outputs, has_streaming));
};

absl::StatusOr<std::unique_ptr<LoadedStableHlo::ModuleBundle>>
LoadedStableHlo::CloneModuleBundle(
    absl::Span<const xla::ifrt::Shape> input_shapes) const {
  absl::MutexLock lock(mutex_);
  if (concrete_module_bundles_.has_value()) {
    // For streaming models, an extra request ID input is appended to the end of
    // `input_shapes`. We truncate it here because `concrete_module_bundles_`
    // is keyed only by user input shapes.
    absl::Span<const xla::ifrt::Shape> user_input_shapes = input_shapes;
    if (input_shapes.size() > num_user_inputs_) {
      user_input_shapes = input_shapes.subspan(0, num_user_inputs_);
    }
    auto it = concrete_module_bundles_->find(std::vector<xla::ifrt::Shape>(
        user_input_shapes.begin(), user_input_shapes.end()));
    if (it == concrete_module_bundles_->end()) {
      std::string allowed_input_shapes_str;
      for (const auto& [key, value] : *concrete_module_bundles_) {
        allowed_input_shapes_str.append(
            absl::StrCat("{", absl::StrJoin(key, ","), "} "));
      }
      return absl::FailedPreconditionError(
          absl::StrCat("No MLIR module found for input shapes ",
                       absl::StrJoin(input_shapes, ","),
                       ". Allowed input shapes: ", allowed_input_shapes_str));
    }

    // TODO b/401364010: Consider adding a check to make sure the input DTypes
    // are compatible with the MLIR module. May not be feasible as this code
    // runs in the critical path of serving.
    return std::make_unique<LoadedStableHlo::ModuleBundle>(
        it->second->bundle.Clone());
  }

  if (symbolic_module_bundle_.has_value()) {
    LoadedStableHlo::ModuleBundle& bundle =
        symbolic_module_bundle_.value()->bundle;
    return std::make_unique<LoadedStableHlo::ModuleBundle>(bundle.Clone());
  }

  return absl::NotFoundError(
      "Neither concrete nor symbolic module bundle is "
      "available. This should never happen.");
}

void LoadedStableHlo::ReleaseAllModules() {
  absl::MutexLock lock(mutex_);
  if (concrete_module_bundles_.has_value()) {
    for (auto& bundle : concrete_module_bundles_.value()) {
      bundle.second->bundle.module = nullptr;
    }
  }
  if (symbolic_module_bundle_.has_value()) {
    symbolic_module_bundle_.value()->bundle.module = nullptr;
  }
}

}  // namespace xla_serving

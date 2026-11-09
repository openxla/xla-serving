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

#include "runtime/core/stablehlo.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

namespace {

// Computes the static element count of `shape`, or `std::nullopt` if any
// dimension is dynamic (`std::nullopt` or negative) or overflows `int64_t`.
std::optional<int64_t> ComputeStaticNumElements(const ArgShape& shape) {
  int64_t num_elements = 1;
  for (const std::optional<int64_t>& dim : shape) {
    if (!dim.has_value() || *dim < 0) {
      return std::nullopt;
    }
    if (*dim > 0 &&
        num_elements > std::numeric_limits<int64_t>::max() / *dim) {
      return std::nullopt;
    }
    num_elements *= *dim;
  }
  return num_elements;
}

absl::Status ValidateHostTranspose(absl::Span<const int64_t> permutation,
                                   size_t expected_rank, size_t input_index) {
  if (permutation.size() != expected_rank) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Input #", input_index, " host_transpose [",
        absl::StrJoin(permutation, ", "), "] rank ", permutation.size(),
        " does not match original input rank ", expected_rank, "."));
  }
  std::vector<bool> seen_axes(expected_rank, false);
  for (int64_t axis : permutation) {
    if (axis < 0 || static_cast<size_t>(axis) >= expected_rank ||
        seen_axes[axis]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Input #", input_index, " has invalid host_transpose permutation [",
          absl::StrJoin(permutation, ", "), "] for rank ", expected_rank, "."));
    }
    seen_axes[axis] = true;
  }
  return absl::OkStatus();
}

absl::StatusOr<StableHlo::Spec> BuildSpec(
    const xla_saved_model_manifest::Function& func) {
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> original_input_specs,
                      GetInputSpec(func));
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> original_output_specs,
                      GetOutputSpec(func));

  if (!func.body().has_stable_hlo_body()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Function should have a stable HLO body, but got ",
                     func.DebugString()));
  }

  const xla_saved_model_manifest::StableHloFunctionBody& stable_hlo_body =
      func.body().stable_hlo_body();
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> optimized_input_specs,
                      GetOptimizedInputSpec(func, original_input_specs));
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> optimized_output_specs,
                      GetOptimizedOutputSpec(func, original_output_specs));

  if (stable_hlo_body.num_input_aliased_output() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("num_input_aliased_output must be non-negative: ",
                     stable_hlo_body.num_input_aliased_output()));
  }
  constexpr uint32_t kMaxInt =
      static_cast<uint32_t>(std::numeric_limits<int>::max());
  std::vector<int> kept_var_indices;
  kept_var_indices.reserve(stable_hlo_body.module_kept_var_idx().size());
  for (uint32_t kept_index : stable_hlo_body.module_kept_var_idx()) {
    if (kept_index > kMaxInt) {
      return absl::InvalidArgumentError(
          absl::StrCat("module_kept_var_idx exceeds INT_MAX: ", kept_index));
    }
    kept_var_indices.push_back(static_cast<int>(kept_index));
  }

  return StableHlo::Spec{
      .inputs = std::move(optimized_input_specs),
      .outputs = std::move(optimized_output_specs),
      .original_inputs = std::move(original_input_specs),
      .original_outputs = std::move(original_output_specs),
      .optimized_at_export = stable_hlo_body.has_optimized_input_signature() ||
                             stable_hlo_body.has_optimized_output_signature(),
      .mlir_module =
          absl::string_view(stable_hlo_body.stable_hlo().inlined_bytes()),
      .mlir_module_kept_var_idx = std::move(kept_var_indices),
      .target_platforms = {stable_hlo_body.lowering_platforms().begin(),
                           stable_hlo_body.lowering_platforms().end()},
      .num_input_aliased_output =
          static_cast<int>(stable_hlo_body.num_input_aliased_output()),
  };
}

}  // namespace

absl::StatusOr<std::unique_ptr<StableHlo>> StableHlo::Create(
    absl::Span<const xla_saved_model_manifest::Function* const> funcs) {
  if (funcs.empty()) {
    return absl::InvalidArgumentError(
        "`funcs` should have at least one element");
  }
  std::vector<Spec> specs;
  specs.reserve(funcs.size());
  for (const auto& func : funcs) {
    TF_ASSIGN_OR_RETURN(Spec spec, BuildSpec(*func));
    specs.emplace_back(std::move(spec));
  }
  return Create(std::move(specs));
}

absl::StatusOr<std::unique_ptr<StableHlo>> StableHlo::Create(
    std::vector<Spec> specs) {
  if (specs.empty()) {
    return absl::InvalidArgumentError(
        "`specs` should have at least one element");
  }
  for (Spec& spec : specs) {
    if (spec.num_input_aliased_output < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("num_input_aliased_output must be non-negative: ",
                       spec.num_input_aliased_output));
    }
    if (spec.inputs.size() <
        static_cast<size_t>(spec.num_input_aliased_output)) {
      return absl::InvalidArgumentError(absl::StrCat(
          "StableHlo spec has fewer inputs (", spec.inputs.size(),
          ") than num_input_aliased_output (", spec.num_input_aliased_output,
          ")."));
    }
    if (spec.original_inputs.empty()) {
      spec.original_inputs = spec.inputs;
    } else if (spec.original_inputs.size() != spec.inputs.size()) {
      return absl::InvalidArgumentError(
          "StableHlo spec original_inputs size must match inputs size.");
    }
    if (spec.original_outputs.empty()) {
      spec.original_outputs = spec.outputs;
    } else if (spec.original_outputs.size() != spec.outputs.size()) {
      return absl::InvalidArgumentError(
          "StableHlo spec original_outputs size must match outputs size.");
    }
    for (size_t i = 0; i < spec.inputs.size(); ++i) {
      if (spec.original_inputs[i].dtype != spec.inputs[i].dtype) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input #", i, " dtype mismatch between original_inputs (",
            spec.original_inputs[i].dtype, ") and inputs (",
            spec.inputs[i].dtype, ")."));
      }
      if (spec.original_inputs[i].host_transpose.empty() &&
          !spec.inputs[i].host_transpose.empty()) {
        spec.original_inputs[i].host_transpose = spec.inputs[i].host_transpose;
      } else if (spec.inputs[i].host_transpose.empty() &&
                 !spec.original_inputs[i].host_transpose.empty()) {
        spec.inputs[i].host_transpose = spec.original_inputs[i].host_transpose;
      } else if (spec.original_inputs[i].host_transpose !=
                 spec.inputs[i].host_transpose) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input #", i,
            " host_transpose mismatch between original_inputs and inputs."));
      }
      if (!spec.original_inputs[i].host_transpose.empty()) {
        TF_RETURN_IF_ERROR(ValidateHostTranspose(
            spec.original_inputs[i].host_transpose,
            spec.original_inputs[i].shape.size(), i));
      }
      const std::optional<int64_t> original_elements =
          ComputeStaticNumElements(spec.original_inputs[i].shape);
      const std::optional<int64_t> optimized_elements =
          ComputeStaticNumElements(spec.inputs[i].shape);
      const bool requires_static_input_shape =
          spec.optimized_at_export ||
          spec.original_inputs[i].shape != spec.inputs[i].shape ||
          !spec.original_inputs[i].host_transpose.empty();
      if (requires_static_input_shape &&
          (!original_elements.has_value() || !optimized_elements.has_value())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input #", i,
            " must have static non-negative dimensions when export-time "
            "optimization or host_transpose is used."));
      }
      if (original_elements.has_value() && optimized_elements.has_value() &&
          *original_elements != *optimized_elements) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input #", i,
            " static element count mismatch between original_inputs (",
            *original_elements, ") and inputs (", *optimized_elements, ")."));
      }
    }
    const size_t first_aliased_input =
        spec.inputs.size() - static_cast<size_t>(spec.num_input_aliased_output);
    for (size_t i = first_aliased_input; i < spec.inputs.size(); ++i) {
      if (spec.original_inputs[i] != spec.inputs[i] ||
          !spec.original_inputs[i].host_transpose.empty()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Input-aliased output buffer argument #", i,
            " cannot use export-time optimization or host_transpose."));
      }
    }
    for (size_t i = 0; i < spec.outputs.size(); ++i) {
      if (!spec.original_outputs[i].host_transpose.empty() ||
          !spec.outputs[i].host_transpose.empty()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " cannot specify host_transpose; host_transpose is only supported "
            "on inputs."));
      }
      if (spec.original_outputs[i].dtype != spec.outputs[i].dtype) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i, " dtype mismatch between original_outputs (",
            spec.original_outputs[i].dtype, ") and outputs (",
            spec.outputs[i].dtype, ")."));
      }
      if (!spec.outputs[i].placement.has_value() &&
          spec.original_outputs[i].placement.has_value()) {
        spec.outputs[i].placement = spec.original_outputs[i].placement;
      } else if (!spec.original_outputs[i].placement.has_value() &&
                 spec.outputs[i].placement.has_value()) {
        spec.original_outputs[i].placement = spec.outputs[i].placement;
      } else if (spec.outputs[i].placement !=
                 spec.original_outputs[i].placement) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " placement mismatch between original_outputs and outputs."));
      }
      if (spec.num_input_aliased_output > 0 &&
          spec.original_outputs[i].shape != spec.outputs[i].shape) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " shape optimization is incompatible with input-aliased outputs."));
      }
      const std::optional<int64_t> original_elements =
          ComputeStaticNumElements(spec.original_outputs[i].shape);
      const std::optional<int64_t> optimized_elements =
          ComputeStaticNumElements(spec.outputs[i].shape);
      const bool requires_static_output_shape =
          spec.optimized_at_export ||
          spec.original_outputs[i].shape != spec.outputs[i].shape;
      if (requires_static_output_shape &&
          (!original_elements.has_value() || !optimized_elements.has_value())) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " must have static non-negative dimensions when export-time "
            "optimization is used."));
      }
      if (original_elements.has_value() && optimized_elements.has_value() &&
          *original_elements != *optimized_elements) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " static element count mismatch between original_outputs (",
            *original_elements, ") and outputs (", *optimized_elements, ")."));
      }
    }
    int previous_kept_index = -1;
    for (int kept_index : spec.mlir_module_kept_var_idx) {
      if (kept_index < 0 ||
          static_cast<size_t>(kept_index) >= spec.inputs.size()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "mlir_module_kept_var_idx entry ", kept_index,
            " is out of range [0, ", spec.inputs.size(), ")."));
      }
      if (kept_index <= previous_kept_index) {
        return absl::InvalidArgumentError(absl::StrCat(
            "mlir_module_kept_var_idx must be strictly increasing, but saw ",
            kept_index, " after ", previous_kept_index, "."));
      }
      previous_kept_index = kept_index;
    }
  }
  return absl::WrapUnique(new StableHlo(std::move(specs)));
}

}  // namespace xla_serving

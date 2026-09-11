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

#include "runtime/core/stablehlo.h"  // from @neptune

#include <memory>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "google/protobuf/message.h"
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

namespace {

absl::StatusOr<StableHlo::Spec> BuildSpec(
    const neptune_model_manifest::Function& func) {
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> input_spec, GetInputSpec(func));
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> output_spec, GetOutputSpec(func));

  if (!func.body().has_stable_hlo_body()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Function should have a stable HLO body, but got ",
                     func.DebugString()));
  }

  const neptune_model_manifest::StableHloFunctionBody& stable_hlo_body =
      func.body().stable_hlo_body();

  return StableHlo::Spec{
      .inputs = std::move(input_spec),
      .outputs = std::move(output_spec),
      .mlir_module =
          absl::string_view(stable_hlo_body.stable_hlo().inlined_bytes()),
      .mlir_module_kept_var_idx =
          {stable_hlo_body.module_kept_var_idx().begin(),
           stable_hlo_body.module_kept_var_idx().end()},
      .target_platforms = {stable_hlo_body.lowering_platforms().begin(),
                           stable_hlo_body.lowering_platforms().end()},
  };
}

}  // namespace

absl::StatusOr<std::unique_ptr<StableHlo>> StableHlo::Create(
    absl::Span<const neptune_model_manifest::Function* const> funcs) {
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
  return absl::WrapUnique(new StableHlo(std::move(specs)));
}

absl::StatusOr<std::unique_ptr<StableHlo>> StableHlo::Create(
    std::vector<Spec> specs) {
  if (specs.empty()) {
    return absl::InvalidArgumentError(
        "`specs` should have at least one element");
  }
  return absl::WrapUnique(new StableHlo(std::move(specs)));
}

}  // namespace xla_serving

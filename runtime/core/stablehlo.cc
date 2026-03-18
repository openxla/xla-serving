#include "runtime/core/stablehlo.h"  // from @neptune

#include <memory>
#include <utility>
#include <vector>

#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "google/protobuf/message.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.pb.h"
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace jsv {

namespace {

absl::StatusOr<StableHlo::Spec> BuildSpec(
    const orbax_model_manifest::Function& func) {
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> input_spec, GetInputSpec(func));
  TF_ASSIGN_OR_RETURN(std::vector<ArgSpec> output_spec, GetOutputSpec(func));

  if (!func.body().has_stable_hlo_body()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Function should have a stable HLO body, but got ",
                     func.DebugString()));
  }

  const orbax_model_manifest::StableHloFunctionBody& stable_hlo_body =
      func.body().stable_hlo_body();

  return StableHlo::Spec{
      .inputs = std::move(input_spec),
      .outputs = std::move(output_spec),
      .mlir_module_string = stable_hlo_body.stable_hlo().inlined_bytes(),
      .mlir_module_kept_var_idx =
          {stable_hlo_body.module_kept_var_idx().begin(),
           stable_hlo_body.module_kept_var_idx().end()},
      .target_platforms = {stable_hlo_body.lowering_platforms().begin(),
                           stable_hlo_body.lowering_platforms().end()},
  };
}

}  // namespace

absl::StatusOr<std::unique_ptr<StableHlo>> StableHlo::Create(
    absl::Span<const orbax_model_manifest::Function* const> funcs) {
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

}  // namespace jsv

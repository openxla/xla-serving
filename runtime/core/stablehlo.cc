#include "third_party/neptune/runtime/core/stablehlo.h"

#include <memory>
#include <utility>
#include <vector>

#include "learning/infra/mira/experimental/orbax_model/interpreter/manifest_utils.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/memory/memory.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/stablehlo_utils.h"
#include "third_party/protobuf/message.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

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

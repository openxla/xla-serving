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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_

#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

// TODO b/401364010 - Add unit test for this class.

// A class that holds a StableHLO or a list of concretized StableHLOs for
// a polymorphic StableHLO.
//
// When there is one StableHLO spec, it should be
// - either a StableHLO with concrete inputs and outputs
// - or a polymorphic StableHLO using symbols for dynamic dimensions, which
//   will be replaced by concrete values at runtime as part of shape refinement
//   process.
//
// When there are multiple StableHLO specs, it should be
// - a polymorphic StableHLO that gets concretized at export time to a set of
//   concrete StableHLOs based on user discrete values for dynamic dimensions.
class StableHlo {
 public:
  struct Spec {
    // Specs of all inputs including weights and user inputs irrespective of
    // whether they are actually passed to the main function in the MLIR module.
    std::vector<ArgSpec> inputs;

    // Specs of outputs.
    std::vector<ArgSpec> outputs;

    // The MLIR module (in StableHLO dialect). Can be owned string or view.
    std::variant<std::string, absl::string_view> mlir_module;

    // Returns a view of the MLIR module string.
    absl::string_view mlir_module_string() const {
      if (std::holds_alternative<std::string>(mlir_module)) {
        return std::get<std::string>(mlir_module);
      }
      return std::get<absl::string_view>(mlir_module);
    }

    // Sorted indices of inputs that are actually passed to the main function in
    // MLIR module. The other inputs are unused and don't need to be passed.
    std::vector<int> mlir_module_kept_var_idx;

    // A list of target platforms for which the MLIR module is lowered to
    // (e.g cpu, tpu, cuda, rocm)
    std::vector<std::string> target_platforms;
  };

  // Creates an StableHlo object from one or multiple `Function`s.
  static absl::StatusOr<std::unique_ptr<StableHlo>> Create(
      absl::Span<const neptune_model_manifest::Function* const> funcs);

  // Creates an StableHlo object directly from a list of `Spec`s.
  static absl::StatusOr<std::unique_ptr<StableHlo>> Create(
      std::vector<Spec> specs);

  absl::Span<const Spec> specs() const { return specs_; }

 private:
  explicit StableHlo(std::vector<Spec> specs) : specs_(std::move(specs)) {}

  // Specs of one or multiple StableHLOs that represent a single logic
  // JAX function which can be either a concrete or a polymorphic function.
  const std::vector<Spec> specs_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_

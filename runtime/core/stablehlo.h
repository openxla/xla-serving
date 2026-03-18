#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "third_party/py/orbax/experimental/model/core/protos/manifest.pb.h"
#include "xla/xla_data.pb.h"  // from @xla

namespace jsv {

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

    // The MLIR module string (in StableHLO dialect).
    absl::string_view mlir_module_string;

    // Sorted indices of inputs that are actually passed to the main function in
    // MLIR module. The other inputs are unused and don't need to be passed.
    std::vector<int> mlir_module_kept_var_idx;

    // A list of target platforms for which the MLIR module is lowered to
    // (e.g cpu, tpu, cuda, rocm)
    std::vector<std::string> target_platforms;
  };

  // Creates an StableHlo object from one or multiple `Function`s.
  static absl::StatusOr<std::unique_ptr<StableHlo>> Create(
      absl::Span<const orbax_model_manifest::Function* const> funcs);

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_STABLEHLO_H_

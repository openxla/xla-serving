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

#ifndef XLA_SERVING_RUNTIME_LOADED_STABLEHLO_H_
#define XLA_SERVING_RUNTIME_LOADED_STABLEHLO_H_

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "mlir/include/mlir/IR/BuiltinOps.h"  // from @llvm-project
#include "mlir/include/mlir/IR/OwningOpRef.h"  // from @llvm-project
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/libs/ifrt/stablehlo_loader.h"  // from @neptune
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

// A class that holds loaded StableHLO(s) (i.e. MLIR module(s)) representing
// a single JAX model. The loaded StableHLO(s) could be:
// - a single concrete StableHLO
// - or multiple concrete StableHLO(s) concretized from a polymorphic
// - or a single polymorphic StableHLO.
//
// For loaded concrete StableHLO(s), they can be looked up by user input shapes
// (i.e. excluding weights). For loaded polymorphic StableHLO, there is only
// one module bundle that gets returned irrespective of the user input shapes.
class LoadedStableHlo {
 public:
  struct ModuleBundle {
    mlir::OwningOpRef<mlir::ModuleOp> module;
    // Sorted indices of weights and user inputs that are actually passed to the
    // the main function in MLIR module. The other inputs are unused and don't
    // need to be passed.
    std::vector<int> module_kept_var_indices;
    // The offset of user input index in list of arguments to the main function
    // of the MLIR module. Note input_offset may not be the same as the number
    // of weights, since some weights may be unused and don't need to be passed
    // to the main function.
    int input_offset;

    // Creates a copy of the ModuleBundle.
    ModuleBundle Clone() {
      return ModuleBundle{
          .module = module->clone(),
          .module_kept_var_indices = module_kept_var_indices,
          .input_offset = input_offset,
      };
    }
  };

  // Creates a LoadedStableHlo object from a StableHlo object.
  //
  // Args:
  // - `num_weights` is the total number of weights irrespective of whether
  //    they are passed to the main function in MLIR module or not.
  // - `platform_name` is the name of the platform that will run the StableHlo.
  // - `compile_options` is the compilation options used to compile the
  //    StableHlo
  static absl::StatusOr<std::unique_ptr<LoadedStableHlo>> Create(
      absl_nonnull std::shared_ptr<StableHlo> stablehlo, int num_weights,
      absl::string_view platform_name,
      std::optional<xla::CompileOptionsProto> compile_options);

  // Creates a LoadedStableHlo object specifically configured for AOT execution.
  //
  // Args:
  //   stablehlo: The StableHlo object containing the specifications.
  //   num_weights: The number of weights irrespective of whether they are
  //                passed to the main function in MLIR module or not.
  //
  // Returns:
  //   A unique pointer to the created LoadedStableHlo object, or an error
  //   status.
  static absl::StatusOr<std::unique_ptr<LoadedStableHlo>> CreateForAot(
      std::shared_ptr<StableHlo> stablehlo, int num_weights);

  // Returns a clone of ModuleBundle for the given input shapes.
  // When the loaded StableHLO(s) is concrete, the ModuleBundle will be looked
  // up by the input shapes (user inputs without weights).
  // When the loaded StableHLO(s) is polymorphic, the same ModuleBundle will be
  // returned irrespective of the input shapes.
  // Args:
  // - `user_input_shapes` is shapes of user inputs (i.e. without weights).
  absl::StatusOr<std::unique_ptr<LoadedStableHlo::ModuleBundle>>
  CloneModuleBundle(absl::Span<const xla::ifrt::Shape> user_input_shapes) const;

  // Releases all `mlir::ModuleOp`s in the module bundles.
  // This is useful to free up memory when all MLIR modules have been compiled
  // to executables and we no longer need the MLIR modules.
  void ReleaseAllModules();

  // Returns the `ArgSpec`s for model weights.
  // When the weight is not used by any of the StableHLO(s), the corresponding
  // `ArgSpec` will be `std::nullopt`. For weights that are used by some or
  // all of the StableHLO(s), the `ArgSpec` should be the same across all
  // them, since all these StableHLO(s) represent the same JAX model.
  const std::vector<std::optional<ArgSpec>>& weight_specs() const {
    return weight_specs_;
  }

  // Returns the `ArgSpec`s for model outputs.
  const std::vector<ArgSpec>& output_specs() const { return output_specs_; }

  // Returns the number of weights irrespective of whether they are passed to
  // the main function in MLIR module or not.
  int num_weights() const { return weight_specs_.size(); }

  // Return the number of user inputs irrespective of whether they are passed
  // to the main function in MLIR module or not.
  int num_user_inputs() const { return num_user_inputs_; }

  // Returns the number of outputs.
  int num_outputs() const { return num_outputs_; }

  bool has_streaming() const { return has_streaming_; }

 private:
  // A bundle of MLIR module and its loader so they have the same lifetime and
  // make sure the MLIR context in the loader is alive when the module is
  // destroyed.
  struct ModuleBundleWithLoader {
    std::unique_ptr<StableHloLoader> loader;
    ModuleBundle bundle;
  };

  explicit LoadedStableHlo(
      std::optional<
          absl::flat_hash_map<std::vector<xla::ifrt::Shape>,
                              std::unique_ptr<ModuleBundleWithLoader>>>
          concrete_module_bundles,
      std::optional<std::unique_ptr<ModuleBundleWithLoader>>
          symbolic_module_bundle,
      std::vector<std::optional<ArgSpec>> weight_specs,
      std::vector<ArgSpec> output_specs, int num_user_inputs, int num_outputs,
      bool has_streaming)
      : concrete_module_bundles_(std::move(concrete_module_bundles)),
        symbolic_module_bundle_(std::move(symbolic_module_bundle)),
        weight_specs_(std::move(weight_specs)),
        output_specs_(std::move(output_specs)),
        num_user_inputs_(num_user_inputs),
        num_outputs_(num_outputs),
        has_streaming_(has_streaming) {}

  mutable absl::Mutex mutex_;

  // MLIR module bundles for StableHLOs with concrete input shapes. The key is
  // the input shapes (user inputs without weights). Typically there is only
  // one entry in the map, but there can be multiple entries if the StableHLO
  // function is concretized from a polymorphic one.
  std::optional<absl::flat_hash_map<std::vector<xla::ifrt::Shape>,
                                    std::unique_ptr<ModuleBundleWithLoader>>>
      concrete_module_bundles_ ABSL_GUARDED_BY(mutex_);

  // MLIR module bundle for StableHLOs with symbolic input shapes.
  std::optional<std::unique_ptr<ModuleBundleWithLoader>> symbolic_module_bundle_
      ABSL_GUARDED_BY(mutex_);

  // `ArgSpec`s for model weights. When a weight is not used by any of the
  // StableHLO(s), the corresponding `ArgSpec` will be `std::nullopt`.
  const std::vector<std::optional<ArgSpec>> weight_specs_;

  // `ArgSpec`s for model outputs.
  const std::vector<ArgSpec> output_specs_;

  // Number of user-provided inputs to the model (excluding weights and any
  // system-added inputs like streaming request IDs). This should be the same
  // for all StableHLO modules.
  const int num_user_inputs_;

  // Number of outputs of the model. Should be the same for all StableHLO
  // modules when there are multiple concretized StableHLOs representing one
  // polymorphic StableHLO.
  const int num_outputs_;

  const bool has_streaming_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_LOADED_STABLEHLO_H_

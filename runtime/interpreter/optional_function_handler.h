#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/interpreter/interpreter_create_options.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune

namespace neptune {

// An interface to handle optional (non-StableHLO) functions.
class OptionalFunctionHandler {
 public:
  virtual ~OptionalFunctionHandler() = default;

  // Handles the function by creating a sub-graph that can be inserted into the
  // current graph. `pre_data_conversion` and `post_data_conversion` is used to
  // determine whether to convert the input and output of the function to
  // HostBuffer.
  virtual absl::StatusOr<std::vector<std::unique_ptr<Kernel>>> HandleFunction(
      absl::string_view model_path,
      const neptune_model_manifest::Function& function,
      const InterpreterCreateOptions& create_options, bool pre_data_conversion,
      bool post_data_conversion,
      const neptune_model_orchestration::NamedTensorTypeList* named_inputs,
      const neptune_model_orchestration::NamedTensorTypeList*
          named_outputs) = 0;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_

#include <memory>
#include <vector>

#include "third_party/neptune/runtime/interpreter/interpreter_create_options.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/py/orbax/experimental/model/core/protos/manifest.proto.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"

namespace jsv {

// An interface to handle optional (non-StableHLO) functions.
class OptionalFunctionHandler {
 public:
  virtual ~OptionalFunctionHandler() = default;

  // Handles the function by creating a sub-graph that can be inserted into the
  // current graph.
  // TODO: `insert_pre_data_conversion` and
  // `insert_post_data_conversion` are used to determine whether to insert data
  // conversion kernels before and after the optional function. Remove these two
  // params once the TfSavedModelFunctionRunnerOp supports HostBuffer as input
  // and output.
  virtual absl::StatusOr<std::vector<std::unique_ptr<Kernel>>> HandleFunction(
      absl::string_view model_path,
      const orbax_model_manifest::Function& function,
      const InterpreterCreateOptions& create_options,
      bool insert_pre_data_conversion, bool insert_post_data_conversion,
      const orbax_export_orchestration::NamedTensorTypeList* named_inputs,
      const orbax_export_orchestration::NamedTensorTypeList* named_outputs) = 0;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_OPTIONAL_FUNCTION_HANDLER_H_

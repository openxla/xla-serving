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

#include "runtime/tensorflow/tf_compatible_optional_function_handler.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "protos/orchestration.pb.h"  // from @xla_serving
#include "protos/tf_concrete_function_handle.pb.h"  // from @xla_serving
#include "protos/type.pb.h"  // from @xla_serving
#include "runtime/core/constants.h"  // from @xla_serving
#include "runtime/core/flat_signature.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/tensorflow/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/tensorflow/tf_compatible_interpreter_create_options.h"  // from @xla_serving
#include "runtime/tensorflow/tf_saved_model_runner.h"  // from @xla_serving
#include "runtime/tensorflow/utils/common_utils.h"  // from @xla_serving
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorflow/core/framework/function.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tensorflow/core/protobuf/meta_graph.pb.h"  // from @org_tensorflow
#include "tensorflow/core/protobuf/saved_model.pb.h"  // from @org_tensorflow

namespace xla_serving {
namespace {

// A handle to a TF concrete function that contains the information needed to
// execute the function.
struct TfConcreteFunctionHandle {
 public:
  static TfConcreteFunctionHandle FromProto(
      const xla_saved_model_protos::TfConcreteFunctionHandle& proto) {
    return TfConcreteFunctionHandle{
        .fn_name = std::string(proto.fn_name()),
        .input_names = {proto.input_names().begin(), proto.input_names().end()},
        .output_names = {proto.output_names().begin(),
                         proto.output_names().end()},
    };
  }

  std::string fn_name;
  std::vector<std::string> input_names;
  std::vector<std::string> output_names;
};

// Gets a TF concrete function handle from an
// `xla_saved_model_manifest::Function`.
absl::StatusOr<TfConcreteFunctionHandle> GetTfConcreteFunctionHandle(
    const xla_saved_model_manifest::Function& func,
    absl::string_view model_path) {
  const xla_saved_model_manifest::UnstructuredData& tf_body = func.body().other();

  TF_ASSIGN_OR_RETURN(
      std::string full_path,
      GetAbsoluteFileSystemPath(model_path, tf_body.file_system_location()));

  xla_saved_model_protos::TfConcreteFunctionHandle handle_proto;
  VLOG(4) << "Reading TF concrete function handle from: " << full_path;
  TF_RETURN_IF_ERROR(
      tsl::ReadBinaryProto(tsl::Env::Default(), full_path, &handle_proto));
  return TfConcreteFunctionHandle::FromProto(handle_proto);
}

// Gets the TF SavedModel path from the manifest.

template <typename Dict>
std::vector<std::string> SortedKeys(const Dict& dict) {
  std::vector<std::string> keys;
  keys.reserve(dict.size());
  for (const auto& [key, unused] : dict) {
    keys.push_back(key);
  }
  absl::c_sort(keys);
  return keys;
}

}  // namespace

absl::StatusOr<std::shared_ptr<TfSavedModelRunner>>
TfCompatibleOptionalFunctionHandler::CreateTfSavedModelRunner(
    absl::string_view model_path, TfrtOptions tfrt_options) {
  TF_ASSIGN_OR_RETURN(xla_saved_model_manifest::Manifest manifest,
                      GetManifest(model_path));
  TF_ASSIGN_OR_RETURN(std::string saved_model_dir,
                      GetTfSavedModelDirPath(model_path));
  return TfSavedModelRunner::Create(
      saved_model_dir, std::move(tfrt_options.saved_model_options));
}

absl::StatusOr<std::unique_ptr<Kernel>>
TfCompatibleOptionalFunctionHandler::HandleFunction(
    absl::string_view model_path,
    const xla_saved_model_manifest::Function& function,
    const InterpreterCreateOptions& create_options, bool pre_data_conversion,
    bool post_data_conversion,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_inputs,
    const xla_saved_model_orchestration::NamedTensorTypeList* named_outputs) {
  if (!function.has_body()) {
    return absl::InvalidArgumentError("Function does not have body.");
  }

  if (!function.body().has_other()) {
    return absl::InvalidArgumentError("Function body does not have `Other`.");
  }
  const xla_saved_model_manifest::UnstructuredData& tf_body =
      function.body().other();

  std::unique_ptr<Kernel> tf_kernel;

  TF_RETURN_IF_ERROR(CheckMimeAndDataType(
      tf_body,
      {kTfConcreteFunctionHandleMimeTypeLegacy,
       kTfConcreteFunctionHandleMimeType},
      {xla_saved_model_manifest::UnstructuredData::kFileSystemLocation},
      "Function body for TF Function"));

  TF_ASSIGN_OR_RETURN(auto tf_concrete_function_handle,
                      GetTfConcreteFunctionHandle(function, model_path));
  std::string signature_name = tf_concrete_function_handle.fn_name;
  LOG(INFO) << "Signature name in TF SavedModel: " << signature_name;

  const std::vector<std::string>& input_names =
      tf_concrete_function_handle.input_names;

  const std::vector<std::string>& output_names =
      tf_concrete_function_handle.output_names;

  absl::call_once(tf_saved_model_runner_init_once_, [&]() {
    // Load the TF SavedModel and initialize the `TfSavedModelRunner` if not
    // yet. For post-processing function, it is possible that the TF SavedModel
    // has already been loaded.
    if (tf_saved_model_runner_.ok()) {
      tf_saved_model_runner_ = absl::InternalError(
          "TF SavedModel runner is already initialized. This is unexpected.");
      return;
    }
    auto tf_compatible_load_options =
        dynamic_cast<const TfCompatibleInterpreterCreateOptions*>(
            &create_options);
    if (tf_compatible_load_options != nullptr) {
      tf_saved_model_runner_ = CreateTfSavedModelRunner(
          model_path, tf_compatible_load_options->tfrt_options);
    } else {
      LOG(WARNING)
          << "No options for loading and running TF SavedModel is provided "
             "during loading the model. Using the default options. "
             "This will greatly impact the performance and should only be "
             "used for testing.";
      tf_saved_model_runner_ =
          CreateTfSavedModelRunner(model_path, TfrtOptions());
    }
  });
  TF_ASSIGN_OR_RETURN(std::shared_ptr<TfSavedModelRunner> tf_saved_model_runner,
                      tf_saved_model_runner_);

  // We do not need to verify that the signature name exists in the TF
  // SavedModel here. This will be done in the kernel.
  KernelBuilderContext builder_context;
  TF_RETURN_IF_ERROR(
      builder_context.SetConfig(kTfSavedModelRunner, tf_saved_model_runner));
  TF_RETURN_IF_ERROR(
      builder_context.SetConfig(kTfFunctionSignatureName, signature_name));

  TF_RETURN_IF_ERROR(
      builder_context.SetConfig(kTfFunctionOrderedInputNames, input_names));
  TF_RETURN_IF_ERROR(
      builder_context.SetConfig(kTfFunctionOrderedOutputNames, output_names));
  if (named_inputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature input_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_inputs));
    LOG(INFO) << "Flat input signature: " << input_flat_signature.DebugString();
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kTfFunctionFlatInputSignature, std::move(input_flat_signature)));
  }
  if (named_outputs != nullptr) {
    TF_ASSIGN_OR_RETURN(FlatSignature output_flat_signature,
                        GetFlatSignatureFromNamedSignature(*named_outputs));
    LOG(INFO) << "Flat output signature: "
              << output_flat_signature.DebugString();
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kTfFunctionFlatOutputSignature, std::move(output_flat_signature)));
  }

  TF_RETURN_IF_ERROR(builder_context.SetConfig(kTfFunctionPreDataConversion,
                                               pre_data_conversion));

  TF_RETURN_IF_ERROR(builder_context.SetConfig(kTfFunctionPostDataConversion,
                                               post_data_conversion));

  TF_ASSIGN_OR_RETURN(tf_kernel,
                      GetGlobalKernelRegistry().Create(
                          "TfSavedModelFunctionRunnerOp", builder_context));

  return tf_kernel;
}

static const bool kUnused = []() {
  RegisterOptionalFunctionHandlerFactory<
      TfCompatibleOptionalFunctionHandlerFactory>(
      kTfConcreteFunctionHandleMimeTypeLegacy);
  RegisterOptionalFunctionHandlerFactory<
      TfCompatibleOptionalFunctionHandlerFactory>(
      kTfConcreteFunctionHandleMimeType);
  return true;
}();

}  // namespace xla_serving

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

#include "runtime/neptune_data/composite_processor_function_handler.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/composite_processor.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/interpreter/optional_function_handler_factory.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/neptune_data/composite_processor_op.h"  // from @xla_serving
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {
namespace composite {

absl::StatusOr<std::unique_ptr<Kernel>>
CompositeProcessorFunctionHandler::HandleFunction(
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
  const xla_saved_model_manifest::UnstructuredData& processor_body =
      function.body().other();

  TF_RETURN_IF_ERROR(CheckMimeAndDataType(
      processor_body,
      {kCompositeProcessorMimeType, kLegacyCompositeProcessorMimeType},
      {xla_saved_model_manifest::UnstructuredData::kFileSystemLocation},
      "Function body filesystem path for CompositeProcessor"));

  std::string absolute_config_path = tsl::io::JoinPath(
      model_path, processor_body.file_system_location().string_path());
  std::string serialized_processor_config;
  TF_RETURN_IF_ERROR(tsl::ReadFileToString(
      tsl::Env::Default(), absolute_config_path, &serialized_processor_config));

  neptune::data::composite::CompositeProcessorConfig config;
  if (!config.ParseFromString(serialized_processor_config)) {
    return absl::InvalidArgumentError(
        "Failed to parse CompositeProcessorConfig");
  }
  // Rewrite all paths in the config to be absolute paths given the model path.
  std::string base_asset_path(model_path);
  if (!config.custom_asset_path().empty()) {
    if (tsl::io::IsAbsolutePath(config.custom_asset_path())) {
      base_asset_path = config.custom_asset_path();
    } else {
      base_asset_path =
          tsl::io::JoinPath(model_path, config.custom_asset_path());
    }
    base_asset_path = tsl::io::CleanPath(base_asset_path);
  }

  for (auto& resource : *config.mutable_resources()) {
    auto& assets = *resource.mutable_assets();
    for (auto& asset : assets) {
      if (asset.has_path()) {
        asset.set_path(tsl::io::JoinPath(base_asset_path, asset.path()));
      }
    }
  }
  TF_ASSIGN_OR_RETURN(
      neptune::data::composite::core::CompositeProcessor processor,
      neptune::data::composite::core::CompositeProcessor::FromProto(config));

  return std::make_unique<CompositeProcessorOp>(
      std::make_unique<neptune::data::composite::core::CompositeProcessor>(
          std::move(processor)));
}

static const bool kUnused = []() {
  RegisterOptionalFunctionHandlerFactory<
      CompositeProcessorFunctionHandlerFactory>(kCompositeProcessorMimeType);
  RegisterOptionalFunctionHandlerFactory<
      CompositeProcessorFunctionHandlerFactory>(
      kLegacyCompositeProcessorMimeType);
  return true;
}();

}  // namespace composite
}  // namespace xla_serving

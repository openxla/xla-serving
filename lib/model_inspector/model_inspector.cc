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

#include "lib/model_inspector/model_inspector.h"  // from @neptune

#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "third_party/gloop/util/status/status_macros.h"
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune

namespace neptune {

namespace {

constexpr absl::string_view kOrchestrationFilename = "orchestration.pb";
constexpr absl::string_view kManifestFilename = "manifest.pb";

}  // namespace

neptune::BatchAccessor::BatchAccessor(
    const neptune_model_orchestration::BatchOptions& options)
    : options_(options) {}

int32_t neptune::BatchAccessor::max_batch_size() const {
  return options_.max_batch_size();
}

int32_t neptune::BatchAccessor::batch_timeout_micros() const {
  return options_.batch_timeout_micros();
}

std::vector<int32_t> neptune::BatchAccessor::allowed_batch_sizes() const {
  return std::vector<int32_t>(options_.allowed_batch_sizes().begin(),
                              options_.allowed_batch_sizes().end());
}

bool neptune::BatchAccessor::disable_large_batch_splitting() const {
  return options_.disable_large_batch_splitting();
}

int32_t neptune::BatchAccessor::num_batch_threads() const {
  return options_.num_batch_threads();
}

int32_t neptune::BatchAccessor::max_enqueued_batches() const {
  return options_.max_enqueued_batches();
}

absl::StatusOr<neptune::ModelInspector> neptune::ModelInspector::Create(
    absl::string_view model_root_path) {
  std::string orchestration_path =
      file::JoinPath(model_root_path, kOrchestrationFilename);
  std::string manifest_path =
      file::JoinPath(model_root_path, kManifestFilename);

  std::string content;

  RETURN_IF_ERROR(
      file::GetContents(orchestration_path, &content, file::Defaults()));
  neptune_model_orchestration::Pipelines pipelines;
  if (!pipelines.ParseFromString(content)) {
    return absl::InvalidArgumentError(
        "Failed to parse Pipelines proto from file.");
  }

  RETURN_IF_ERROR(file::GetContents(manifest_path, &content, file::Defaults()));
  neptune_model_manifest::Manifest manifest;
  if (!manifest.ParseFromString(content)) {
    return absl::InvalidArgumentError(
        "Failed to parse Manifest proto from file.");
  }

  return ModelInspector(std::move(pipelines), std::move(manifest));
}

absl::StatusOr<neptune::BatchAccessor>
neptune::ModelInspector::GetBatchAccessor(
    absl::string_view pipeline_name) const {
  const auto& map = pipelines_.name_to_pipeline();
  auto it = map.find(std::string(pipeline_name));
  if (it == map.end()) {
    return absl::NotFoundError(
        absl::StrCat("Pipeline not found: ", pipeline_name));
  }

  const auto& pipeline = it->second;
  if (!pipeline.has_batch_options()) {
    return absl::NotFoundError(
        absl::StrCat("Batch options not found for pipeline: ", pipeline_name));
  }

  return BatchAccessor(pipeline.batch_options());
}

}  // namespace neptune

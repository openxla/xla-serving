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

#ifndef THIRD_PARTY_NEPTUNE_MODEL_MODEL_INSPECTOR_MODEL_INSPECTOR_H_
#define THIRD_PARTY_NEPTUNE_MODEL_MODEL_INSPECTOR_MODEL_INSPECTOR_H_

#include <cstdint>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @neptune
#include "protos/orchestration.pb.h"  // from @neptune

namespace neptune {

class ModelInspector;

class BatchAccessor {
 public:
  int32_t max_batch_size() const;
  int32_t batch_timeout_micros() const;
  std::vector<int32_t> allowed_batch_sizes() const;
  bool disable_large_batch_splitting() const;
  int32_t num_batch_threads() const;
  int32_t max_enqueued_batches() const;

 private:
  friend class ModelInspector;
  explicit BatchAccessor(
      const neptune_model_orchestration::BatchOptions& options);
  neptune_model_orchestration::BatchOptions options_;
};

class ModelInspector {
 public:
  static absl::StatusOr<ModelInspector> Create(
      absl::string_view model_root_path);

  absl::StatusOr<BatchAccessor> GetBatchAccessor(
      absl::string_view pipeline_name) const;

 private:
  ModelInspector(neptune_model_orchestration::Pipelines pipelines,
                 neptune_model_manifest::Manifest manifest)
      : pipelines_(std::move(pipelines)), manifest_(std::move(manifest)) {}

  neptune_model_orchestration::Pipelines pipelines_;
  neptune_model_manifest::Manifest manifest_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_MODEL_MODEL_INSPECTOR_MODEL_INSPECTOR_H_

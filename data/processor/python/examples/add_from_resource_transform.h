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

#ifndef XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_
#define XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/processor/python/examples/int_resource.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace neptune::data {
namespace composite {

// Simple transform for testing purposes that adds an integer loaded from a
// resource.
class AddFromResourceTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddFromResourceTransform>> Create(
      std::shared_ptr<const IntResource> resource);
  absl::StatusOr<int> operator()(int a);
  absl::StatusOr<TransformConfig> ToProto() const override;
  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;
  absl::string_view DebugString() const override {
    return "AddFromResourceTransform";
  }

 private:
  explicit AddFromResourceTransform(
      std::shared_ptr<const IntResource> resource);
  std::shared_ptr<const IntResource> resource_;
};

// Simple transform for testing purposes that adds an integer loaded from a
// resource using the content field.
class AddFromResourceContentTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddFromResourceContentTransform>>
  Create(std::shared_ptr<const IntContentResource> resource);
  absl::StatusOr<int> operator()(int a);
  absl::StatusOr<TransformConfig> ToProto() const override;
  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;
  absl::string_view DebugString() const override {
    return "AddFromResourceContentTransform";
  }

 private:
  explicit AddFromResourceContentTransform(
      std::shared_ptr<const IntContentResource> resource);
  std::shared_ptr<const IntContentResource> resource_;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_

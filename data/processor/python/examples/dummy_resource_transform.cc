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

#include "data/processor/python/examples/dummy_resource_transform.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving

namespace neptune::data {
namespace composite {

absl::Status DummyResource::LoadImpl() {
  return absl::OkStatus();
}

ND_REGISTER_RESOURCE(DummyResource, "DummyResource");

absl::StatusOr<std::unique_ptr<DummyResourceTransform>>
DummyResourceTransform::Create() {
  auto resource = std::make_shared<DummyResource>(
      /*id=*/123, /*name=*/"dummy_resource", /*assets=*/std::vector<Asset>{},
      "DummyResource");
  return std::make_unique<DummyResourceTransform>(resource);
}

DummyResourceTransform::DummyResourceTransform(
    std::shared_ptr<neptune::data::Resource> resource) {
  resources_.push_back(resource);
}

void DummyResourceTransform::operator()() {}

absl::StatusOr<TransformConfig> DummyResourceTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("DummyResourceTransform");
  if (!resources_.empty()) {
    config.add_resource_ids(resources_[0]->id());
  }
  return config;
}

std::vector<std::shared_ptr<neptune::data::Resource>>
DummyResourceTransform::GetAllResources() const {
  return resources_;
}

ND_REGISTER_TRANSFORM(DummyResourceTransform, "DummyResourceTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          for (uint64_t id : config.resource_ids()) {
            for (const auto& r : resources) {
              if (r->id() == id) {
                return std::make_unique<DummyResourceTransform>(r);
              }
            }
          }
          return DummyResourceTransform::Create();
        });

}  // namespace composite
}  // namespace neptune::data

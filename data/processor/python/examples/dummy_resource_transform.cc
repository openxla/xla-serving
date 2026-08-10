#include "data/processor/python/examples/dummy_resource_transform.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @neptune

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

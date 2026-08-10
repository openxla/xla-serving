#include "data/processor/python/examples/add_from_resource_transform.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/processor/python/examples/int_resource.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

AddFromResourceTransform::AddFromResourceTransform(
    std::shared_ptr<const IntResource> resource)
    : resource_(std::move(resource)) {}

absl::StatusOr<std::unique_ptr<AddFromResourceTransform>>
AddFromResourceTransform::Create(std::shared_ptr<const IntResource> resource) {
  return absl::WrapUnique(new AddFromResourceTransform(std::move(resource)));
}

std::vector<std::shared_ptr<neptune::data::Resource>>
AddFromResourceTransform::GetAllResources() const {
  return {std::static_pointer_cast<neptune::data::Resource>(
      std::const_pointer_cast<IntResource>(resource_))};
}

absl::StatusOr<int> AddFromResourceTransform::operator()(int a) {
  if (resource_ == nullptr) {
    return absl::FailedPreconditionError(
        "Resource not loaded. Ensure this transform runs in a "
        "CompositeProcessor and is properly deserialized.");
  }
  return a + resource_->value();
}

absl::StatusOr<TransformConfig> AddFromResourceTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("AddFromResourceTransform");
  if (resource_ != nullptr) {
    config.add_resource_ids(resource_->id());
  }
  return config;
}

ND_REGISTER_TRANSFORM(AddFromResourceTransform, "AddFromResourceTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          if (resources.empty()) {
            return absl::InternalError(
                "AddFromResourceTransform requires a resource.");
          }
          auto int_resource =
              std::dynamic_pointer_cast<const IntResource>(resources[0]);
          if (int_resource == nullptr) {
            return absl::InternalError("Resource is not an IntResource");
          }
          return AddFromResourceTransform::Create(std::move(int_resource));
        });

AddFromResourceContentTransform::AddFromResourceContentTransform(
    std::shared_ptr<const IntContentResource> resource)
    : resource_(std::move(resource)) {}

absl::StatusOr<std::unique_ptr<AddFromResourceContentTransform>>
AddFromResourceContentTransform::Create(
    std::shared_ptr<const IntContentResource> resource) {
  return absl::WrapUnique(
      new AddFromResourceContentTransform(std::move(resource)));
}

std::vector<std::shared_ptr<neptune::data::Resource>>
AddFromResourceContentTransform::GetAllResources() const {
  return {std::static_pointer_cast<neptune::data::Resource>(
      std::const_pointer_cast<IntContentResource>(resource_))};
}

absl::StatusOr<int> AddFromResourceContentTransform::operator()(int a) {
  if (resource_ == nullptr) {
    return absl::FailedPreconditionError(
        "Resource not loaded. Ensure this transform runs in a "
        "CompositeProcessor and is properly deserialized.");
  }
  return a + resource_->value();
}

absl::StatusOr<TransformConfig> AddFromResourceContentTransform::ToProto()
    const {
  TransformConfig config;
  config.set_op_name("AddFromResourceContentTransform");
  if (resource_ != nullptr) {
    config.add_resource_ids(resource_->id());
  }
  return config;
}

ND_REGISTER_TRANSFORM(AddFromResourceContentTransform,
                      "AddFromResourceContentTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          if (resources.empty()) {
            return absl::InternalError(
                "AddFromResourceContentTransform requires a resource.");
          }
          auto int_resource =
              std::dynamic_pointer_cast<const IntContentResource>(resources[0]);
          if (int_resource == nullptr) {
            return absl::InternalError("Resource is not an IntContentResource");
          }
          return AddFromResourceContentTransform::Create(
              std::move(int_resource));
        });

}  // namespace composite
}  // namespace neptune::data

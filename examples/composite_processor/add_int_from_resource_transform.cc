#include "examples/composite_processor/add_int_from_resource_transform.h"  // from @neptune

#include <cstdint>
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
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @neptune
#include "examples/composite_processor/int_resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace neptune_examples {
namespace composite {

AddIntFromResourceTransform::AddIntFromResourceTransform(
    std::shared_ptr<const IntResource> resource)
    : resource_(std::move(resource)) {}

absl::StatusOr<std::unique_ptr<AddIntFromResourceTransform>>
AddIntFromResourceTransform::Create(
    std::shared_ptr<const IntResource> resource) {
  return absl::WrapUnique(new AddIntFromResourceTransform(std::move(resource)));
}

std::vector<std::shared_ptr<neptune::data::Resource>>
AddIntFromResourceTransform::GetAllResources() const {
  if (resource_ == nullptr) {
    return {};
  }
  return {std::static_pointer_cast<neptune::data::Resource>(
      std::const_pointer_cast<IntResource>(resource_))};
}

absl::StatusOr<neptune::HostBuffer> AddIntFromResourceTransform::operator()(
    neptune::HostBuffer in_buffer) {
  if (resource_ == nullptr) {
    return absl::FailedPreconditionError(
        "Resource not loaded. Ensure this transform runs in a "
        "CompositeProcessor and is properly deserialized.");
  }
  if (in_buffer.dtype().kind() != xla::ifrt::DType::kS32) {
    return absl::InvalidArgumentError("Input buffer must have S32 dtype");
  }

  neptune::HostBuffer out_buffer =
      neptune::HostBuffer(in_buffer.dtype(), in_buffer.shape());

  int64_t num_elements = in_buffer.shape().num_elements();
  const int* data = static_cast<const int*>(in_buffer.base());
  int* out_data = static_cast<int*>(out_buffer.base());
  int val = resource_->value();
  for (int64_t i = 0; i < num_elements; ++i) {
    out_data[i] = data[i] + val;
  }
  return out_buffer;
}

absl::StatusOr<neptune::data::composite::TransformConfig>
AddIntFromResourceTransform::ToProto() const {
  neptune::data::composite::TransformConfig config;
  config.set_op_name("AddIntFromResourceTransform");
  if (resource_ != nullptr) {
    config.add_resource_ids(resource_->id());
  }
  return config;
}

ND_REGISTER_TRANSFORM(AddIntFromResourceTransform,
                      "AddIntFromResourceTransform")
    .SetDeserializer(
        [](const neptune::data::composite::TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<
                std::unique_ptr<neptune::data::composite::TransformBase>> {
          if (resources.empty()) {
            return absl::InternalError(
                "AddIntFromResourceTransform requires a resource.");
          }
          auto int_resource =
              std::dynamic_pointer_cast<const IntResource>(resources[0]);
          if (int_resource == nullptr) {
            return absl::InternalError("Resource is not an IntResource");
          }
          return AddIntFromResourceTransform::Create(std::move(int_resource));
        });

}  // namespace composite
}  // namespace neptune_examples

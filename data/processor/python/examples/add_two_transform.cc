#include "data/processor/python/examples/add_two_transform.h"  // from @neptune

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<AddTwoTransform>> AddTwoTransform::Create() {
  return std::make_unique<AddTwoTransform>();
}

int AddTwoTransform::operator()(int a) { return a + 2; }

absl::StatusOr<TransformConfig> AddTwoTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("AddTwoTransform");
  return config;
}

ND_REGISTER_TRANSFORM(AddTwoTransform, "AddTwoTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<AddTwoTransform>();
        });

}  // namespace composite
}  // namespace neptune::data

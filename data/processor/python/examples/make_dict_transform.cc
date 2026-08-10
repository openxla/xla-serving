#include "data/processor/python/examples/make_dict_transform.h"  // from @neptune

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<MakeDictTransform>> MakeDictTransform::Create() {
  return std::make_unique<MakeDictTransform>();
}

std::unordered_map<std::string, int> MakeDictTransform::operator()(
    std::string s, int i) {
  return {{std::move(s), i}};
}

absl::StatusOr<TransformConfig> MakeDictTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("MakeDictTransform");
  return config;
}

ND_REGISTER_TRANSFORM(MakeDictTransform, "MakeDictTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<MakeDictTransform>();
        });

}  // namespace composite
}  // namespace neptune::data

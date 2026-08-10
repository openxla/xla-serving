#include "data/processor/python/examples/suffix_transform.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>

#include "google/protobuf/wrappers.pb.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<SuffixTransform>> SuffixTransform::Create(
    std::string suffix) {
  return std::make_unique<SuffixTransform>(std::move(suffix));
}

std::string SuffixTransform::operator()(std::string val) {
  return val + suffix_;
}

absl::StatusOr<TransformConfig> SuffixTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("SuffixTransform");
  google::protobuf::StringValue suffix_proto;
  suffix_proto.set_value(suffix_);
  config.mutable_detail()->PackFrom(suffix_proto);
  return config;
}

ND_REGISTER_TRANSFORM(SuffixTransform, "SuffixTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          google::protobuf::StringValue suffix_proto;
          if (config.has_detail()) {
            config.detail().UnpackTo(&suffix_proto);
          }
          return std::make_unique<SuffixTransform>(suffix_proto.value());
        });

}  // namespace composite
}  // namespace neptune::data

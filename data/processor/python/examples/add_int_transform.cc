#include "data/processor/python/examples/add_int_transform.h"  // from @neptune

#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<AddIntTransform>> AddIntTransform::Create() {
  return std::make_unique<AddIntTransform>();
}

int AddIntTransform::operator()(int a, int b) { return a + b; }

absl::StatusOr<TransformConfig> AddIntTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("AddIntTransform");
  return config;
}

ND_REGISTER_TRANSFORM(AddIntTransform, "AddIntTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<AddIntTransform>();
        });

absl::Status AddIntTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 2) {
    return absl::InvalidArgumentError("AddIntTransform expects 2 inputs");
  }
  if (!context.inputs[0].IsSpec() || !context.inputs[1].IsSpec()) {
    return absl::InvalidArgumentError(
        "AddIntTransform expects single spec inputs");
  }
  const auto& spec0 = context.inputs[0].spec();
  const auto& spec1 = context.inputs[1].spec();
  if (spec0.shape != spec1.shape) {
    return absl::InvalidArgumentError("Inputs must have the same shape");
  }
  if (spec0.dtype != spec1.dtype) {
    return absl::InvalidArgumentError("Inputs must have the same dtype");
  }
  context.outputs.push_back(context.inputs[0]);
  return absl::OkStatus();
}

}  // namespace composite
}  // namespace neptune::data

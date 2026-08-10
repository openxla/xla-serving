#ifndef THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "examples/composite_processor/int_resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune_examples {
namespace composite {

class AddIntFromResourceTransform
    : public neptune::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddIntFromResourceTransform>> Create(
      std::shared_ptr<const IntResource> resource);
  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer in_buffer);
  absl::StatusOr<neptune::data::composite::TransformConfig> ToProto()
      const override;
  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;

 private:
  explicit AddIntFromResourceTransform(
      std::shared_ptr<const IntResource> resource);
  std::shared_ptr<const IntResource> resource_;
};

}  // namespace composite
}  // namespace neptune_examples

#endif  // THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_

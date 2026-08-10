#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/processor/python/examples/int_resource.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

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

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_FROM_RESOURCE_TRANSFORM_H_

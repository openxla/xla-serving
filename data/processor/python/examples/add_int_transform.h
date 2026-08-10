#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_INT_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_INT_TRANSFORM_H_

#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune

namespace neptune::data {
namespace composite {

// Simple transform for testing purposes.
class AddIntTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddIntTransform>> Create();
  int operator()(int a, int b);
  absl::StatusOr<TransformConfig> ToProto() const override;
  absl::string_view DebugString() const override { return "AddIntTransform"; }

 protected:
  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_INT_TRANSFORM_H_

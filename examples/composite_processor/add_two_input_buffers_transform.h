#ifndef THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune_examples {
namespace composite {

class AddTwoInputBuffersTransform
    : public neptune::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddTwoInputBuffersTransform>> Create();
  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer in_buffer1,
                                             neptune::HostBuffer in_buffer2);
  absl::StatusOr<neptune::data::composite::TransformConfig> ToProto()
      const override;
};

}  // namespace composite
}  // namespace neptune_examples

#endif  // THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_

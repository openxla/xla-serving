#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_SUFFIX_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_SUFFIX_TRANSFORM_H_

#include <memory>
#include <string>
#include <utility>

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

class SuffixTransform : public TransformBase {
 public:
  explicit SuffixTransform(std::string suffix) : suffix_(std::move(suffix)) {}
  static absl::StatusOr<std::unique_ptr<SuffixTransform>> Create(
      std::string suffix);
  std::string operator()(std::string val);
  absl::StatusOr<TransformConfig> ToProto() const override;
  absl::string_view DebugString() const override { return "SuffixTransform"; }

 private:
  std::string suffix_;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_SUFFIX_TRANSFORM_H_

#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MAKE_DICT_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MAKE_DICT_TRANSFORM_H_

#include <memory>
#include <string>
#include <unordered_map>

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

class MakeDictTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MakeDictTransform>> Create();
  std::unordered_map<std::string, int> operator()(std::string s, int i);
  absl::StatusOr<TransformConfig> ToProto() const override;
  absl::string_view DebugString() const override { return "MakeDictTransform"; }
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MAKE_DICT_TRANSFORM_H_

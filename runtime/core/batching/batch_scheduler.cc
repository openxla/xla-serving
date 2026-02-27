#include "third_party/neptune/runtime/core/batching/batch_scheduler.h"

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_format.h"
#include "third_party/absl/strings/string_view.h"

namespace jsv {

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    absl::string_view attr_value) {
  if (attr_value == kLowPriorityPaddingWithMaxBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
  } else if (attr_value ==
             kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue) {
    return MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize;
  } else if (attr_value == kPriorityIsolationAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityIsolation;
  } else if (attr_value == kPriorityMergeAttrValue) {
    return MixedPriorityBatchingPolicy::kPriorityMerge;
  }
  return absl::InvalidArgumentError(absl::StrFormat(
      "Unknown mixed priority batching policy: %s", attr_value));
}

}  // namespace jsv

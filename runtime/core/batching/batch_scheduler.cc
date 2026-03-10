#include "runtime/core/batching/batch_scheduler.h"  // from @neptune

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/str_format.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl

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

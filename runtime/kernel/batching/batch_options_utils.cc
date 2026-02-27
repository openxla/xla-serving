#include "third_party/neptune/runtime/kernel/batching/batch_options_utils.h"

#include <string>

#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/neptune/runtime/core/batching/batch_scheduler.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"

namespace jsv {

absl::StatusOr<std::string> GetBatchPaddingPolicyString(
    BatchOptions::BatchPaddingPolicy batch_padding_policy) {
  switch (batch_padding_policy) {
    case BatchOptions::PADDING_POLICY_UNSPECIFIED:
    case BatchOptions::PAD_UP:
      return std::string(kPadUpPolicy);
    case BatchOptions::BATCH_DOWN:
      return std::string(kBatchDownPolicy);
    case BatchOptions::MINIMIZE_TPU_COST_PER_REQUEST:
      return std::string(kMinimizeTpuCostPerRequestPolicy);
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported batch padding policy: ", batch_padding_policy));
  }
}

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    BatchOptions::MixedPriorityBatchingPolicy mixed_priority_batching_policy) {
  switch (mixed_priority_batching_policy) {
    case BatchOptions::MIXED_PRIORITY_BATCHING_POLICY_UNSPECIFIED:
    case BatchOptions::LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE:
      return MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize;
    case BatchOptions::LOW_PRIORITY_PADDING_WITH_NEXT_ALLOWED_BATCH_SIZE:
      return MixedPriorityBatchingPolicy::
          kLowPriorityPaddingWithNextAllowedBatchSize;
    case BatchOptions::PRIORITY_ISOLATION:
      return MixedPriorityBatchingPolicy::kPriorityIsolation;
    case BatchOptions::PRIORITY_MERGE:
      return MixedPriorityBatchingPolicy::kPriorityMerge;
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported mixed priority batching policy: ",
                       mixed_priority_batching_policy));
  }
}

}  // namespace jsv

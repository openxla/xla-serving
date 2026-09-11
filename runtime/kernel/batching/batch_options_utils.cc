// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "runtime/kernel/batching/batch_options_utils.h"  // from @neptune

#include <string>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "xla/tsl/platform/criticality.h"  // from @xla

namespace xla_serving {

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

absl::string_view MixedPriorityBatchingPolicyToString(
    MixedPriorityBatchingPolicy policy) {
  switch (policy) {
    case MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize:
      return kLowPriorityPaddingWithMaxBatchSizeAttrValue;
    case MixedPriorityBatchingPolicy::
        kLowPriorityPaddingWithNextAllowedBatchSize:
      return kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue;
    case MixedPriorityBatchingPolicy::kPriorityIsolation:
      return kPriorityIsolationAttrValue;
    case MixedPriorityBatchingPolicy::kPriorityMerge:
      return kPriorityMergeAttrValue;
  }
  return "unknown_mixed_priority_batching_policy";
}

absl::StatusOr<PriorityBatchingPolicy> GetPriorityBatchingPolicy(
    BatchOptions::PriorityBatchingPolicy priority_batching_policy) {
  switch (priority_batching_policy) {
    case BatchOptions::PRIORITY_BATCHING_POLICY_UNSPECIFIED:
    case BatchOptions::STRICT_FIFO:
      return PriorityBatchingPolicy::kStrictFifo;
    case BatchOptions::PRIORITY_AWARE:
      return PriorityBatchingPolicy::kPriorityAware;
    default:
      return absl::InvalidArgumentError(absl::StrCat(
          "Unsupported priority batching policy: ", priority_batching_policy));
  }
}

absl::string_view PriorityBatchingPolicyToString(
    PriorityBatchingPolicy policy) {
  switch (policy) {
    case PriorityBatchingPolicy::kStrictFifo:
      return kStrictFifoAttrValue;
    case PriorityBatchingPolicy::kPriorityAware:
      return kPriorityAwareAttrValue;
  }
  return "unknown_priority_batching_policy";
}

absl::StatusOr<PriorityAwareBatchingOptions> GetPriorityAwareBatchingOptions(
    const BatchOptions::PriorityAwareBatchingOptions& options) {
  if (options.max_queue_depth() < 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("`priority_aware_batching_options.max_queue_depth` must "
                     "be non-negative, but got: ",
                     options.max_queue_depth()));
  }
  return PriorityAwareBatchingOptions{
      .max_queue_depth = static_cast<size_t>(options.max_queue_depth()),
      .enable_task_resplit = options.enable_task_resplit(),
  };
}

absl::string_view CriticalityToString(
    tsl::criticality::Criticality criticality) {
  switch (criticality) {
    case tsl::criticality::Criticality::kSheddable:
      return "sheddable";
    case tsl::criticality::Criticality::kSheddablePlus:
      return "sheddable_plus";
    case tsl::criticality::Criticality::kCritical:
      return "critical";
    case tsl::criticality::Criticality::kCriticalPlus:
      return "critical_plus";
  }
  return "unknown_criticality";
}

}  // namespace xla_serving

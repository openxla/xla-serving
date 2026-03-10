#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "third_party/py/orbax/export/protos/oex_orchestration.pb.h"

namespace jsv {

using orbax_export_orchestration::BatchOptions;

// batch_padding_policy string constants matching with the
// BatchOptions.BatchPaddingPolicy enum.
//
// See the documentation of BatchOptions.BatchPaddingPolicy for details.
inline constexpr absl::string_view kPadUpPolicy = "PAD_UP";
inline constexpr absl::string_view kBatchDownPolicy = "BATCH_DOWN";
inline constexpr absl::string_view kMinimizeTpuCostPerRequestPolicy =
    "MINIMIZE_TPU_COST_PER_REQUEST";

absl::StatusOr<std::string> GetBatchPaddingPolicyString(
    BatchOptions::BatchPaddingPolicy batch_padding_policy);

absl::StatusOr<MixedPriorityBatchingPolicy> GetMixedPriorityBatchingPolicy(
    BatchOptions::MixedPriorityBatchingPolicy mixed_priority_batching_policy);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_

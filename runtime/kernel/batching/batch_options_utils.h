#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_

#include <string>

#include "learning/infra/mira/experimental/orbax_model/core/batching/batch_scheduler.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"

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

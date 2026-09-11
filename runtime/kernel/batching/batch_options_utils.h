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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_

#include <string>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "protos/orchestration.pb.h"  // from @neptune
#include "runtime/core/batching/batch_scheduler.h"  // from @neptune
#include "xla/tsl/platform/criticality.h"  // from @xla

namespace xla_serving {

using neptune_model_orchestration::BatchOptions;

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

absl::string_view MixedPriorityBatchingPolicyToString(
    MixedPriorityBatchingPolicy policy);

absl::StatusOr<PriorityBatchingPolicy> GetPriorityBatchingPolicy(
    BatchOptions::PriorityBatchingPolicy priority_batching_policy);

absl::string_view PriorityBatchingPolicyToString(
    PriorityBatchingPolicy policy);

absl::StatusOr<PriorityAwareBatchingOptions> GetPriorityAwareBatchingOptions(
    const BatchOptions::PriorityAwareBatchingOptions& options);

absl::string_view CriticalityToString(
    tsl::criticality::Criticality criticality);

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_BATCHING_BATCH_OPTIONS_UTILS_H_

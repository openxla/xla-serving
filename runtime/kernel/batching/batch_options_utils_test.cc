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

#include "runtime/kernel/batching/batch_options_utils.h"  // from @xla_serving

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "protos/orchestration.pb.h"  // from @xla_serving
#include "runtime/core/batching/batch_scheduler.h"  // from @xla_serving
#include "xla/tsl/platform/criticality.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::absl_testing::StatusIs;
using xla_saved_model_orchestration::BatchOptions;
using ::testing::HasSubstr;

TEST(BatchOptionsUtilsTest, GetBatchPaddingPolicyString) {
  EXPECT_THAT(
      GetBatchPaddingPolicyString(BatchOptions::PADDING_POLICY_UNSPECIFIED),
      IsOkAndHolds(kPadUpPolicy));
  EXPECT_THAT(GetBatchPaddingPolicyString(BatchOptions::PAD_UP),
              IsOkAndHolds(kPadUpPolicy));
  EXPECT_THAT(GetBatchPaddingPolicyString(BatchOptions::BATCH_DOWN),
              IsOkAndHolds(kBatchDownPolicy));
  EXPECT_THAT(
      GetBatchPaddingPolicyString(BatchOptions::MINIMIZE_TPU_COST_PER_REQUEST),
      IsOkAndHolds(kMinimizeTpuCostPerRequestPolicy));
  EXPECT_THAT(GetBatchPaddingPolicyString(
                  static_cast<BatchOptions::BatchPaddingPolicy>(4)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported batch padding policy: 4"));
}

TEST(BatchOptionsUtilsTest, GetMixedPriorityBatchingPolicy) {
  EXPECT_THAT(
      GetMixedPriorityBatchingPolicy(
          BatchOptions::MIXED_PRIORITY_BATCHING_POLICY_UNSPECIFIED),
      IsOkAndHolds(
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize));
  EXPECT_THAT(
      GetMixedPriorityBatchingPolicy(
          BatchOptions::LOW_PRIORITY_PADDING_WITH_MAX_BATCH_SIZE),
      IsOkAndHolds(
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize));
  EXPECT_THAT(
      GetMixedPriorityBatchingPolicy(
          BatchOptions::LOW_PRIORITY_PADDING_WITH_NEXT_ALLOWED_BATCH_SIZE),
      IsOkAndHolds(MixedPriorityBatchingPolicy::
                       kLowPriorityPaddingWithNextAllowedBatchSize));
  EXPECT_THAT(GetMixedPriorityBatchingPolicy(BatchOptions::PRIORITY_ISOLATION),
              IsOkAndHolds(MixedPriorityBatchingPolicy::kPriorityIsolation));
  EXPECT_THAT(GetMixedPriorityBatchingPolicy(BatchOptions::PRIORITY_MERGE),
              IsOkAndHolds(MixedPriorityBatchingPolicy::kPriorityMerge));
  EXPECT_THAT(GetMixedPriorityBatchingPolicy(
                  static_cast<BatchOptions::MixedPriorityBatchingPolicy>(5)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported mixed priority batching policy: 5"));
}

TEST(BatchOptionsUtilsTest, MixedPriorityBatchingPolicyToString) {
  EXPECT_EQ(
      MixedPriorityBatchingPolicyToString(
          MixedPriorityBatchingPolicy::kLowPriorityPaddingWithMaxBatchSize),
      kLowPriorityPaddingWithMaxBatchSizeAttrValue);
  EXPECT_EQ(MixedPriorityBatchingPolicyToString(
                MixedPriorityBatchingPolicy::
                    kLowPriorityPaddingWithNextAllowedBatchSize),
            kLowPriorityPaddingWithNextAllowedBatchSizeAttrValue);
  EXPECT_EQ(MixedPriorityBatchingPolicyToString(
                MixedPriorityBatchingPolicy::kPriorityIsolation),
            kPriorityIsolationAttrValue);
  EXPECT_EQ(MixedPriorityBatchingPolicyToString(
                MixedPriorityBatchingPolicy::kPriorityMerge),
            kPriorityMergeAttrValue);
  EXPECT_EQ(MixedPriorityBatchingPolicyToString(
                static_cast<MixedPriorityBatchingPolicy>(5)),
            "unknown_mixed_priority_batching_policy");
}

TEST(BatchOptionsUtilsTest, GetSchedulerScope) {
  EXPECT_THAT(GetSchedulerScope(BatchOptions::SCHEDULER_SCOPE_UNSPECIFIED),
              IsOkAndHolds(SchedulerScope::kLocalToPipeline));
  EXPECT_THAT(GetSchedulerScope(BatchOptions::LOCAL_TO_PIPELINE),
              IsOkAndHolds(SchedulerScope::kLocalToPipeline));
  EXPECT_THAT(GetSchedulerScope(BatchOptions::SHARED_GLOBAL),
              IsOkAndHolds(SchedulerScope::kSharedGlobal));
  EXPECT_THAT(GetSchedulerScope(static_cast<BatchOptions::SchedulerScope>(3)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported scheduler scope: 3"));
}

TEST(BatchOptionsUtilsTest, SchedulerScopeToString) {
  EXPECT_EQ(SchedulerScopeToString(SchedulerScope::kLocalToPipeline),
            kLocalToPipelineAttrValue);
  EXPECT_EQ(SchedulerScopeToString(SchedulerScope::kSharedGlobal),
            kSharedGlobalAttrValue);
  EXPECT_EQ(SchedulerScopeToString(static_cast<SchedulerScope>(3)),
            "unknown_scheduler_scope");
}

TEST(BatchOptionsUtilsTest, GetQueueSelectionPolicy) {
  EXPECT_THAT(
      GetQueueSelectionPolicy(BatchOptions::QUEUE_SELECTION_POLICY_UNSPECIFIED),
      IsOkAndHolds(QueueSelectionPolicy::kRoundRobin));
  EXPECT_THAT(GetQueueSelectionPolicy(BatchOptions::ROUND_ROBIN),
              IsOkAndHolds(QueueSelectionPolicy::kRoundRobin));
  EXPECT_THAT(GetQueueSelectionPolicy(BatchOptions::PRIORITY_RANKED),
              IsOkAndHolds(QueueSelectionPolicy::kPriorityRanked));
  EXPECT_THAT(GetQueueSelectionPolicy(
                  static_cast<BatchOptions::QueueSelectionPolicy>(3)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported queue selection policy: 3"));
}

TEST(BatchOptionsUtilsTest, QueueSelectionPolicyToString) {
  EXPECT_EQ(QueueSelectionPolicyToString(QueueSelectionPolicy::kRoundRobin),
            kRoundRobinAttrValue);
  EXPECT_EQ(QueueSelectionPolicyToString(QueueSelectionPolicy::kPriorityRanked),
            kPriorityRankedAttrValue);
  EXPECT_EQ(QueueSelectionPolicyToString(static_cast<QueueSelectionPolicy>(3)),
            "unknown_queue_selection_policy");
}

TEST(BatchOptionsUtilsTest, BasicSchedulerConfig) {
  BatchOptions::SchedulerConfig proto_config;
  proto_config.set_scheduler_scope(BatchOptions::SHARED_GLOBAL);
  proto_config.set_queue_selection_policy(BatchOptions::PRIORITY_RANKED);

  EXPECT_THAT(GetSchedulerConfig(proto_config),
              IsOkAndHolds(SchedulerConfig{
                  .scheduler_scope = SchedulerScope::kSharedGlobal,
                  .queue_selection_policy =
                      QueueSelectionPolicy::kPriorityRanked,
              }));
}

TEST(BatchOptionsUtilsTest, DefaultSchedulerConfig) {
  BatchOptions::SchedulerConfig default_config;

  EXPECT_THAT(GetSchedulerConfig(default_config),
              IsOkAndHolds(SchedulerConfig{
                  .scheduler_scope = SchedulerScope::kLocalToPipeline,
                  .queue_selection_policy = QueueSelectionPolicy::kRoundRobin,
              }));
}

TEST(BatchOptionsUtilsTest, InvalidSchedulerConfig) {
  BatchOptions::SchedulerConfig invalid_scope_config;
  invalid_scope_config.set_scheduler_scope(
      static_cast<BatchOptions::SchedulerScope>(3));
  BatchOptions::SchedulerConfig invalid_policy_config;
  invalid_policy_config.set_queue_selection_policy(
      static_cast<BatchOptions::QueueSelectionPolicy>(3));

  EXPECT_THAT(GetSchedulerConfig(invalid_scope_config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported scheduler scope: 3"));
  EXPECT_THAT(GetSchedulerConfig(invalid_policy_config),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported queue selection policy: 3"));
}

TEST(BatchOptionsUtilsTest, GetPriorityBatchingPolicy) {
  EXPECT_THAT(
      GetPriorityBatchingPolicy(
          BatchOptions::PRIORITY_BATCHING_POLICY_UNSPECIFIED),
      IsOkAndHolds(PriorityBatchingPolicy::kStrictFifo));
  EXPECT_THAT(GetPriorityBatchingPolicy(BatchOptions::STRICT_FIFO),
              IsOkAndHolds(PriorityBatchingPolicy::kStrictFifo));
  EXPECT_THAT(GetPriorityBatchingPolicy(BatchOptions::PRIORITY_AWARE),
              IsOkAndHolds(PriorityBatchingPolicy::kPriorityAware));
  EXPECT_THAT(GetPriorityBatchingPolicy(
                  static_cast<BatchOptions::PriorityBatchingPolicy>(3)),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       "Unsupported priority batching policy: 3"));
}

TEST(BatchOptionsUtilsTest, PriorityBatchingPolicyToString) {
  EXPECT_EQ(PriorityBatchingPolicyToString(PriorityBatchingPolicy::kStrictFifo),
            kStrictFifoAttrValue);
  EXPECT_EQ(
      PriorityBatchingPolicyToString(PriorityBatchingPolicy::kPriorityAware),
      kPriorityAwareAttrValue);
  EXPECT_EQ(
      PriorityBatchingPolicyToString(static_cast<PriorityBatchingPolicy>(3)),
      "unknown_priority_batching_policy");
}

TEST(BatchOptionsUtilsTest, BasicPriorityAwareBatchingOptions) {
  BatchOptions::PriorityAwareBatchingOptions batch_options;
  batch_options.set_max_queue_depth(100);
  batch_options.set_enable_task_resplit(true);

  EXPECT_THAT(
      GetPriorityAwareBatchingOptions(batch_options),
      IsOkAndHolds(PriorityAwareBatchingOptions{
          .max_queue_depth = 100,
          .enable_task_resplit = true,
      }));
}

TEST(BatchOptionsUtilsTest, DefaultPriorityAwareBatchingOptions) {
  BatchOptions::PriorityAwareBatchingOptions default_options;

  EXPECT_THAT(
      GetPriorityAwareBatchingOptions(default_options),
      IsOkAndHolds(PriorityAwareBatchingOptions{
          .max_queue_depth = 0,
          .enable_task_resplit = false,
      }));
}

TEST(BatchOptionsUtilsTest, NegativeMaxQueueDepth) {
  BatchOptions::PriorityAwareBatchingOptions invalid_proto_options;
  invalid_proto_options.set_max_queue_depth(-1);

  EXPECT_THAT(
      GetPriorityAwareBatchingOptions(invalid_proto_options),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("`priority_aware_batching_options.max_queue_depth` must "
                    "be non-negative")));
}

TEST(BatchOptionsUtilsTest, CriticalityToString) {
  EXPECT_EQ(CriticalityToString(tsl::criticality::Criticality::kSheddable),
            "sheddable");
  EXPECT_EQ(CriticalityToString(tsl::criticality::Criticality::kSheddablePlus),
            "sheddable_plus");
  EXPECT_EQ(CriticalityToString(tsl::criticality::Criticality::kCritical),
            "critical");
  EXPECT_EQ(CriticalityToString(tsl::criticality::Criticality::kCriticalPlus),
            "critical_plus");
  EXPECT_EQ(CriticalityToString(static_cast<tsl::criticality::Criticality>(5)),
            "unknown_criticality");
}

}  // namespace
}  // namespace xla_serving

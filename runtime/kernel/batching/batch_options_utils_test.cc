#include "third_party/neptune/runtime/kernel/batching/batch_options_utils.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/neptune/runtime/core/batching/batch_scheduler.h"
#include "third_party/py/orbax/export/protos/oex_orchestration.proto.h"

namespace jsv {
namespace {

using orbax_export_orchestration::BatchOptions;
using ::testing::status::IsOkAndHolds;
using ::testing::status::StatusIs;

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

}  // namespace
}  // namespace jsv

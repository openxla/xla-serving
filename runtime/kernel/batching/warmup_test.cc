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

#include "runtime/kernel/batching/warmup.h"  // from @xla_serving

#include <memory>
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/lib/monitoring/test_utils.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;
using ::tsl::monitoring::testing::CellReader;
using ::tsl::monitoring::testing::Histogram;

TEST(WarmupStateRegistryTest, RegisterAndLookup) {
  // Before registering, the key should not be present.
  WarmupStateRegistry registry;
  const WarmupStateRegistry::Key key("model", 1);
  EXPECT_THAT(registry.Lookup(key), IsNull());
  // After registration, the key should be present.
  TF_ASSERT_OK_AND_ASSIGN(
      const auto unused_handle,
      registry.Register(key,
                        std::make_unique<WarmupStateRegistry::PerModelData>()));
  ASSERT_THAT(registry.Lookup(key), NotNull());
}

TEST(WarmupStateRegistryTest, HandleReleasesOnDestruction) {
  WarmupStateRegistry registry;
  const WarmupStateRegistry::Key key("model", 1);
  {
    TF_ASSERT_OK_AND_ASSIGN(
        const auto handle,
        registry.Register(
            key, std::make_unique<WarmupStateRegistry::PerModelData>()));
    EXPECT_THAT(registry.Lookup(key), NotNull());
  }
  EXPECT_THAT(registry.Lookup(key), IsNull());
}

TEST(WarmupStateRegistryTest, HandleMove) {
  WarmupStateRegistry registry;
  const WarmupStateRegistry::Key key("model", 1);
  WarmupStateRegistry::Handle outer_handle;
  {
    TF_ASSERT_OK_AND_ASSIGN(
        auto inner_handle,
        registry.Register(
            key, std::make_unique<WarmupStateRegistry::PerModelData>()));
    outer_handle = std::move(inner_handle);
  }
  EXPECT_THAT(registry.Lookup(key), NotNull());
  WarmupStateRegistry::Handle outer_handle2 = std::move(outer_handle);
  EXPECT_THAT(registry.Lookup(key), NotNull());
}

TEST(WarmupStateRegistryGlobalTest, ShouldWarmupAllBatchSizes) {
  const WarmupStateRegistry::Key key1("model1", 1);
  const WarmupStateRegistry::Key key2("model2", 1);
  EXPECT_FALSE(ShouldWarmupAllBatchSizes(key1.name, key1.version));
  EXPECT_FALSE(ShouldWarmupAllBatchSizes(key2.name, key2.version));
  auto data1 = std::make_unique<WarmupStateRegistry::PerModelData>();
  data1->warmup_all_batch_sizes = true;
  TF_ASSERT_OK_AND_ASSIGN(
      const auto handle1,
      GetGlobalWarmupStateRegistry().Register(key1, std::move(data1)));
  auto data2 = std::make_unique<WarmupStateRegistry::PerModelData>();
  data2->warmup_all_batch_sizes = false;
  TF_ASSERT_OK_AND_ASSIGN(
      const auto handle2,
      GetGlobalWarmupStateRegistry().Register(key2, std::move(data2)));
  EXPECT_TRUE(ShouldWarmupAllBatchSizes(key1.name, key1.version));
  EXPECT_FALSE(ShouldWarmupAllBatchSizes(key2.name, key2.version));
}

TEST(RecordModelWarmupLatencyTest, RecordsLatencyByModelVersionAndStatus) {
  CellReader<Histogram> latency("/orbax/serving/model_warmup_latency");

  RecordModelWarmupLatency("model", 1, absl::Seconds(2), absl::OkStatus());
  RecordModelWarmupLatency("model", 1, absl::Seconds(3),
                           absl::InvalidArgumentError("warmup failed"));

  const Histogram ok_latency = latency.Delta("model", "1", "OK");
  EXPECT_EQ(ok_latency.num(), 1);
  EXPECT_EQ(ok_latency.sum(), 2e6);
  const Histogram failed_latency =
      latency.Delta("model", "1", "INVALID_ARGUMENT");
  EXPECT_EQ(failed_latency.num(), 1);
  EXPECT_EQ(failed_latency.sum(), 3e6);
}

}  // namespace
}  // namespace xla_serving

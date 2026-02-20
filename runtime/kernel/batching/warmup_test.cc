#include "third_party/neptune/runtime/kernel/batching/warmup.h"

#include <memory>
#include <utility>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {

using ::testing::IsNull;
using ::testing::NotNull;

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

}  // namespace
}  // namespace jsv

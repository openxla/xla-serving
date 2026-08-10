#include "data/resource_manager/cc/asset.h"  // from @neptune

#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace neptune::data {
namespace {

using ::absl_testing::StatusIs;

TEST(AssetTest, LoadContent) {
  std::string path = tsl::io::JoinPath(::testing::TempDir(), "my_asset");
  ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), path, "test content"));
  Asset asset = {.name = "test", .path = path};
  EXPECT_OK(asset.LoadContent());
  EXPECT_EQ(asset.content, "test content");
}

TEST(AssetTest, LoadContentNotFound) {
  Asset asset = {.name = "test", .path = "/no/such/file"};
  EXPECT_THAT(asset.LoadContent(), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace neptune::data

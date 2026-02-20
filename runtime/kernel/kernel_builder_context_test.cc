#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_matchers.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/status_matchers.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {

TEST(KernelBuilderContextTest, Basic) {
  KernelBuilderContext context;
  TF_ASSERT_OK(context.SetConfig("test_config", 100));
  TF_ASSERT_OK_AND_ASSIGN(int test_config,
                          context.GetConfig<int>("test_config"));
  EXPECT_EQ(test_config, 100);
}

TEST(KernelBuilderContextTest, RepeatedSetConfigShallFail) {
  KernelBuilderContext context;
  TF_ASSERT_OK(context.SetConfig("test_config", 100));
  auto status = context.SetConfig("test_config", 10);
  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(KernelBuilderContextTest, GetUnsetConfigShallFail) {
  KernelBuilderContext context;
  auto status = context.GetConfig<int>("test_config");

  EXPECT_THAT(status, absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST(KernelBuilderContextTest, GetMismatchedTypedConfigShallFail) {
  KernelBuilderContext context;

  TF_ASSERT_OK(context.SetConfig("test_config", 100));
  auto status = context.GetConfig<float>("test_config");

  EXPECT_THAT(status,
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace jsv

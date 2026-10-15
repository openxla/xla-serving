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

#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/status_matchers.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
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
}  // namespace xla_serving

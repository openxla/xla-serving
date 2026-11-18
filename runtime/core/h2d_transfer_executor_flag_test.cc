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

#include "runtime/core/h2d_transfer_executor_flag.h"  // from @xla_serving

#include <memory>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/flags/reflection.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/arena_h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/batched_h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;
using ::testing::IsNull;
using ::testing::NotNull;
using ::testing::WhenDynamicCastTo;

class H2DTransferExecutorFlagTest : public ::testing::Test {
 private:
  absl::FlagSaver flag_saver_;
};

TEST_F(H2DTransferExecutorFlagTest, SelectsNoExecutorByDefault) {
  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<H2DTransferExecutorFactory> factory,
                          CreateH2DTransferExecutorFactoryFromFlag());

  EXPECT_THAT(factory, IsNull());
}

TEST_F(H2DTransferExecutorFlagTest, SelectsArenaExecutor) {
  absl::SetFlag(&FLAGS_xla_serving_h2d_transfer_executor, "arena");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<H2DTransferExecutorFactory> factory,
                          CreateH2DTransferExecutorFactoryFromFlag());

  EXPECT_THAT(factory.get(),
              WhenDynamicCastTo<ArenaH2DTransferExecutorFactory*>(NotNull()));
}

TEST_F(H2DTransferExecutorFlagTest, SelectsBatchedExecutor) {
  absl::SetFlag(&FLAGS_xla_serving_h2d_transfer_executor, "batched");

  TF_ASSERT_OK_AND_ASSIGN(std::unique_ptr<H2DTransferExecutorFactory> factory,
                          CreateH2DTransferExecutorFactoryFromFlag());

  EXPECT_THAT(factory.get(),
              WhenDynamicCastTo<BatchedH2DTransferExecutorFactory*>(NotNull()));
}

TEST_F(H2DTransferExecutorFlagTest, RejectsUnknownExecutor) {
  absl::SetFlag(&FLAGS_xla_serving_h2d_transfer_executor, "Arena");

  EXPECT_THAT(CreateH2DTransferExecutorFactoryFromFlag(),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("--xla_serving_h2d_transfer_executor")));
}

}  // namespace
}  // namespace xla_serving

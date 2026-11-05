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
#include "runtime/core/arena_h2d_transfer_executor.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::NiceMock;
using ::testing::Return;

bool IsBackedByCommonPjRtClient(xla::ifrt::Client* client) {
  auto* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  return pjrt_client != nullptr && dynamic_cast<xla::CommonPjRtClient*>(
                                       pjrt_client->pjrt_client()) != nullptr;
}

bool SupportsArenaTransfer(xla::ifrt::Client* client) {
  if (client == nullptr || client->platform_id() == xla::CpuId()) {
    return false;
  }
  auto* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (pjrt_client == nullptr) {
    return false;
  }
  auto* common_client =
      dynamic_cast<xla::CommonPjRtClient*>(pjrt_client->pjrt_client());
  return common_client != nullptr && common_client->raw_client() != nullptr &&
         !absl::IsUnimplemented(common_client->raw_client()
                                    ->GetDeviceAddressAlignment()
                                    .status());
}

class ArenaH2DTransferExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    if (!SupportsArenaTransfer(client_.get())) {
      GTEST_SKIP()
          << "ArenaH2DTransferExecutor is not active on this test backend.";
    }
    TF_ASSERT_OK_AND_ASSIGN(
        device_list_, xla::ifrt::test_util::GetDevices(client_.get(), {0}));
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "TestThreadPool",
        /*num_threads=*/4);
  }

  std::shared_ptr<xla::ifrt::Client> client_;
  xla::ifrt::DeviceListRef device_list_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST(ArenaH2DTransferExecutorCreateTest, NullClientReturnsInvalidArgument) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);
  EXPECT_THAT(
      ArenaH2DTransferExecutor::Create(/*client=*/nullptr, &thread_pool),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArenaH2DTransferExecutorCreateTest, NullThreadPoolReturnsInvalidArgument) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  EXPECT_THAT(
      ArenaH2DTransferExecutor::Create(client.get(), /*thread_pool=*/nullptr),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArenaH2DTransferExecutorCreateTest, CpuClientReturnsUnimplemented) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  if (client->platform_id() != xla::CpuId()) {
    GTEST_SKIP() << "Client is not a CPU backend.";
  }
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);
  ArenaH2DTransferExecutorFactory factory;

  EXPECT_THAT(ArenaH2DTransferExecutor::Create(client.get(), &thread_pool),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(factory.Create(client.get(), device_list, &thread_pool),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ArenaH2DTransferExecutorCreateTest,
     NonCommonPjRtClientReturnsUnimplemented) {
  NiceMock<xla::ifrt::MockClient> mock_client;
  ON_CALL(mock_client, platform_id()).WillByDefault(Return(xla::TpuId()));
  ON_CALL(mock_client, platform_name()).WillByDefault(Return("tpu"));

  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);

  EXPECT_THAT(ArenaH2DTransferExecutor::Create(&mock_client, &thread_pool),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST(ArenaH2DTransferExecutorCreateTest,
     UnimplementedDeviceAddressAlignmentReturnsUnimplemented) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  if (client->platform_id() == xla::CpuId() ||
          !IsBackedByCommonPjRtClient(client.get()) ||
          SupportsArenaTransfer(client.get())) {
    GTEST_SKIP() << "Requires a non-CPU CommonPjRtClient without "
                    "GetDeviceAddressAlignment support.";
  }
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);

  EXPECT_THAT(ArenaH2DTransferExecutor::Create(client.get(), &thread_pool),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(ArenaH2DTransferExecutorTest, SizeMismatchReturnsInvalidArgument) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  ASSERT_NE(executor, nullptr);

  HostBuffer input0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer input1 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));
  std::vector<HostBuffer> inputs = {input0, input1};

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  EXPECT_THAT(
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ArenaH2DTransferExecutorTest, EmptyInputsSucceeds) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  std::vector<HostBuffer> inputs = {};
  std::vector<xla::ifrt::ShardingRef> shardings = {};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  EXPECT_TRUE(results.empty());
}

TEST_F(ArenaH2DTransferExecutorTest, FallbackTransferSingleDevice) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer in0 =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({2, 2}));

  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest, StaticShapeIsUnimplemented) {
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({1, 3}));

  EXPECT_THAT(
      executor->ExecuteTransfer({in}, {sharding}, {xla::ifrt::Shape({4, 3})}),
      StatusIs(absl::StatusCode::kUnimplemented));
  // Size mismatch between inputs and static shapes.
  EXPECT_THAT(
      executor->ExecuteTransfer({in}, {sharding}, {std::nullopt, std::nullopt}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  // Unset entries leave the transfer unaffected.
  TF_ASSERT_OK_AND_ASSIGN(auto results, executor->ExecuteTransfer(
                                            {in}, {sharding}, {std::nullopt}));
  ASSERT_EQ(results.size(), 1);
}

TEST_F(ArenaH2DTransferExecutorTest, FactoryCreate) {
  ArenaH2DTransferExecutorFactory factory;

  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      factory.Create(client_.get(), device_list_, thread_pool_.get()));
  ASSERT_NE(executor, nullptr);

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in = AsHostBuffer<float>({42.0f}, xla::ifrt::Shape({1}));

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer({in}, {sharding}, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out, HostBufferEq(in));
}

}  // namespace
}  // namespace xla_serving

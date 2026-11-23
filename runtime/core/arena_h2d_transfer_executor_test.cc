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
#include <numeric>
#include <optional>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/mock.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/lib/monitoring/cell_reader.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::HasSubstr;
using ::testing::NiceMock;
using ::testing::Return;
using ::tsl::monitoring::testing::CellReader;
using StaticShape = H2DTransferExecutor::StaticShape;

constexpr char kDynamicDmaBytesCounter[] =
    "/xla_serving/input_transfer/dynamic_dma_bytes";

bool IsBackedByCommonPjRtClient(xla::ifrt::Client* client) {
  xla::ifrt::PjRtCompatibleClient* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  return pjrt_client != nullptr && dynamic_cast<xla::CommonPjRtClient*>(
                                       pjrt_client->pjrt_client()) != nullptr;
}

bool SupportsArenaTransfer(xla::ifrt::Client* client) {
  if (client == nullptr || client->platform_id() == xla::CpuId()) {
    return false;
  }
  xla::ifrt::PjRtCompatibleClient* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (pjrt_client == nullptr) {
    return false;
  }
  xla::CommonPjRtClient* common_client =
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
        device_list_, xla::ifrt::test_util::GetDevices(client_.get(), {0, 1}));
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
      std::unique_ptr<H2DTransferExecutor> executor,
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
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  std::vector<HostBuffer> inputs = {};
  std::vector<xla::ifrt::ShardingRef> shardings = {};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  EXPECT_TRUE(results.empty());
}

TEST_F(ArenaH2DTransferExecutorTest, ArenaTransferSingleDevice) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer in0 =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({2, 2}));
  HostBuffer in2 =
      AsHostBuffer<tsl::bfloat16, float>({5.0f, 6.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {in0, in1, in2};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding,
                                                   sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 3);

  for (int i = 0; i < 3; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest, ArenaTransferStridedSingleDevice) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2}), byte_strides, owner);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> results,
                          executor->ExecuteTransfer({strided_in}, {sharding},
                                                    /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  HostBuffer expected = AsHostBuffer<int32_t>({10, 30}, xla::ifrt::Shape({2}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(ArenaH2DTransferExecutorTest, ArenaTransferTransposed2D) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{1, 4, 2, 5, 3, 6});
  absl::InlinedVector<int64_t, 4> byte_strides = {4, 8};
  HostBuffer transposed_in(owner->data(),
                           xla::ifrt::DType(xla::ifrt::DType::kS32),
                           xla::ifrt::Shape({2, 3}), byte_strides, owner);

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> results,
                          executor->ExecuteTransfer({transposed_in}, {sharding},
                                                    /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  HostBuffer expected =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6}, xla::ifrt::Shape({2, 3}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(ArenaH2DTransferExecutorTest, MultiDeviceSingleDeviceSharding) {
  ASSERT_GE(device_list_->devices().size(), 2);

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::ShardingRef sharding0 = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef sharding1 = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[1], xla::ifrt::MemoryKind());

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 = AsHostBuffer<int32_t>({10, 20}, xla::ifrt::Shape({2}));
  HostBuffer in2 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));
  HostBuffer in3 = AsHostBuffer<int32_t>({30, 40}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {in0, in1, in2, in3};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding0, sharding1,
                                                   sharding0, sharding1};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 4);

  for (int i = 0; i < 4; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest,
       ByteStridesRankMismatchReturnsInvalidArgument) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());

  auto owner =
      std::make_shared<std::vector<int32_t>>(std::vector<int32_t>{10, 20});
  absl::InlinedVector<int64_t, 4> mismatched_strides = {4, 4};
  HostBuffer bad_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                    xla::ifrt::Shape({2}), mismatched_strides, owner);

  EXPECT_THAT(
      executor->ExecuteTransfer({bad_in}, {sharding}, /*static_shapes=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ArenaH2DTransferExecutorTest, StringTensorsRouteToDefaultPath) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer str_in(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                    xla::ifrt::Shape({2}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({str_in}, {sharding}, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out, HostBufferEq(str_in));
}

TEST_F(ArenaH2DTransferExecutorTest, PackedSubByteDtypeRoutesToDefaultPath) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer float_in =
      AsHostBuffer<float>({10.0f, 20.0f}, xla::ifrt::Shape({2}));
  HostBuffer int4_in =
      AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {float_in, int4_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
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

TEST_F(ArenaH2DTransferExecutorTest, ZeroElementTensorSupport) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer empty_in = AsHostBuffer<float>({}, xla::ifrt::Shape({0}));
  HostBuffer non_empty_in =
      AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {empty_in, non_empty_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({0}));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out1,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[1], shardings[1], device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out1, HostBufferEq(non_empty_in));
}

TEST_F(ArenaH2DTransferExecutorTest, HloShardingSingleDeviceSupport) {
  ASSERT_GE(device_list_->devices().size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::Device* dev0 = device_list_->devices()[0];
  xla::ifrt::Device* dev1 = device_list_->devices()[1];

  xla::ifrt::ShardingRef sharding_dev0 = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(),
      xla::HloSharding::SingleDevice(dev0->Id().value()));
  xla::ifrt::ShardingRef sharding_dev1 = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(),
      xla::HloSharding::SingleDevice(dev1->Id().value()));

  HostBuffer in0 = AsHostBuffer<int32_t>({100, 200}, xla::ifrt::Shape({2}));
  HostBuffer in1 = AsHostBuffer<float>({1.5f, 2.5f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding_dev0,
                                                   sharding_dev1};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);

  EXPECT_THAT(results[0]->sharding().devices()->devices(), ElementsAre(dev0));
  EXPECT_THAT(results[1]->sharding().devices()->devices(), ElementsAre(dev1));

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest,
       NonSingleDeviceHloShardingRoutesToDefaultPath) {
  ASSERT_GE(device_list_->devices().size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::ShardingRef replicated_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::Replicate());
  xla::ifrt::ShardingRef tiled_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::IotaTile({2}));

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({4}));
  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {replicated_sharding,
                                                   tiled_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
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

TEST_F(ArenaH2DTransferExecutorTest,
       UnsupportedShardingFailsFastWithUnimplemented) {
  ASSERT_GE(device_list_->devices().size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  xla::ifrt::ShardingRef single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[0],
                                              xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef manual_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::Manual());
  xla::ifrt::ShardingRef opaque_sharding = xla::ifrt::OpaqueSharding::Create(
      device_list_, xla::ifrt::MemoryKind());

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));

  EXPECT_THAT(executor->ExecuteTransfer(
                  {in0, in1}, {single_device_sharding, manual_sharding},
                  /*static_shapes=*/{}),
              StatusIs(absl::StatusCode::kUnimplemented));
  EXPECT_THAT(executor->ExecuteTransfer(
                  {in0, in1}, {single_device_sharding, opaque_sharding},
                  /*static_shapes=*/{}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

// Reads back the first `num_rows` rows of `array`.
absl::StatusOr<HostBuffer> ReadBackRows(xla::ifrt::Client& client,
                                        xla::ifrt::Array& array,
                                        const xla::ifrt::ShardingRef& sharding,
                                        const xla::ifrt::DeviceListRef& devices,
                                        tsl::thread::ThreadPool& thread_pool,
                                        int64_t num_rows) {
  TF_ASSIGN_OR_RETURN(HostBuffer out,
                      sharding_internal::MakeHostBufferFromArray(
                          client, array, sharding, devices, thread_pool)
                          .Await());
  return out.SliceAlongFirstDimension(0, num_rows);
}

TEST_F(ArenaH2DTransferExecutorTest, SupportsStaticShapes) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));

  EXPECT_TRUE(executor->SupportsStaticShapes());
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaAllocatesStaticShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  CellReader<int64_t> dynamic_dma_bytes(kDynamicDmaBytesCounter);

  // 12 elements for a static shape of 2048 elements, between two regular
  // inputs in the same arena.
  std::vector<float> values(12);
  std::iota(values.begin(), values.end(), 1.0f);
  HostBuffer in = AsHostBuffer<float>(values, xla::ifrt::Shape({12}));
  HostBuffer before =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({2, 2}));
  HostBuffer after =
      AsHostBuffer<float>({5.0f, 6.0f, 7.0f}, xla::ifrt::Shape({3}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(
          {before, in, after}, {sharding, sharding, sharding},
          {std::nullopt, StaticShape{.shape = xla::ifrt::Shape({2048})},
           std::nullopt}));
  ASSERT_EQ(results.size(), 3);

  // The array has the static shape. Its first 12 elements hold the data.
  EXPECT_EQ(results[1]->shape(), xla::ifrt::Shape({2048}));
  EXPECT_EQ(results[1]->dtype(), in.dtype());
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows,
      ReadBackRows(*client_, *results[1], sharding, device_list_, *thread_pool_,
                   /*num_rows=*/12));
  EXPECT_THAT(rows, HostBufferEq(in));

  // The regular inputs around it are transferred as before.
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer before_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(before_out, HostBufferEq(before));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer after_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[2], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(after_out, HostBufferEq(after));

  // The slice is sized for 2048 elements and 12 elements are copied. A device
  // may round the size of the slice up to whole tiles and the size of the copy
  // up to whole DMA granules.
  const int64_t allocated_bytes = dynamic_dma_bytes.Delta("allocated");
  const int64_t transferred_bytes = dynamic_dma_bytes.Delta("transferred");
  EXPECT_GE(allocated_bytes, 2048 * sizeof(float));
  EXPECT_GE(transferred_bytes, 12 * sizeof(float));
  EXPECT_LT(transferred_bytes, allocated_bytes);
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaWithNoDataRows) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  CellReader<int64_t> dynamic_dma_bytes(kDynamicDmaBytesCounter);

  HostBuffer in = AsHostBuffer<float>({}, xla::ifrt::Shape({0}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({8})}}));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({8}));
  // The buffer can be read although nothing was written to it.
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_EQ(out.shape(), xla::ifrt::Shape({8}));
  EXPECT_GE(dynamic_dma_bytes.Delta("allocated"), 8 * sizeof(float));
  EXPECT_EQ(dynamic_dma_bytes.Delta("transferred"), 0);
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaWithStridedInput) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());

  // Every other element of the storage.
  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2}), byte_strides, owner);

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({strided_in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({4})}}));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({4}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows,
      ReadBackRows(*client_, *results[0], sharding, device_list_, *thread_pool_,
                   /*num_rows=*/2));
  EXPECT_THAT(rows, HostBufferEq(AsHostBuffer<int32_t>({10, 30},
                                                       xla::ifrt::Shape({2}))));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaOnTwoDevices) {
  ASSERT_GE(device_list_->devices().size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::Device* dev0 = device_list_->devices()[0];
  xla::ifrt::Device* dev1 = device_list_->devices()[1];
  // The whole array is on one device of the two-device list.
  xla::ifrt::ShardingRef sharding_dev0 =
      xla::ifrt::SingleDeviceSharding::Create(dev0, xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef sharding_dev1 = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(),
      xla::HloSharding::SingleDevice(dev1->Id().value()));

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 = AsHostBuffer<int32_t>({100}, xla::ifrt::Shape({1}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in0, in1}, {sharding_dev0, sharding_dev1},
                                {StaticShape{.shape = xla::ifrt::Shape({6})},
                                 StaticShape{.shape = xla::ifrt::Shape({4})}}));
  ASSERT_EQ(results.size(), 2);

  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({6}));
  EXPECT_THAT(results[0]->sharding().devices()->devices(), ElementsAre(dev0));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows0,
      ReadBackRows(*client_, *results[0], sharding_dev0, device_list_,
                   *thread_pool_, /*num_rows=*/2));
  EXPECT_THAT(rows0, HostBufferEq(in0));

  EXPECT_EQ(results[1]->shape(), xla::ifrt::Shape({4}));
  EXPECT_THAT(results[1]->sharding().devices()->devices(), ElementsAre(dev1));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows1,
      ReadBackRows(*client_, *results[1], sharding_dev1, device_list_,
                   *thread_pool_, /*num_rows=*/1));
  EXPECT_THAT(rows1, HostBufferEq(in1));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaWithOneDeviceList) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  // A replicated sharding over one device puts the whole array on it.
  xla::ifrt::Device* device = device_list_->devices()[0];
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef one_device,
                          client_->MakeDeviceList({device}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
      one_device, xla::ifrt::MemoryKind(), xla::HloSharding::Replicate());

  HostBuffer in = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({6})}}));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({6}));
  EXPECT_THAT(results[0]->sharding().devices()->devices(), ElementsAre(device));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows, ReadBackRows(*client_, *results[0], sharding, one_device,
                                    *thread_pool_, /*num_rows=*/2));
  EXPECT_THAT(rows, HostBufferEq(in));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaWithRank2Input) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());

  // Three rows for a static shape of 512 rows.
  std::vector<float> values(3 * 4);
  std::iota(values.begin(), values.end(), 1.0f);
  HostBuffer in = AsHostBuffer<float>(values, xla::ifrt::Shape({3, 4}));

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> results =
      executor->ExecuteTransfer(
          {in}, {sharding}, {StaticShape{.shape = xla::ifrt::Shape({512, 4})}});

  if (client_->platform_id() == xla::TpuId()) {
    // The layout that TPU uses for f32[512,4] by default has dimension 1 as
    // the major-most dimension, so the rows of the input are not the front of
    // the buffer.
    EXPECT_THAT(results, StatusIs(absl::StatusCode::kUnimplemented,
                                  HasSubstr("major-most dimension")));
    return;
  }
  ASSERT_THAT(results, IsOk());
  ASSERT_EQ(results->size(), 1);
  EXPECT_EQ((*results)[0]->shape(), xla::ifrt::Shape({512, 4}));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows,
      ReadBackRows(*client_, *(*results)[0], sharding, device_list_,
                   *thread_pool_, /*num_rows=*/3));
  EXPECT_THAT(rows, HostBufferEq(in));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaArrayHasLayoutOfStaticShape) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());
  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));

  // Without a layout, the array has the default layout of the device.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({8})}}));
  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<const xla::PjRtLayout> layout,
                          results[0]->pjrt_layout());
  EXPECT_EQ(layout, nullptr);

  // With a layout, the array is created with it.
  TF_ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<const xla::PjRtLayout> static_layout,
      client_->GetDefaultPjRtLayout(in.dtype(), {8}, device,
                                    xla::ifrt::MemoryKind()));
  ASSERT_NE(static_layout, nullptr);
  TF_ASSERT_OK_AND_ASSIGN(
      results,
      executor->ExecuteTransfer({in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({8}),
                                             .layout = static_layout}}));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({8}));
  TF_ASSERT_OK_AND_ASSIGN(layout, results[0]->pjrt_layout());
  ASSERT_NE(layout, nullptr);
  EXPECT_EQ(*layout, *static_layout);
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rows,
      ReadBackRows(*client_, *results[0], sharding, device_list_, *thread_pool_,
                   /*num_rows=*/3));
  EXPECT_THAT(rows, HostBufferEq(in));
}

TEST_F(ArenaH2DTransferExecutorTest, StaticShapeEqualToInputShapeIsRegular) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  CellReader<int64_t> dynamic_dma_bytes(kDynamicDmaBytesCounter);

  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({3})}}));
  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out, HostBufferEq(in));
  EXPECT_EQ(dynamic_dma_bytes.Delta("allocated"), 0);
  EXPECT_EQ(dynamic_dma_bytes.Delta("transferred"), 0);
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaRejectsMultiDeviceSharding) {
  ASSERT_GE(device_list_->devices().size(), 2);
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef replicated = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::Replicate());

  // The default path, which takes a regular input with this sharding, cannot
  // transfer an input with fewer rows than its static shape.
  HostBuffer in = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  EXPECT_THAT(
      executor->ExecuteTransfer({in}, {replicated},
                                {StaticShape{.shape = xla::ifrt::Shape({6})}}),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("single-device")));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaRejectsUnsupportedDtypes) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());

  // The default path, which takes regular inputs of these types, cannot
  // transfer an input with fewer rows than its static shape.
  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer str_in(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                    xla::ifrt::Shape({2}));
  EXPECT_THAT(
      executor->ExecuteTransfer({str_in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({4})}}),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("dtype")));

  HostBuffer int4_in =
      AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2}));
  EXPECT_THAT(
      executor->ExecuteTransfer({int4_in}, {sharding},
                                {StaticShape{.shape = xla::ifrt::Shape({4})}}),
      StatusIs(absl::StatusCode::kUnimplemented, HasSubstr("dtype")));
}

TEST_F(ArenaH2DTransferExecutorTest, DynamicDmaRejectsInvalidStaticShapes) {
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), thread_pool_.get()));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({1, 3}));

  // No rows left for the input, another inner dimension, another rank.
  for (const xla::ifrt::Shape& static_shape :
       {xla::ifrt::Shape({0, 3}), xla::ifrt::Shape({4, 2}),
        xla::ifrt::Shape({12})}) {
    EXPECT_THAT(executor->ExecuteTransfer({in}, {sharding},
                                          {StaticShape{.shape = static_shape}}),
                StatusIs(absl::StatusCode::kInvalidArgument,
                         HasSubstr("not a dimension-0 prefix")))
        << static_shape;
  }
  // One entry too many.
  EXPECT_THAT(
      executor->ExecuteTransfer({in}, {sharding}, {std::nullopt, std::nullopt}),
      StatusIs(absl::StatusCode::kInvalidArgument));
  // An unset entry is a regular input.
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer({in}, {sharding}, {std::nullopt}));
  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({1, 3}));
}

TEST_F(ArenaH2DTransferExecutorTest, FactoryCreate) {
  ArenaH2DTransferExecutorFactory factory;

  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      factory.Create(client_.get(), device_list_, thread_pool_.get()));
  ASSERT_NE(executor, nullptr);

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in = AsHostBuffer<float>({42.0f}, xla::ifrt::Shape({1}));

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
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

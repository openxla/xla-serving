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
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/bfloat16.h"  // from @tsl
#include "tsl/platform/ml_dtypes.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;

bool IsBackedByCommonPjRtClient(xla::ifrt::Client* client) {
  auto* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  return pjrt_client != nullptr && dynamic_cast<xla::CommonPjRtClient*>(
                                       pjrt_client->pjrt_client()) != nullptr;
}

bool SupportsDeviceAddressAlignment(xla::ifrt::Client* client) {
  auto* pjrt_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (pjrt_client == nullptr) {
    return false;
  }
  auto* common_client =
      dynamic_cast<xla::CommonPjRtClient*>(pjrt_client->pjrt_client());
  return common_client != nullptr && common_client->raw_client() != nullptr &&
         common_client->raw_client()->GetDeviceAddressAlignment().ok();
}

class ArenaH2DTransferExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    if (!IsBackedByCommonPjRtClient(client_.get())) {
      GTEST_SKIP() << "ArenaH2DTransferExecutor requires CommonPjRtClient.";
    }
    if (!SupportsDeviceAddressAlignment(client_.get())) {
      GTEST_SKIP()
          << "GetDeviceAddressAlignment is not implemented on this backend.";
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

TEST(ArenaH2DTransferExecutorClientTest,
     NonCommonPjRtClientReturnsInvalidArgument) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  if (IsBackedByCommonPjRtClient(client.get())) {
    GTEST_SKIP() << "Client is backed by CommonPjRtClient.";
  }
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);
  ArenaH2DTransferExecutor::Options options;
  EXPECT_THAT(ArenaH2DTransferExecutor::Create(client.get(), device_list,
                                               &thread_pool, options),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ArenaH2DTransferExecutorClientTest,
     UnimplementedDeviceAddressAlignmentReturnsUnimplemented) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  if (!IsBackedByCommonPjRtClient(client.get()) ||
      SupportsDeviceAddressAlignment(client.get())) {
    GTEST_SKIP() << "Requires a CommonPjRtClient without "
                    "GetDeviceAddressAlignment support.";
  }
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/2);
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor, ArenaH2DTransferExecutor::Create(client.get(), device_list,
                                                      &thread_pool, options));
  HostBuffer in = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list->devices()[0], xla::ifrt::MemoryKind());
  EXPECT_THAT(executor->ExecuteTransfer({in}, {sharding}),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(ArenaH2DTransferExecutorTest, SizeMismatchReturnsInvalidArgument) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  HostBuffer input0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer input1 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));
  std::vector<HostBuffer> inputs = {input0, input1};

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};  // size 1 vs 2

  EXPECT_THAT(executor->ExecuteTransfer(inputs, shardings),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(ArenaH2DTransferExecutorTest, NullThreadPoolReturnsInvalidArgument)
#if ABSL_HAVE_ATTRIBUTE(no_sanitize)
__attribute__((no_sanitize("nullability")))
#endif
{
  ArenaH2DTransferExecutor::Options options;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
  EXPECT_THAT(
      ArenaH2DTransferExecutor::Create(client_.get(), device_list_,
                                       /*thread_pool=*/nullptr, options),
      StatusIs(absl::StatusCode::kInvalidArgument));
#pragma GCC diagnostic pop
}

TEST_F(ArenaH2DTransferExecutorTest, EmptyInputsSucceeds) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  std::vector<HostBuffer> inputs = {};
  std::vector<xla::ifrt::ShardingRef> shardings = {};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  EXPECT_TRUE(results.empty());
}

TEST_F(ArenaH2DTransferExecutorTest, SmallTensorsCoalescedSingleDevice) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

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

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), 3);

  for (int i = 0; i < 3; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        auto old_array,
        sharding_internal::MakeArrayFromHostBuffer(
            *client_, inputs[i], device_list_, shardings[i], *thread_pool_));
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer old_out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *old_array, shardings[i], device_list_, *thread_pool_)
            .Await());

    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer new_out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());

    EXPECT_THAT(new_out, HostBufferEq(old_out));
    EXPECT_THAT(new_out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest, LargeTensorsIndividualSingleDevice) {
  ArenaH2DTransferExecutor::Options options;
  options.small_tensor_threshold_in_bytes = 64 * 1024;  // 64 KB
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  // 20,000 floats = 80,000 bytes (> 64 KB)
  std::vector<float> large_data0(20000);
  std::iota(large_data0.begin(), large_data0.end(), 1.0f);
  HostBuffer in0 = AsHostBuffer<float>(large_data0, xla::ifrt::Shape({20000}));

  // 25,000 int32s = 100,000 bytes (> 64 KB)
  std::vector<int32_t> large_data1(25000);
  std::iota(large_data1.begin(), large_data1.end(), 100);
  HostBuffer in1 =
      AsHostBuffer<int32_t>(large_data1, xla::ifrt::Shape({250, 100}));

  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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

TEST_F(ArenaH2DTransferExecutorTest, MixedSmallAndLargeTensorsSingleDevice) {
  ArenaH2DTransferExecutor::Options options;
  // Set threshold to 32 bytes so we can easily mix small (<= 32B) and large
  // (> 32B) tensors and verify the "Large then Small" arena layout and output
  // slot mapping.
  options.small_tensor_threshold_in_bytes = 32;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  // Small (8 bytes)
  HostBuffer small0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  // Large (64 bytes)
  std::vector<int32_t> large_vec0(16);
  std::iota(large_vec0.begin(), large_vec0.end(), 10);
  HostBuffer large0 =
      AsHostBuffer<int32_t>(large_vec0, xla::ifrt::Shape({4, 4}));
  // Small (16 bytes)
  HostBuffer small1 =
      AsHostBuffer<int32_t>({100, 200, 300, 400}, xla::ifrt::Shape({4}));
  // Large (128 bytes)
  std::vector<float> large_vec1(32);
  std::iota(large_vec1.begin(), large_vec1.end(), 0.5f);
  HostBuffer large1 = AsHostBuffer<float>(large_vec1, xla::ifrt::Shape({8, 4}));
  // Small (4 bytes)
  HostBuffer small2 =
      AsHostBuffer<tsl::bfloat16, float>({7.0f, 8.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {small0, large0, small1, large1, small2};
  std::vector<xla::ifrt::ShardingRef> shardings(inputs.size(), sharding);

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), inputs.size());

  for (size_t i = 0; i < inputs.size(); ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest,
       ThresholdZeroDisablesSmallTensorCoalescing) {
  ArenaH2DTransferExecutor::Options options;
  options.small_tensor_threshold_in_bytes = 0;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 = AsHostBuffer<int32_t>({10, 20, 30}, xla::ifrt::Shape({3}));

  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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

TEST_F(ArenaH2DTransferExecutorTest, StringTensorsRouteToDefaultPath) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer str_in(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                    xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {str_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out, HostBufferEq(str_in));
}

TEST_F(ArenaH2DTransferExecutorTest, PackedSubByteDtypeRoutesToDefaultPath) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer float_in =
      AsHostBuffer<float>({10.0f, 20.0f}, xla::ifrt::Shape({2}));
  HostBuffer int4_in =
      AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {float_in, int4_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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

TEST_F(ArenaH2DTransferExecutorTest, ArenaTransferStridedSingleDevice) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {strided_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());
  HostBuffer expected = AsHostBuffer<int32_t>({10, 30}, xla::ifrt::Shape({2}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(ArenaH2DTransferExecutorTest, ArenaTransferTransposed2D) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{1, 4, 2, 5, 3, 6});
  absl::InlinedVector<int64_t, 4> byte_strides = {4, 8};
  HostBuffer transposed_in(owner->data(),
                           xla::ifrt::DType(xla::ifrt::DType::kS32),
                           xla::ifrt::Shape({2, 3}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {transposed_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());
  HostBuffer expected =
      AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6}, xla::ifrt::Shape({2, 3}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(ArenaH2DTransferExecutorTest, ZeroElementTensorSupport) {
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer empty_in = AsHostBuffer<float>({}, xla::ifrt::Shape({0}));
  HostBuffer non_empty_in =
      AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {empty_in, non_empty_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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
  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

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

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
  ASSERT_EQ(results.size(), 2);

  EXPECT_THAT(results[0]->sharding().devices()->devices(),
              ::testing::ElementsAre(dev0));
  EXPECT_THAT(results[1]->sharding().devices()->devices(),
              ::testing::ElementsAre(dev1));

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        HostBuffer out,
        sharding_internal::MakeHostBufferFromArray(
            *client_, *results[i], shardings[i], device_list_, *thread_pool_)
            .Await());
    EXPECT_THAT(out, HostBufferEq(inputs[i]));
  }
}

TEST_F(ArenaH2DTransferExecutorTest, MultiDeviceSingleDeviceSharding) {
  ASSERT_GE(device_list_->devices().size(), 2);

  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(auto executor, ArenaH2DTransferExecutor::Create(
                                             client_.get(), device_list_,
                                             thread_pool_.get(), options));

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

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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

TEST_F(ArenaH2DTransferExecutorTest, UnsupportedHloShardingReturnsError) {
  ASSERT_GE(device_list_->devices().size(), 2);

  ArenaH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      ArenaH2DTransferExecutor::Create(client_.get(), device_list_,
                                       thread_pool_.get(), options));

  xla::ifrt::ShardingRef manual_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::Manual());

  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f}, xla::ifrt::Shape({4}));
  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {manual_sharding};

  EXPECT_THAT(executor->ExecuteTransfer(inputs, shardings),
              StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(ArenaH2DTransferExecutorTest, FactoryCreate) {
  ArenaH2DTransferExecutor::Options options;
  options.small_tensor_threshold_in_bytes = 32 * 1024;
  ArenaH2DTransferExecutorFactory factory(options);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      factory.Create(client_.get(), device_list_, thread_pool_.get()));
  ASSERT_NE(executor, nullptr);

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in = AsHostBuffer<float>({42.0f}, xla::ifrt::Shape({1}));

  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(auto results,
                          executor->ExecuteTransfer(inputs, shardings));
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

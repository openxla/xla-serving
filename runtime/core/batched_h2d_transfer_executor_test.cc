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
#include "runtime/core/batched_h2d_transfer_executor.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/attributes.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
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

class BatchedH2DTransferExecutorTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
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

TEST_F(BatchedH2DTransferExecutorTest, SizeMismatchReturnsInvalidArgument) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  HostBuffer input0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer input1 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));
  std::vector<HostBuffer> inputs = {input0, input1};

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};  // size 1 vs 2

  EXPECT_THAT(
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}),
      StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST_F(BatchedH2DTransferExecutorTest, NullThreadPoolReturnsInvalidArgument)
#if ABSL_HAVE_ATTRIBUTE(no_sanitize)
__attribute__((no_sanitize("nullability")))
#endif
{
  BatchedH2DTransferExecutor::Options options;
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnonnull"
  EXPECT_THAT(
      BatchedH2DTransferExecutor::Create(client_.get(), device_list_,
                                         /*thread_pool=*/nullptr, options),
      StatusIs(absl::StatusCode::kInvalidArgument));
#pragma GCC diagnostic pop
}

TEST_F(BatchedH2DTransferExecutorTest, EmptyInputsSucceeds) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  std::vector<HostBuffer> inputs = {};
  std::vector<xla::ifrt::ShardingRef> shardings = {};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  EXPECT_TRUE(results.empty());
}

TEST_F(BatchedH2DTransferExecutorTest,
       BatchedTransferSingleDevice) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

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

  // 1. Run new batched transfer.
  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 3);

  // 2. Validate side-by-side against old/default method and ground truth.
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


TEST_F(BatchedH2DTransferExecutorTest, StringTensorsRouteToDefaultPath) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  tsl::tstring str_data[2] = {"hello", "world"};
  HostBuffer str_in(str_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                    xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {str_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  // String tensors bypass batched PJRT literal transfer and route to
  // sharding_internal::MakeArrayFromHostBuffer.
  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], sharding, device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out, HostBufferEq(str_in));
}

TEST_F(BatchedH2DTransferExecutorTest, PackedSubByteDtypeRoutesToDefaultPath) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

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

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferStridedSingleDevice) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  // Creates a buffer with 4 ints, but strided to only select the first and
  // third (indices 0 and 2 -> {10, 30}).
  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {strided_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());
  HostBuffer expected =
      AsHostBuffer<int32_t>({10, 30}, xla::ifrt::Shape({2}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferStridedReplicated) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::HloSharding replicated_sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), replicated_sharding);

  // Creates a strided buffer: elements 10, 30.
  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({2}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {strided_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());
  HostBuffer expected =
      AsHostBuffer<int32_t>({10, 30}, xla::ifrt::Shape({2}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferStridedTiled) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  // Tiled 1D sharding across 2 devices.
  xla::HloSharding tiled_sharding = xla::HloSharding::IotaTile({2});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding);

  // Underlying data: 8 elements, stride 8 selects indices 0, 2, 4, 6 ->
  // {10, 20, 30, 40}.
  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{10, 999, 20, 999, 30, 999, 40, 999});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_in(owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
                        xla::ifrt::Shape({4}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {strided_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());
  HostBuffer expected =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({4}));
  EXPECT_THAT(out, HostBufferEq(expected));
}

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferTransposed2D) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  // 3x2 matrix in row-major memory: {{1, 4}, {2, 5}, {3, 6}}, viewed as a
  // transposed 2x3 matrix {{1, 2, 3}, {4, 5, 6}} with byte_strides = {4, 8}.
  auto owner = std::make_shared<std::vector<int32_t>>(
      std::vector<int32_t>{1, 4, 2, 5, 3, 6});
  absl::InlinedVector<int64_t, 4> byte_strides = {4, 8};
  HostBuffer transposed_in(
      owner->data(), xla::ifrt::DType(xla::ifrt::DType::kS32),
      xla::ifrt::Shape({2, 3}), byte_strides, owner);

  std::vector<HostBuffer> inputs = {transposed_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
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

TEST_F(BatchedH2DTransferExecutorTest, ZeroElementTensorSupport) {
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::Device* device = device_list_->devices()[0];
  xla::ifrt::ShardingRef sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, xla::ifrt::MemoryKind());

  HostBuffer empty_in = AsHostBuffer<float>({}, xla::ifrt::Shape({0}));
  HostBuffer non_empty_in =
      AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {empty_in, non_empty_in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding, sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->shape(), xla::ifrt::Shape({0}));

  // Verify the non-empty tensor batched alongside the 0-element tensor
  // round-trips correctly.
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer out1,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[1], shardings[1], device_list_, *thread_pool_)
          .Await());
  EXPECT_THAT(out1, HostBufferEq(non_empty_in));
}

TEST_F(BatchedH2DTransferExecutorTest, HloShardingSingleDeviceSupport) {
  ASSERT_GE(device_list_->devices().size(), 2);
  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

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
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
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

TEST_F(BatchedH2DTransferExecutorTest, MultiDeviceSingleDeviceSharding) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

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
      auto results,
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

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferReplicated) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::HloSharding replicated_sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), replicated_sharding);

  HostBuffer in0 =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({2, 2}));
  HostBuffer in2 =
      AsHostBuffer<tsl::bfloat16, float>({5.0f, 6.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {in0, in1, in2};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding, ifrt_sharding,
                                                   ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
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

TEST_F(BatchedH2DTransferExecutorTest,
       BatchedTransferMixedSingleAndReplicated) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::ShardingRef sharding_dev0 =
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[0],
                                              xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef sharding_dev1 =
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[1],
                                              xla::ifrt::MemoryKind());
  xla::HloSharding replicated_sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef sharding_replicated = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), replicated_sharding);

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({100, 200, 300, 400}, xla::ifrt::Shape({4}));
  HostBuffer in2 = AsHostBuffer<float>({3.0f, 4.0f}, xla::ifrt::Shape({2}));
  HostBuffer in3 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({2, 2}));

  std::vector<HostBuffer> inputs = {in0, in1, in2, in3};
  std::vector<xla::ifrt::ShardingRef> shardings = {
      sharding_dev0, sharding_replicated, sharding_dev1, sharding_replicated};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
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

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferTiled1D) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::HloSharding tiled_sharding = xla::HloSharding::IotaTile({2});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding);

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f, 3.0f, 4.0f},
                                       xla::ifrt::Shape({4}));
  HostBuffer in1 = AsHostBuffer<int32_t>({10, 20, 30, 40, 50, 60},
                                         xla::ifrt::Shape({6}));
  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding,
                                                   ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef old_array,
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

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferTiled2D) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  // Tiled along axis 0 (rows): shape [4, 2] partitioned 2x1 -> [2, 2] per device.
  xla::HloSharding tiled_sharding_axis0 = xla::HloSharding::IotaTile({2, 1});
  xla::ifrt::ShardingRef sharding_axis0 = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding_axis0);

  // Tiled along axis 1 (cols): shape [2, 4] partitioned 1x2 -> [2, 2] per device.
  xla::HloSharding tiled_sharding_axis1 = xla::HloSharding::IotaTile({1, 2});
  xla::ifrt::ShardingRef sharding_axis1 = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding_axis1);

  HostBuffer in0 = AsHostBuffer<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f},
      xla::ifrt::Shape({4, 2}));
  HostBuffer in1 = AsHostBuffer<int32_t>(
      {10, 20, 30, 40, 50, 60, 70, 80},
      xla::ifrt::Shape({2, 4}));

  std::vector<HostBuffer> inputs = {in0, in1};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding_axis0,
                                                   sharding_axis1};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 2);

  for (int i = 0; i < 2; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef old_array,
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

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferTiled3D) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  // 3D tensor of shape [2, 4, 2] tiled along axis 1 (2 partitions).
  xla::HloSharding tiled_sharding = xla::HloSharding::IotaTile({1, 2, 1});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding);

  std::vector<float> data(16);
  for (int i = 0; i < 16; ++i) {
    data[i] = static_cast<float>(i * 1.5f);
  }
  HostBuffer in = AsHostBuffer<float>(data, xla::ifrt::Shape({2, 4, 2}));

  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef old_array,
      sharding_internal::MakeArrayFromHostBuffer(
          *client_, inputs[0], device_list_, shardings[0], *thread_pool_));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer old_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *old_array, shardings[0], device_list_, *thread_pool_)
          .Await());

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer new_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());

  EXPECT_THAT(new_out, HostBufferEq(old_out));
  EXPECT_THAT(new_out, HostBufferEq(inputs[0]));
}

TEST_F(BatchedH2DTransferExecutorTest, BatchedTransferTiled4D) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  // 4D tensor of shape [2, 2, 4, 2] tiled along axis 2 (2 partitions).
  xla::HloSharding tiled_sharding = xla::HloSharding::IotaTile({1, 1, 2, 1});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding);

  std::vector<int32_t> data(32);
  for (int i = 0; i < 32; ++i) {
    data[i] = i * 10;
  }
  HostBuffer in = AsHostBuffer<int32_t>(data, xla::ifrt::Shape({2, 2, 4, 2}));

  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 1);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef old_array,
      sharding_internal::MakeArrayFromHostBuffer(
          *client_, inputs[0], device_list_, shardings[0], *thread_pool_));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer old_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *old_array, shardings[0], device_list_, *thread_pool_)
          .Await());

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer new_out,
      sharding_internal::MakeHostBufferFromArray(
          *client_, *results[0], shardings[0], device_list_, *thread_pool_)
          .Await());

  EXPECT_THAT(new_out, HostBufferEq(old_out));
  EXPECT_THAT(new_out, HostBufferEq(inputs[0]));
}

TEST_F(BatchedH2DTransferExecutorTest,
       BatchedTransferMixedSingleReplicatedAndTiled) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  xla::ifrt::ShardingRef sharding_dev0 =
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[0],
                                              xla::ifrt::MemoryKind());
  xla::ifrt::ShardingRef sharding_dev1 =
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[1],
                                              xla::ifrt::MemoryKind());
  xla::HloSharding replicated_sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef sharding_replicated = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), replicated_sharding);
  xla::HloSharding tiled_sharding_2d = xla::HloSharding::IotaTile({2, 1});
  xla::ifrt::ShardingRef sharding_tiled_2d = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding_2d);
  xla::HloSharding tiled_sharding_1d = xla::HloSharding::IotaTile({2});
  xla::ifrt::ShardingRef sharding_tiled_1d = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding_1d);

  HostBuffer in0 = AsHostBuffer<float>({1.0f, 2.0f}, xla::ifrt::Shape({2}));
  HostBuffer in1 =
      AsHostBuffer<int32_t>({100, 200, 300, 400}, xla::ifrt::Shape({4}));
  HostBuffer in2 = AsHostBuffer<float>(
      {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f}, xla::ifrt::Shape({2, 3}));
  HostBuffer in3 =
      AsHostBuffer<int32_t>({10, 20, 30, 40}, xla::ifrt::Shape({4}));
  HostBuffer in4 = AsHostBuffer<float>({9.0f, 8.0f}, xla::ifrt::Shape({2}));

  std::vector<HostBuffer> inputs = {in0, in1, in2, in3, in4};
  std::vector<xla::ifrt::ShardingRef> shardings = {
      sharding_dev0, sharding_replicated, sharding_tiled_2d,
      sharding_tiled_1d, sharding_dev1};

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
  ASSERT_EQ(results.size(), 5);

  for (int i = 0; i < 5; ++i) {
    TF_ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef old_array,
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

TEST_F(BatchedH2DTransferExecutorTest, UnevenTiledInputReturnsError) {
  ASSERT_GE(device_list_->devices().size(), 2);

  BatchedH2DTransferExecutor::Options options;
  TF_ASSERT_OK_AND_ASSIGN(
      std::unique_ptr<H2DTransferExecutor> executor,
      BatchedH2DTransferExecutor::Create(
          client_.get(), device_list_, thread_pool_.get(), options));

  // Tensor of size 3 tiled across 2 devices is not evenly splittable.
  xla::HloSharding tiled_sharding = xla::HloSharding::IotaTile({2});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), tiled_sharding);

  HostBuffer in =
      AsHostBuffer<float>({1.0f, 2.0f, 3.0f}, xla::ifrt::Shape({3}));
  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {ifrt_sharding};

  // A size-3 tensor tiled over 2 devices yields unequal shard shapes ([2] vs
  // [1]), which the batching planner rejects with UnimplementedError.
  EXPECT_THAT(
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}),
      StatusIs(absl::StatusCode::kUnimplemented));
}

TEST_F(BatchedH2DTransferExecutorTest, FactoryCreate) {
  BatchedH2DTransferExecutor::Options options;
  BatchedH2DTransferExecutorFactory factory(options);

  TF_ASSERT_OK_AND_ASSIGN(
      auto executor,
      factory.Create(client_.get(), device_list_, thread_pool_.get()));
  ASSERT_NE(executor, nullptr);

  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  HostBuffer in = AsHostBuffer<float>({42.0f}, xla::ifrt::Shape({1}));

  std::vector<HostBuffer> inputs = {in};
  std::vector<xla::ifrt::ShardingRef> shardings = {sharding};

  TF_ASSERT_OK_AND_ASSIGN(
      auto results,
      executor->ExecuteTransfer(inputs, shardings, /*static_shapes=*/{}));
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

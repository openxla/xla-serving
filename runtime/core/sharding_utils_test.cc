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

#include "runtime/core/sharding_utils.h"  // from @xla_serving

#include <algorithm>
#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/notification.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/device_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
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
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/lib/core/status_test_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/ml_dtypes.h"  // from @tsl

namespace xla_serving::sharding_internal {
namespace {

struct ReshardToHostBufferTestParam {
  // split host buffers in natural device order.
  std::vector<HostBuffer> split_buffers;
  HostBuffer expected_out_buffer;
  std::vector<int> device_indices;
  xla::HloSharding sharding;
};

struct HostBufferToArrayTestParam {
  HostBuffer in_buffer;
  std::vector<HostBuffer> expected_out_buffers;
  std::vector<int> device_ids;
  xla::HloSharding sharding;
};

struct HostBuffersDisassembleParam {
  HostBuffer in_buffer;
  std::vector<int> split_partitions;
  absl::StatusOr<std::vector<HostBuffer>> expected_out_buffer;
};

struct HostBuffersAssembleParam {
  std::vector<HostBuffer> in_buffers;
  std::vector<int> concat_nums;
  absl::StatusOr<HostBuffer> expected_out_buffer;
};

using ReshardToHostBufferTest =
    ::testing::TestWithParam<ReshardToHostBufferTestParam>;
using HostBufferToArrayTest =
    ::testing::TestWithParam<HostBufferToArrayTestParam>;

using HostBuffersDisassembleTest =
    ::testing::TestWithParam<HostBuffersDisassembleParam>;

using HostBuffersAssembleTest =
    ::testing::TestWithParam<HostBuffersAssembleParam>;

// Wrapper functions for building sharding specs for a given shape with a
// natural device order.
xla::HloSharding Tile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::IotaTile(dims);
}
xla::HloSharding PartialTile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::PartialTile(xla::TileAssignment(dims));
}
xla::HloSharding Replicate() { return xla::HloSharding::Replicate(); }
xla::HloSharding Maximal(int64_t device_index = 0) {
  return xla::HloSharding::SingleDevice(device_index);
}

TEST_P(HostBuffersDisassembleTest, DisassembledBuffers) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "SplitHostBuffer", kMaxParallelism);
  absl::StatusOr<std::vector<HostBuffer>> out_buffers = DisassembleHostBuffer(
      GetParam().in_buffer, GetParam().split_partitions, thread_pool);
  ASSERT_THAT(out_buffers, absl_testing::StatusIs(
                               GetParam().expected_out_buffer.status().code()));
  if (out_buffers.ok()) {
    for (int i = 0; i < out_buffers->size(); ++i) {
      EXPECT_THAT((*out_buffers)[i],
                  HostBufferEq((*GetParam().expected_out_buffer)[i]));
    }
  }
}

INSTANTIATE_TEST_SUITE_P(
    HostBuffersManipulationTests, HostBuffersDisassembleTest,
    ::testing::ValuesIn<HostBuffersDisassembleParam>({
        {
            .in_buffer = AsHostBuffer<int32_t>(3),
            .split_partitions = {2},
            .expected_out_buffer = absl::InvalidArgumentError("out of range"),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>(3),
            .split_partitions = {1},
            .expected_out_buffer =
                std::vector<HostBuffer>{
                    AsHostBuffer<int32_t>(3),
                },
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               xla::ifrt::Shape({4, 2})),
            // Split on first (batching) dimension.
            .split_partitions = {2, 1},
            .expected_out_buffer =
                std::vector<HostBuffer>{
                    AsHostBuffer<int32_t>({1, 2, 3, 4},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({5, 6, 7, 8},
                                          xla::ifrt::Shape({2, 2})),
                },
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               xla::ifrt::Shape({4, 2})),
            // Split on multiple dimension.
            .split_partitions = {2, 2},
            .expected_out_buffer =
                std::vector<HostBuffer>{
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({5, 7}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({6, 8}, xla::ifrt::Shape({2, 1})),
                },
        },
    }));

TEST_P(HostBuffersAssembleTest, AssembledBuffers) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "SplitHostBuffer", kMaxParallelism);
  absl::StatusOr<HostBuffer> out_buffer = MakeHostBufferFromDisassembledBuffers(
      GetParam().in_buffers, GetParam().concat_nums,
      GetParam().in_buffers[0].dtype(), GetParam().expected_out_buffer->shape(),
      thread_pool);
  ASSERT_THAT(out_buffer, absl_testing::StatusIs(
                              GetParam().expected_out_buffer.status().code()));
  if (out_buffer.ok()) {
    EXPECT_THAT((*out_buffer), HostBufferEq((*GetParam().expected_out_buffer)));
  }
}

INSTANTIATE_TEST_SUITE_P(
    HostBuffersManipulationTests, HostBuffersAssembleTest,
    ::testing::ValuesIn<HostBuffersAssembleParam>({
        {
            .in_buffers =
                std::vector<HostBuffer>{
                    AsHostBuffer<int32_t>({1, 2, 3, 4},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({5, 6, 7, 8},
                                          xla::ifrt::Shape({2, 2})),
                },
            .concat_nums = {2, 1},
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2})),
        },
    }));

TEST_P(ReshardToHostBufferTest, MakeHostBufferFromDeviceArrays) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ifrt::test_util::GetDevices(
                              client.get(), GetParam().device_indices));

  std::vector<xla::ifrt::ArrayRef> split_arrays;
  for (int i = 0; i < GetParam().split_buffers.size(); ++i) {
    const auto& split_buffer = GetParam().split_buffers[i];
    auto single_device_sharding = xla::ifrt::SingleDeviceSharding::Create(
        device_list->devices()[i], xla::ifrt::MemoryKind());
    TF_ASSERT_OK_AND_ASSIGN(
        auto array,
        client->MakeArrayFromHostBuffer(
            split_buffer.base(), split_buffer.dtype(),
            std::move(split_buffer.shape()), std::nullopt,
            std::move(single_device_sharding),
            /*layout=*/nullptr,
            xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/{}));
    split_arrays.push_back(std::move(array));
  }

  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), GetParam().sharding);
  xla::ifrt::ArrayRef assembled_array;

  TF_ASSERT_OK_AND_ASSIGN(
      assembled_array,
      client->AssembleArrayFromSingleDeviceArrays(
          split_arrays[0]->dtype(), GetParam().expected_out_buffer.shape(),
          ifrt_sharding, absl::MakeSpan(split_arrays),
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  TF_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                          MakeHostBufferFromArray(*client, *assembled_array,
                                                  std::move(ifrt_sharding),
                                                  device_list, thread_pool)
                              .Await());

  EXPECT_THAT(GetParam().expected_out_buffer, HostBufferEq(output_buffer));
}

INSTANTIATE_TEST_SUITE_P(
    HloShardingTests, ReshardToHostBufferTest,
    ::testing::ValuesIn<ReshardToHostBufferTestParam>({
        // Maximal
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>(3),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(3),
            .device_indices = {0},
            .sharding = Maximal(0),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>(3),
                    AsHostBuffer<int32_t>(4),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>({4},
                                                         xla::ifrt::Shape({})),
            .device_indices = {0, 1},
            .sharding = Maximal(1),
        },

        // Full replication.
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>(1),
                    AsHostBuffer<int32_t>(1),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>({1},
                                                         xla::ifrt::Shape({})),
            .device_indices = {0, 1},
            .sharding = Replicate(),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3, 1})),
                    AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3, 1})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3, 1})),
            .device_indices = {0, 1},
            .sharding = Replicate(),
        },

        // 1-D sharding
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({2})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                                         xla::ifrt::Shape({4})),
            .device_indices = {0, 1},
            .sharding = Tile({2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<tsl::int4, int>({1, 2}, xla::ifrt::Shape({2})),
                    AsHostBuffer<tsl::int4, int>({3, 4}, xla::ifrt::Shape({2})),
                },
            .expected_out_buffer = AsHostBuffer<tsl::int4, int>(
                {1, 2, 3, 4}, xla::ifrt::Shape({4})),
            .device_indices = {0, 1},
            .sharding = Tile({2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})),
            .device_indices = {0, 1},
            .sharding = Tile({2, 1}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({1, 2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({1, 2, 1})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4}, xla::ifrt::Shape({1, 2, 2})),
            .device_indices = {0, 1},
            .sharding = Tile({1, 1, 2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({5, 6}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({7, 8}, xla::ifrt::Shape({1, 2})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4, 5, 6, 7, 8}, xla::ifrt::Shape({4, 2})),
            .device_indices = {0, 1, 2, 3},
            .sharding = Tile({4, 1}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3, 5, 7},
                                          xla::ifrt::Shape({4, 1})),
                    AsHostBuffer<int32_t>({2, 4, 6, 8},
                                          xla::ifrt::Shape({4, 1})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},

                                      xla::ifrt::Shape({4, 2})),
            .device_indices = {0, 1},
            .sharding = Tile({1, 2}),
        },
        // 2-D sharding
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 5, 6},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({3, 4, 7, 8},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({9, 10, 13, 14},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({11, 12, 15, 16},
                                          xla::ifrt::Shape({2, 2})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},

                xla::ifrt::Shape({4, 4})),
            .device_indices = {0, 1, 2, 3},
            .sharding = Tile({2, 2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 5, 6},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({3, 4, 7, 8},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({9, 10, 13, 14},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({11, 12, 15, 16},
                                          xla::ifrt::Shape({2, 1, 2})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},

                xla::ifrt::Shape({4, 1, 4})),
            .device_indices = {0, 1, 2, 3},
            .sharding = Tile({2, 1, 2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 5, 6},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({3, 4, 7, 8},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({9, 10, 13, 14},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({11, 12, 15, 16},
                                          xla::ifrt::Shape({2, 1, 2})),
                },
            .expected_out_buffer = AsHostBuffer<int32_t>(
                {1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16},

                xla::ifrt::Shape({4, 1, 4})),
            .device_indices = {3, 2, 1, 0},
            .sharding = Tile({2, 1, 2}),
        },
        // Partial replication
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({2, 1})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})),
            .device_indices = {0, 1, 2, 3},
            .sharding = PartialTile({1, 2, 2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})),
            .device_indices = {0, 1, 2, 3},
            .sharding = PartialTile({2, 1, 2}),
        },
        {
            .split_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .expected_out_buffer =
                AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})),
            .device_indices = {3, 2, 1, 0},
            .sharding = PartialTile({2, 1, 2}),
        },
    }));

TEST_P(HostBufferToArrayTest, MakeArrayFromHostBuffer) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  auto input_buffer = GetParam().in_buffer;

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::DeviceListRef device_list,
      xla::ifrt::test_util::GetDevices(client.get(),
                                       absl::MakeSpan(GetParam().device_ids)));

  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), GetParam().sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      MakeArrayFromHostBuffer(*client, input_buffer, device_list,
                              std::move(ifrt_sharding), thread_pool));

  TF_ASSERT_OK_AND_ASSIGN(
      auto disassembled_arrays,
      assembled_array->DisassembleIntoSingleDeviceArrays(
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_EQ(disassembled_arrays.size(), GetParam().expected_out_buffers.size());

  for (int i = 0; i < disassembled_arrays.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Array ", i, " of ", disassembled_arrays.size()));
    auto disassembled_array = disassembled_arrays[i];
    auto expected_out_buffer = GetParam().expected_out_buffers[i];
    ASSERT_EQ(disassembled_array->shape(), expected_out_buffer.shape());
    HostBuffer out_buffer(expected_out_buffer.dtype(),
                          expected_out_buffer.shape());
    TF_ASSERT_OK(
        disassembled_array
            ->CopyToHostBuffer(out_buffer.base(), std::nullopt,
                               xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
            .Await());
    EXPECT_THAT(expected_out_buffer, HostBufferEq(out_buffer));
  }
}

INSTANTIATE_TEST_SUITE_P(
    HostBufferToArrayTests, HostBufferToArrayTest,
    ::testing::ValuesIn<HostBufferToArrayTestParam>({
        // Single device
        {
            .in_buffer = AsHostBuffer<int32_t>(1),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>(1),
                },
            .device_ids = {0},
            .sharding = Replicate(),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>(2),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>(2),
                },
            .device_ids = {0},
            .sharding = Maximal(0),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>(3),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>(3),
                },
            .device_ids = {0, 1},
            .sharding = Maximal(1),
        },
        // Full replication.
        {
            .in_buffer = AsHostBuffer<int32_t>(1),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>(1),
                    AsHostBuffer<int32_t>(1),
                },
            .device_ids = {0, 1},
            .sharding = Replicate(),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3},
                                               xla::ifrt::Shape({3, 1})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3, 1})),
                    AsHostBuffer<int32_t>({1, 2, 3}, xla::ifrt::Shape({3, 1})),
                },
            .device_ids = {0, 1},
            .sharding = Replicate(),
        },
        // 1-D sharding
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                               xla::ifrt::Shape({4})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({2})),
                },
            .device_ids = {0, 1},
            .sharding = Tile({2}),
        },
        {
            .in_buffer = AsHostBuffer<tsl::uint4, int>({1, 2, 3, 4},
                                                       xla::ifrt::Shape({4})),
            .expected_out_buffers =
                {
                    AsHostBuffer<tsl::uint4, int>({1, 2},
                                                  xla::ifrt::Shape({2})),
                    AsHostBuffer<tsl::uint4, int>({3, 4},
                                                  xla::ifrt::Shape({2})),
                },
            .device_ids = {0, 1},
            .sharding = Tile({2}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                               xla::ifrt::Shape({2, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .device_ids = {0, 1},
            .sharding = Tile({2, 1}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                               xla::ifrt::Shape({1, 2, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({1, 2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({1, 2, 1})),
                },
            .device_ids = {0, 1},
            .sharding = Tile({1, 1, 2}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               xla::ifrt::Shape({4, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({5, 6}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({7, 8}, xla::ifrt::Shape({1, 2})),
                },
            .device_ids = {0, 1, 2, 3},
            .sharding = Tile({4, 1}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8},
                                               xla::ifrt::Shape({4, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3, 5, 7},
                                          xla::ifrt::Shape({4, 1})),
                    AsHostBuffer<int32_t>({2, 4, 6, 8},
                                          xla::ifrt::Shape({4, 1})),
                },
            .device_ids = {0, 1},
            .sharding = Tile({1, 2}),
        },
        // 2-D sharding
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 16},
                                               xla::ifrt::Shape({4, 4})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 5, 6},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({3, 4, 7, 8},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({9, 10, 13, 14},
                                          xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<int32_t>({11, 12, 15, 16},
                                          xla::ifrt::Shape({2, 2})),
                },
            .device_ids = {0, 1, 2, 3},
            .sharding = Tile({2, 2}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4, 5, 6, 7, 8, 9, 10,
                                                11, 12, 13, 14, 15, 16},
                                               xla::ifrt::Shape({4, 1, 4})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2, 5, 6},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({3, 4, 7, 8},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({9, 10, 13, 14},
                                          xla::ifrt::Shape({2, 1, 2})),
                    AsHostBuffer<int32_t>({11, 12, 15, 16},
                                          xla::ifrt::Shape({2, 1, 2})),
                },
            .device_ids = {0, 1, 2, 3},
            .sharding = Tile({2, 1, 2}),
        },
        // Partial replication
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                               xla::ifrt::Shape({2, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({1, 3}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({2, 1})),
                    AsHostBuffer<int32_t>({2, 4}, xla::ifrt::Shape({2, 1})),
                },
            .device_ids = {0, 1, 2, 3},
            .sharding = PartialTile({1, 2, 2}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},
                                               xla::ifrt::Shape({2, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .device_ids = {0, 1, 2, 3},
            .sharding = PartialTile({2, 1, 2}),
        },
        {
            .in_buffer = AsHostBuffer<int32_t>({1, 2, 3, 4},

                                               xla::ifrt::Shape({2, 2})),
            .expected_out_buffers =
                {
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({1, 2}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                    AsHostBuffer<int32_t>({3, 4}, xla::ifrt::Shape({1, 2})),
                },
            .device_ids = {3, 2, 1, 0},
            .sharding = PartialTile({2, 1, 2}),
        },
    }));

TEST(ShardingUtilsTest, MismatchRank) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  HostBuffer input_buffer =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 1, 2}));

  // Create contexts required for the compiler execution.
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));

  xla::HloSharding sharding = Tile({2, 1});
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), sharding);

  EXPECT_THAT(MakeArrayFromHostBuffer(*client, input_buffer, device_list,
                                      std::move(ifrt_sharding), thread_pool),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "shape must have 2 dimensions, but has 3 dimensions: "
                  "shape=[2,1,2], sharding={devices=[2,1]<=[2]}"));
}

TEST(ShardingUtilsTest, HostDramReclamation) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", kMaxParallelism);

  absl::Notification memory_freed;
  constexpr size_t kSize = 1024 * 1024;
  float* raw_data = new float[kSize];
  std::fill(raw_data, raw_data + kSize, 1.0f);

  xla::ifrt::Shape shape({static_cast<int64_t>(kSize)});

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));

  xla::HloSharding sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), sharding);

  absl::StatusOr<xla::ifrt::ArrayRef> assembled_array;
  {
    auto owner =
        std::shared_ptr<void>(raw_data, [&memory_freed, raw_data](void*) {
          delete[] raw_data;
          memory_freed.Notify();
        });

    HostBuffer host_buffer(
        raw_data, xla::ifrt::DType(xla::ifrt::DType::Kind::kF32), shape, owner);
    ASSERT_FALSE(memory_freed.HasBeenNotified());

    assembled_array =
        MakeArrayFromHostBuffer(*client, host_buffer, device_list,
                                std::move(ifrt_sharding), thread_pool);
  }

  // At this point, local host_buffer and local owner are destroyed.
  // Memory should only be kept alive by PJRT.
  ASSERT_OK(assembled_array);

  // Verify array ready, which should trigger callback and free memory.
  ASSERT_OK((*assembled_array)->GetReadyFuture().Await());

  memory_freed.WaitForNotification();
}

TEST(ShardingUtilsTest, MakeHostBufferFromSingleDeviceArrayOnDevice1) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", kMaxParallelism);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::DeviceListRef device_list,
      xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));

  HostBuffer input_buffer =
      AsHostBuffer<int32_t>({10, 20}, xla::ifrt::Shape({2}));

  xla::HloSharding sharding = xla::HloSharding::SingleDevice(1);
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), sharding);

  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef array,
      MakeArrayFromHostBuffer(*client, input_buffer, device_list, ifrt_sharding,
                              thread_pool));

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer output_buffer,
      MakeHostBufferFromArray(*client, *array, ifrt_sharding, device_list,
                              thread_pool)
          .Await());

  EXPECT_THAT(output_buffer, HostBufferEq(input_buffer));
}

TEST(ShardingUtilsTest, MakeArrayFromHostBufferExplicitDeviceTarget) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));
  ASSERT_GE(device_list->devices().size(), 2);

  auto target_device = device_list->devices()[1];

  xla::ifrt::ShardingRef ifrt_sharding =
      xla::ifrt::SingleDeviceSharding::Create(target_device,
                                              xla::ifrt::MemoryKind());

  HostBuffer input_buffer = AsHostBuffer<int32_t>(42);

  TF_ASSERT_OK_AND_ASSIGN(
      auto assembled_array,
      MakeArrayFromHostBuffer(*client, input_buffer, device_list,
                              std::move(ifrt_sharding), thread_pool));

  TF_ASSERT_OK_AND_ASSIGN(
      auto disassembled_arrays,
      assembled_array->DisassembleIntoSingleDeviceArrays(
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  ASSERT_EQ(disassembled_arrays.size(), 1);
  EXPECT_EQ(disassembled_arrays[0]->sharding().devices()->devices().size(), 1);
  EXPECT_EQ(disassembled_arrays[0]->sharding().devices()->devices()[0],
            target_device);
}

TEST(ShardingUtilsTest, MakeHostBufferFromNativeSingleDeviceArray) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));
  ASSERT_GE(device_list->devices().size(), 2);

  auto target_device = device_list->devices()[1];

  xla::HloSharding sharding =
      xla::HloSharding::SingleDevice(target_device->Id().value());
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), sharding);

  HostBuffer input_buffer = AsHostBuffer<int32_t>(42);

  auto single_device_sharding = xla::ifrt::SingleDeviceSharding::Create(
      target_device, xla::ifrt::MemoryKind());

  TF_ASSERT_OK_AND_ASSIGN(
      auto single_device_array,
      client->MakeArrayFromHostBuffer(
          input_buffer.base(), input_buffer.dtype(), input_buffer.shape(),
          std::nullopt, std::move(single_device_sharding),
          /*layout=*/nullptr,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/{}));

  TF_ASSERT_OK_AND_ASSIGN(auto output_buffer,
                          MakeHostBufferFromArray(*client, *single_device_array,
                                                  std::move(ifrt_sharding),
                                                  device_list, thread_pool)
                              .Await());

  EXPECT_THAT(input_buffer, HostBufferEq(output_buffer));
}

TEST(ShardingUtilsTest, MakeDeviceBufferFromPinnedHost) {
  constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", kMaxParallelism);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::DeviceListRef device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));

  xla::ifrt::ShardingRef ifrt_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_list->devices()[0],
                                              xla::ifrt::MemoryKind());

  auto storage = std::make_shared<std::vector<int32_t>>(
      std::initializer_list<int32_t>{10, 20, 30, 40});
  HostBuffer host_buffer(storage->data(),
                         xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({4}), storage);

  TF_ASSERT_OK_AND_ASSIGN(
      DeviceBuffer device_buffer,
      MakeDeviceBufferFromPinnedHost(client, host_buffer, device_list,
                                     ifrt_sharding, thread_pool));

  EXPECT_EQ(device_buffer.memory_kind, BufferMemoryKind::kPinnedHost);
  EXPECT_EQ(device_buffer.ifrt_array->shape(), xla::ifrt::Shape({4}));
  EXPECT_EQ(device_buffer.ifrt_array->sharding().memory_kind(),
            xla::ifrt::MemoryKind("pinned_host"));
  EXPECT_EQ(device_buffer.owner, storage);

  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer readback_buffer,
      MakeHostBufferFromArray(*client, *device_buffer.ifrt_array, ifrt_sharding,
                              device_list, thread_pool)
          .Await());
  EXPECT_THAT(readback_buffer, HostBufferEq(host_buffer));
}

TEST(ShardingUtilsTest, MakeArrayFromHostBufferStrided) {
  constexpr int kMaxParallelism = 4;
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", kMaxParallelism);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list, xla::ifrt::test_util::GetDevices(client.get(), {0, 1}));

  auto storage = std::make_shared<std::vector<int32_t>>(
      std::initializer_list<int32_t>{10, 20, 30, 40});
  absl::InlinedVector<int64_t, 4> byte_strides = {8};
  HostBuffer strided_buffer(
      storage->data(), xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
      xla::ifrt::Shape({2}), byte_strides, storage);

  HostBuffer expected_dense =
      AsHostBuffer<int32_t>({10, 30}, xla::ifrt::Shape({2}));

  // SingleDeviceSharding
  xla::ifrt::ShardingRef single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device_list->devices()[0],
                                              xla::ifrt::MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef single_arr,
      MakeArrayFromHostBuffer(*client, strided_buffer, device_list,
                              single_device_sharding, thread_pool));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer single_out,
      MakeHostBufferFromArray(*client, *single_arr, single_device_sharding,
                              device_list, thread_pool)
          .Await());
  EXPECT_THAT(single_out, HostBufferEq(expected_dense));

  // Replicated HloSharding
  xla::HloSharding replicated_sharding = xla::HloSharding::Replicate();
  xla::ifrt::ShardingRef ifrt_replicated = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), replicated_sharding);
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef rep_arr,
      MakeArrayFromHostBuffer(*client, strided_buffer, device_list,
                              ifrt_replicated, thread_pool));
  TF_ASSERT_OK_AND_ASSIGN(
      HostBuffer rep_out,
      MakeHostBufferFromArray(*client, *rep_arr, ifrt_replicated, device_list,
                              thread_pool)
          .Await());
  EXPECT_THAT(rep_out, HostBufferEq(expected_dense));
}

class CopyArrayToHostBufferTest : public ::testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    TF_ASSERT_OK_AND_ASSIGN(
        device_list_, xla::ifrt::test_util::GetDevices(client_.get(), {0, 1}));
  }

  absl::StatusOr<xla::ifrt::ArrayRef> MakeArray(
      const HostBuffer& source, const xla::ifrt::ShardingRef& sharding) {
    return MakeArrayFromHostBuffer(*client_, source, device_list_, sharding,
                                   thread_pool_);
  }

  static constexpr int kMaxParallelism = 16;
  tsl::thread::ThreadPool thread_pool_{
      tsl::Env::Default(), tsl::ThreadOptions(), "CopyTest", kMaxParallelism};
  std::shared_ptr<xla::ifrt::Client> client_;
  xla::ifrt::DeviceListRef device_list_;
};

// The copy must land in the caller's memory itself, so check the bytes at the
// address the caller passed in, for each sharding that keeps the whole value on
// one device.
TEST_F(CopyArrayToHostBufferTest, CopiesIntoCallerMemory) {
  const HostBuffer source =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  const std::vector<xla::ifrt::ShardingRef> shardings = {
      xla::ifrt::SingleDeviceSharding::Create(device_list_->devices()[0],
                                              xla::ifrt::MemoryKind()),
      xla::ifrt::HloSharding::Create(device_list_, xla::ifrt::MemoryKind(),
                                     xla::HloSharding::Replicate()),
  };
  for (const xla::ifrt::ShardingRef& sharding : shardings) {
    SCOPED_TRACE(absl::StrCat(*sharding));
    TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                            MakeArray(source, sharding));
    HostBuffer destination =
        AsHostBuffer<int32_t>({0, 0, 0, 0}, xla::ifrt::Shape({2, 2}));
    ASSERT_OK(
        CopyArrayToHostBuffer(*client_, *array, sharding, destination).Await());
    EXPECT_THAT(destination, HostBufferEq(source));
  }
}

// Only the element count has to match: a destination of another shape but the
// same size receives the data in row-major order.
TEST_F(CopyArrayToHostBufferTest, AcceptsDestinationOfSameSize) {
  const HostBuffer source =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                          MakeArray(source, sharding));
  HostBuffer destination =
      AsHostBuffer<int32_t>({0, 0, 0, 0}, xla::ifrt::Shape({4}));
  ASSERT_OK(
      CopyArrayToHostBuffer(*client_, *array, sharding, destination).Await());
  EXPECT_THAT(destination, HostBufferEq(AsHostBuffer<int32_t>(
                               {1, 2, 3, 4}, xla::ifrt::Shape({4}))));
}

TEST_F(CopyArrayToHostBufferTest, RejectsMismatchedDtype) {
  const HostBuffer source =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                          MakeArray(source, sharding));
  HostBuffer destination =
      AsHostBuffer<float>({0, 0, 0, 0}, xla::ifrt::Shape({2, 2}));
  EXPECT_THAT(
      CopyArrayToHostBuffer(*client_, *array, sharding, destination).Await(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("dtype")));
}

TEST_F(CopyArrayToHostBufferTest, RejectsMismatchedElementCount) {
  const HostBuffer source =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::SingleDeviceSharding::Create(
      device_list_->devices()[0], xla::ifrt::MemoryKind());
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                          MakeArray(source, sharding));
  HostBuffer destination = AsHostBuffer<int32_t>({0, 0}, xla::ifrt::Shape({2}));
  EXPECT_THAT(
      CopyArrayToHostBuffer(*client_, *array, sharding, destination).Await(),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             ::testing::HasSubstr("same number of elements")));
}

// A tiled array's data is split across devices, so no single device-to-host
// copy can fill one buffer with it.
TEST_F(CopyArrayToHostBufferTest, RejectsTiledArray) {
  const HostBuffer source =
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({4}));
  xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
      device_list_, xla::ifrt::MemoryKind(), xla::HloSharding::IotaTile({2}));
  TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef array,
                          MakeArray(source, sharding));
  HostBuffer destination =
      AsHostBuffer<int32_t>({0, 0, 0, 0}, xla::ifrt::Shape({4}));
  EXPECT_THAT(
      CopyArrayToHostBuffer(*client_, *array, sharding, destination).Await(),
      absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                             ::testing::HasSubstr("split across devices")));
}

}  // namespace
}  // namespace xla_serving::sharding_internal

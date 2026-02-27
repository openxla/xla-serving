#include "third_party/neptune/runtime/core/sharding_utils.h"

#include <cstdint>
#include <initializer_list>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_matchers.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/core/host_buffer_matcher.h"
#include "third_party/neptune/runtime/core/host_buffer_test_utils.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/memory.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/xla_sharding.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/lib/core/status_test_util.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/status_matchers.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/tsl/platform/ml_dtypes.h"

namespace jsv::sharding_internal {
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
  return xla::HloSharding::AssignDevice(device_index);
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

  auto ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), GetParam().sharding);
  xla::ifrt::ArrayRef assembled_array;

  TF_ASSERT_OK_AND_ASSIGN(
      assembled_array,
      client->AssembleArrayFromSingleDeviceArrays(
          split_arrays[0]->dtype(), GetParam().expected_out_buffer.shape(),
          std::move(ifrt_sharding), absl::MakeSpan(split_arrays),
          xla::ifrt::ArrayCopySemantics::kAlwaysCopy,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  TF_ASSERT_OK_AND_ASSIGN(
      auto output_buffer,
      MakeHostBufferFromArray(*client, *assembled_array, GetParam().sharding,
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
      auto assembled_array,
      MakeArrayFromHostBuffer(*client, input_buffer,
                              absl::MakeSpan(GetParam().device_ids),
                              GetParam().sharding, thread_pool));

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

  EXPECT_THAT(MakeArrayFromHostBuffer(*client, input_buffer, device_list,
                                      std::move(sharding), thread_pool),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  "shape must have 2 dimensions, but has 3 dimensions: "
                  "shape=[2,1,2], sharding={devices=[2,1]<=[2]}"));
}

}  // namespace
}  // namespace jsv::sharding_internal

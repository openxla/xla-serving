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

#include "runtime/data_loader/ocp_loader/checkpoint_loader.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/test.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

struct CheckpointLoaderTestParam {
  CheckpointType checkpoint_type;
  std::string root_path;
  std::vector<ArraySpec> weight_specs;
  std::vector<std::string> weight_names;
  std::vector<int> device_ids;
  std::vector<HostBuffer> expected_out_host_buffer;
};

using CheckpointLoaderTest =
    ::testing::TestWithParam<CheckpointLoaderTestParam>;

// Wrapper functions for building sharding specs for a given shape with a
// natural device order.
xla::OpSharding Tile(absl::Span<const int64_t> dims) {
  return xla::HloSharding::IotaTile(dims).ToProto();
}
xla::OpSharding Replicate() { return xla::HloSharding::Replicate().ToProto(); }

TEST_P(CheckpointLoaderTest, basic) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(
      auto device_list,
      xla::ifrt::test_util::GetDevices(client.get(), GetParam().device_ids));
  LOG(INFO) << " size of devices " << device_list->devices().size();

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      LoadCheckpoint(GetParam().root_path, GetParam().weight_names,
                     GetParam().weight_specs, device_list, *client,
                     GetParam().checkpoint_type));
  for (int i = 0; i < results.size(); ++i) {
    SCOPED_TRACE(absl::StrCat("Array ", i, " of ", results.size()));
    TF_ASSERT_OK_AND_ASSIGN(
        auto hlo_sharding,
        xla::HloSharding::FromProto(GetParam().weight_specs[i].sharding));
    xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
        device_list, xla::ifrt::MemoryKind(), hlo_sharding);
    tsl::Future<HostBuffer> host_buffer_future =
        sharding_internal::MakeHostBufferFromArray(
            *client, *results[i], ifrt_sharding, device_list, thread_pool);
    TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

    EXPECT_THAT(host_buffer,
                HostBufferEq(GetParam().expected_out_host_buffer[i]));
  }
}

INSTANTIATE_TEST_SUITE_P(
    CheckpointLoaderTests, CheckpointLoaderTest,
    ::testing::ValuesIn<CheckpointLoaderTestParam>({
        {
            .checkpoint_type = CheckpointType::kStandard,
            .root_path = tsl::GetDataDependencyFilepath(
                "runtime/"
                "data_loader/testdata/non_spmd_checkpoint_gen"),
            .weight_specs =
                {
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({4}),
                        .sharding = Replicate(),
                    },
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({8}),
                        .sharding = Replicate(),
                    },
                },
            .weight_names = {"params/0", "params/1"},
            .device_ids = {0},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({4})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({8})),
                },
        },
        {
            .checkpoint_type = CheckpointType::kStandard,
            .root_path = tsl::GetDataDependencyFilepath(
                "runtime/"
                "data_loader/testdata/spmd_checkpoint_gen"),
            .weight_specs =
                {
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 2}),
                        .sharding = Tile({2, 2}),
                    },
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 4}),
                        .sharding = Tile({2, 2}),
                    },
                },
            .weight_names = {"params/0", "params/1"},
            .device_ids = {0, 1, 2, 3},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({2, 4})),
                },
        },
        {
            .checkpoint_type = CheckpointType::kOcdbt,
            .root_path = tsl::GetDataDependencyFilepath(
                "runtime/"
                "data_loader/testdata/non_spmd_ocdbt_gen"),
            .weight_specs =
                {
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({4}),
                        .sharding = Replicate(),
                    },
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({8}),
                        .sharding = Replicate(),
                    },
                },
            .weight_names = {"param_0", "param_1"},
            .device_ids = {0},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({4})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({8})),
                },
        },
        {
            .checkpoint_type = CheckpointType::kOcdbt,
            .root_path = tsl::GetDataDependencyFilepath(
                "runtime/"
                "data_loader/testdata/spmd_ocdbt_gen"),
            .weight_specs =
                {
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 2}),
                        .sharding = Tile({2, 2}),
                    },
                    ArraySpec{
                        .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
                        .shape = xla::ifrt::Shape({2, 4}),
                        .sharding = Tile({2, 2}),
                    },
                },
            .weight_names = {"param_0", "param_1"},
            .device_ids = {0, 1, 2, 3},
            .expected_out_host_buffer =
                {
                    AsHostBuffer<float>({0, 1, 2, 3}, xla::ifrt::Shape({2, 2})),
                    AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                        xla::ifrt::Shape({2, 4})),
                },
        },
    }));

TEST(CheckpointLoaderTest, LoadWithHeterogeneousMixedMemoryKinds) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "Resharding", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  TF_ASSERT_OK_AND_ASSIGN(auto device_list,
                          xla::ifrt::test_util::GetDevices(client.get(), {0}));

  const std::string root_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata/"
      "non_spmd_checkpoint_gen");

  std::vector<ArraySpec> weight_specs = {
      ArraySpec{
          .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
          .shape = xla::ifrt::Shape({4}),
          .sharding = Replicate(),
          .memory_kind = std::nullopt,
      },
      ArraySpec{
          .dtype = xla::ifrt::DType(xla::ifrt::DType::kF32),
          .shape = xla::ifrt::Shape({8}),
          .sharding = Replicate(),
          .memory_kind = xla::ifrt::MemoryKind("pinned_host"),
      },
  };

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      LoadCheckpoint(root_path, {"params/0", "params/1"}, weight_specs,
                     device_list, *client, CheckpointType::kStandard));

  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->sharding().memory_kind(), xla::ifrt::MemoryKind());
  EXPECT_EQ(results[1]->sharding().memory_kind(),
            xla::ifrt::MemoryKind("pinned_host"));

  // Verify data parity for array 0 (Device).
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding_0, xla::HloSharding::FromProto(
                                                   weight_specs[0].sharding));
  xla::ifrt::ShardingRef ifrt_sharding_0 = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding_0);
  tsl::Future<HostBuffer> host_buffer_future_0 =
      sharding_internal::MakeHostBufferFromArray(
          *client, *results[0], ifrt_sharding_0, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer_0, host_buffer_future_0.Await());
  EXPECT_THAT(host_buffer_0, HostBufferEq(AsHostBuffer<float>(
                                 {0, 1, 2, 3}, xla::ifrt::Shape({4}))));

  // Verify data parity for array 1 (PinnedHost).
  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding_1, xla::HloSharding::FromProto(
                                                   weight_specs[1].sharding));
  xla::ifrt::ShardingRef ifrt_sharding_1 = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind("pinned_host"), hlo_sharding_1);
  tsl::Future<HostBuffer> host_buffer_future_1 =
      sharding_internal::MakeHostBufferFromArray(
          *client, *results[1], ifrt_sharding_1, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer_1, host_buffer_future_1.Await());
  EXPECT_THAT(host_buffer_1,
              HostBufferEq(AsHostBuffer<float>({4, 5, 6, 7, 8, 9, 10, 11},
                                               xla::ifrt::Shape({8}))));
}

TEST(CheckpointLoaderFailTest, DuplicateWeightNames) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  auto results = LoadCheckpoint(
      /*root_path=*/"", /*weight_names=*/{"params/0", "params/0", "params/1"},
      /*weight_specs=*/{}, /*device_list=*/{}, *client,
      CheckpointType::kStandard);
  EXPECT_THAT(
      results.status(),
      StatusIs(absl::StatusCode::kFailedPrecondition,
               HasSubstr("Duplicate weight names found: params/0 (count: 2)")));
}

}  // namespace
}  // namespace xla_serving

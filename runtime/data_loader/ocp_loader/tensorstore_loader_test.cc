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

#include "runtime/data_loader/ocp_loader/tensorstore_loader.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @xla_serving
#include "runtime/data_loader/ocp_loader/shared.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/test.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {
namespace {

TEST(TensorstoreLoaderTest, Basic) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  std::vector<int> device_ids = {0};
  TF_ASSERT_OK_AND_ASSIGN(auto device_list, xla::ifrt::test_util::GetDevices(
                                                client.get(), device_ids));

  std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"non_spmd_checkpoint_gen/params/0\"}}",
      .op_sharding = xla::HloSharding::Replicate().ToProto(),
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
      .array_name = "test_array",
  });

  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      LoadFromTensorstore(client.get(), specs,
                          /*base_dir=*/parent_path,
                          /*opts=*/{}, CheckpointType::kStandard));

  ASSERT_EQ(results.size(), 1);
  EXPECT_TRUE(results[0]->sharding().IsFullyReplicated());

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          xla::HloSharding::FromProto(*specs[0].op_sharding));
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding);
  tsl::Future<xla_serving::HostBuffer> host_buffer_future =
      xla_serving::sharding_internal::MakeHostBufferFromArray(
          *client, *results[0], ifrt_sharding, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

  EXPECT_THAT(host_buffer, xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                               {0, 1, 2, 3}, xla::ifrt::Shape({4}))));
}

TEST(TensorstoreLoaderTest, LoadWithPinnedHostMemoryKind) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  std::vector<int> device_ids = {0};
  TF_ASSERT_OK_AND_ASSIGN(auto device_list, xla::ifrt::test_util::GetDevices(
                                                client.get(), device_ids));

  std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"non_spmd_checkpoint_gen/params/0\"}}",
      .op_sharding = xla::HloSharding::Replicate().ToProto(),
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
      .array_name = "test_array",
  });

  LoadFromTensorstoreExtraOptions opts;
  opts.memory_kind = xla::ifrt::MemoryKind("pinned_host");

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> results,
                          LoadFromTensorstore(client.get(), specs,
                                              /*base_dir=*/parent_path, opts,
                                              CheckpointType::kStandard));

  ASSERT_EQ(results.size(), 1);
  EXPECT_EQ(results[0]->sharding().memory_kind(),
            xla::ifrt::MemoryKind("pinned_host"));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                          xla::HloSharding::FromProto(*specs[0].op_sharding));
  xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind("pinned_host"), hlo_sharding);
  tsl::Future<xla_serving::HostBuffer> host_buffer_future =
      xla_serving::sharding_internal::MakeHostBufferFromArray(
          *client, *results[0], ifrt_sharding, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

  EXPECT_THAT(host_buffer, xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                               {0, 1, 2, 3}, xla::ifrt::Shape({4}))));
}

TEST(TensorstoreLoaderTest, LoadWithHeterogeneousPerSpecMemoryKinds) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  std::vector<int> device_ids = {0};
  TF_ASSERT_OK_AND_ASSIGN(auto device_list, xla::ifrt::test_util::GetDevices(
                                                client.get(), device_ids));
  const std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  std::vector<TensorStoreLoadSpec> specs;
  // Spec 0: Default device memory space
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"non_spmd_checkpoint_gen/params/0\"}}",
      .op_sharding = xla::HloSharding::Replicate().ToProto(),
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
      .array_name = "param_0",
      .memory_kind = std::nullopt,
  });
  // Spec 1: Pinned host memory space
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"non_spmd_checkpoint_gen/params/1\"}}",
      .op_sharding = xla::HloSharding::Replicate().ToProto(),
      .per_shard_shape = {8},
      .global_shape = {8},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
      .array_name = "param_1",
      .memory_kind = xla::ifrt::MemoryKind("pinned_host"),
  });

  LoadFromTensorstoreExtraOptions opts;

  TF_ASSERT_OK_AND_ASSIGN(std::vector<xla::ifrt::ArrayRef> results,
                          LoadFromTensorstore(client.get(), specs,
                                              /*base_dir=*/parent_path, opts,
                                              CheckpointType::kStandard));

  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(results[0]->sharding().memory_kind(), xla::ifrt::MemoryKind());
  EXPECT_EQ(results[1]->sharding().memory_kind(),
            xla::ifrt::MemoryKind("pinned_host"));

  // Verify numerical data parity across both memory spaces
  {
    TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                            xla::HloSharding::FromProto(*specs[0].op_sharding));
    xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
        device_list, xla::ifrt::MemoryKind(), hlo_sharding);
    tsl::Future<xla_serving::HostBuffer> host_buffer_future =
        xla_serving::sharding_internal::MakeHostBufferFromArray(
            *client, *results[0], ifrt_sharding, device_list, thread_pool);
    TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

    EXPECT_THAT(host_buffer, xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                                 {0, 1, 2, 3}, xla::ifrt::Shape({4}))));
  }
  {
    TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding,
                            xla::HloSharding::FromProto(*specs[1].op_sharding));
    xla::ifrt::ShardingRef ifrt_sharding = xla::ifrt::HloSharding::Create(
        device_list, xla::ifrt::MemoryKind("pinned_host"), hlo_sharding);
    tsl::Future<xla_serving::HostBuffer> host_buffer_future =
        xla_serving::sharding_internal::MakeHostBufferFromArray(
            *client, *results[1], ifrt_sharding, device_list, thread_pool);
    TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, host_buffer_future.Await());

    EXPECT_THAT(host_buffer,
                xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                    {4, 5, 6, 7, 8, 9, 10, 11}, xla::ifrt::Shape({8}))));
  }
}

TEST(TensorstoreLoaderTest, FailsWithInvalidLayout) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  std::vector<int> device_ids = {0};
  TF_ASSERT_OK_AND_ASSIGN(auto device_list, xla::ifrt::test_util::GetDevices(
                                                client.get(), device_ids));

  std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  // Invalid layout with 2 dimensions.
  xla::LayoutProto layout;
  layout.add_minor_to_major(0);
  layout.add_minor_to_major(1);

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"non_spmd_checkpoint_gen/params/0\"}}",
      .op_sharding = xla::HloSharding::Replicate().ToProto(),
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
      .layout = layout,
  });

  EXPECT_THAT(
      LoadFromTensorstore(client.get(), specs,
                          /*base_dir=*/parent_path,
                          /*opts=*/{}, CheckpointType::kStandard),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument
          ));
}

TEST(TensorstoreLoaderTest, LoadShardedOcdbt) {
  tsl::thread::ThreadPool thread_pool(tsl::Env::Default(), tsl::ThreadOptions(),
                                      "TestThreadPool", /*num_threads=*/4);

  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());
  std::vector<int> device_ids = {0, 1, 2, 3};
  TF_ASSERT_OK_AND_ASSIGN(auto device_list, xla::ifrt::test_util::GetDevices(
                                                client.get(), device_ids));

  std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": \"ocdbt\", "
          "\"base\": {\"driver\": \"file\", \"path\": \"spmd_ocdbt_gen\"}, "
          "\"path\": \"param_0\"}}",
      .op_sharding = xla::HloSharding::IotaTile({2, 2}).ToProto(),
      .per_shard_shape = {1, 1},
      .global_shape = {2, 2},
      .dtype = xla::PrimitiveType::F32,
      .shards =
          {
              {.device_id = 0, .index_offset = {0, 0}},
              {.device_id = 1, .index_offset = {0, 1}},
              {.device_id = 2, .index_offset = {1, 0}},
              {.device_id = 3, .index_offset = {1, 1}},
          },
      .array_name = "param_0",
  });
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": \"ocdbt\", "
          "\"base\": {\"driver\": \"file\", \"path\": \"spmd_ocdbt_gen\"}, "
          "\"path\": \"param_1\"}}",
      .op_sharding = xla::HloSharding::IotaTile({2, 2}).ToProto(),
      .per_shard_shape = {1, 2},
      .global_shape = {2, 4},
      .dtype = xla::PrimitiveType::F32,
      .shards =
          {
              {.device_id = 0, .index_offset = {0, 0}},
              {.device_id = 1, .index_offset = {0, 2}},
              {.device_id = 2, .index_offset = {1, 0}},
              {.device_id = 3, .index_offset = {1, 2}},
          },
      .array_name = "param_1",
  });

  TF_ASSERT_OK_AND_ASSIGN(auto context, GetOcdbtContext());
  TensorStoreLoader loader(client.get(), context);
  int64_t total_transfer = 0;
  TF_ASSERT_OK_AND_ASSIGN(
      std::vector<xla::ifrt::ArrayRef> results,
      loader.Load(specs, /*base_dir=*/parent_path, &total_transfer));

  ASSERT_EQ(results.size(), 2);
  EXPECT_EQ(total_transfer, 48);

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding0,
                          xla::HloSharding::FromProto(*specs[0].op_sharding));
  xla::ifrt::ShardingRef ifrt_sharding0 = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding0);
  tsl::Future<xla_serving::HostBuffer> host_buffer_future0 =
      xla_serving::sharding_internal::MakeHostBufferFromArray(
          *client, *results[0], ifrt_sharding0, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer0, host_buffer_future0.Await());

  EXPECT_THAT(host_buffer0, xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                                {0, 1, 2, 3}, xla::ifrt::Shape({2, 2}))));

  TF_ASSERT_OK_AND_ASSIGN(auto hlo_sharding1,
                          xla::HloSharding::FromProto(*specs[1].op_sharding));
  xla::ifrt::ShardingRef ifrt_sharding1 = xla::ifrt::HloSharding::Create(
      device_list, xla::ifrt::MemoryKind(), hlo_sharding1);
  tsl::Future<xla_serving::HostBuffer> host_buffer_future1 =
      xla_serving::sharding_internal::MakeHostBufferFromArray(
          *client, *results[1], ifrt_sharding1, device_list, thread_pool);
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer1, host_buffer_future1.Await());

  EXPECT_THAT(host_buffer1,
              xla_serving::HostBufferEq(xla_serving::AsHostBuffer<float>(
                  {4, 5, 6, 7, 8, 9, 10, 11}, xla::ifrt::Shape({2, 4}))));
}

TEST(TensorstoreLoaderTest, FailsWithMissingKvstore) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string = "{\"driver\": \"zarr\"}",
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(loader.Load(specs),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("TensorStore spec must have a kvstore.")));
}

TEST(TensorstoreLoaderTest, FailsWithRelativePathAndEmptyBaseDir) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string = "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
                             "\"file\", \"path\": \"relative/path\"}}",
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(loader.Load(specs, /*base_dir=*/""),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Cannot resolve relative checkpoint path with an empty"
                      " base_dir since the old path is not an absolute path")));
}

TEST(TensorstoreLoaderTest, FailsWithMissingPathInKvstore) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string = "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
                             "\"file\"}}",
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(loader.Load(specs, /*base_dir=*/""),
              absl_testing::StatusIs(
                  absl::StatusCode::kNotFound,
                  testing::HasSubstr(
                      "Metadata at local file \".zarray\" does not exist")));
}

TEST(TensorstoreLoaderTest, FailsWithAbsolutePath) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string = "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
                             "\"file\", \"path\": \"/etc/passwd\"}}",
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(loader.Load(specs, /*base_dir=*/""),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Absolute paths are not allowed in TensorStore specs")));
}

TEST(TensorstoreLoaderTest, FailsWithAbsolutePathTraversalDotDot) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
          "\"file\", \"path\": \"/some/base/dir/../../etc/passwd\"}}",
      .per_shard_shape = {4},
      .global_shape = {4},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(loader.Load(specs, /*base_dir=*/"/some/base/dir"),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr(
                      "Absolute paths are not allowed in TensorStore specs")));
}

TEST(TensorstoreLoaderTest, FailsWithHugeShapeOverflow) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string = "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": "
                             "\"file\", \"path\": \"some/relative/path\"}}",
      .per_shard_shape = {999999999999999999LL, 999999999999999999LL},
      .global_shape = {999999999999999999LL, 999999999999999999LL},
      .dtype = xla::PrimitiveType::F32,
      .shards = {{.device_id = 0, .index_offset = {}}},
  });

  TensorStoreLoader loader(client.get());
  EXPECT_THAT(
      loader.Load(specs, /*base_dir=*/"/some/base/dir"),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Integer overflow calculating total_shard_size")));
}

TEST(TensorstoreLoaderTest, FailsWithLoadAggregationOverflow) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                          xla::ifrt::test_util::GetClient());

  std::string parent_path = tsl::GetDataDependencyFilepath(
      "runtime/data_loader/testdata");

  std::vector<TensorStoreLoadSpec> specs;
  specs.push_back({
      .json_ts_spec_string =
          "{\"driver\": \"zarr\", \"kvstore\": {\"driver\": \"ocdbt\", "
          "\"base\": {\"driver\": \"file\", \"path\": \"spmd_ocdbt_gen\"}, "
          "\"path\": \"param_0\"}}",
      .op_sharding = xla::HloSharding::IotaTile({2, 2}).ToProto(),
      .per_shard_shape = {1, 1},
      .global_shape = {2, 2},
      .dtype = xla::PrimitiveType::F32,
      .shards =
          {
              {.device_id = 0, .index_offset = {0, 0}},
              {.device_id = 1, .index_offset = {0, 1}},
              {.device_id = 2, .index_offset = {1, 0}},
              {.device_id = 3, .index_offset = {1, 1}},
          },
      .array_name = "param_0",
  });

  TF_ASSERT_OK_AND_ASSIGN(auto context, GetOcdbtContext());
  TensorStoreLoader loader(client.get(), context);
  int64_t total_transfer = 9223372036854775800LL;  // Near maximal size
  EXPECT_THAT(
      loader.Load(specs, /*base_dir=*/parent_path, &total_transfer),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Integer overflow aggregating total_transfer")));
}

}  // namespace
}  // namespace xla_serving

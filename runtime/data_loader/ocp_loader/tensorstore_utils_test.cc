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

#include "runtime/data_loader/ocp_loader/tensorstore_utils.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorstore/data_type.h"  // from @tensorstore

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::Each;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::TestWithParam;

struct GetJsonTspecTestCase {
  std::string test_name;
  std::string path;
  std::string parent_dir;
  bool use_ocdbt;
  bool use_zarr3;
  std::optional<std::string> name;
  std::optional<std::variant<int, std::string>> process_index;
  std::optional<std::string> metadata_key;
  std::string expected_tspec_json_str;
};

using TensorStoreJsonSpecUtilTest = TestWithParam<GetJsonTspecTestCase>;

TEST_P(TensorStoreJsonSpecUtilTest, GetJsonTspec) {
  const GetJsonTspecTestCase& test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      nlohmann::json json_tspec,
      GetJsonTspec(test_case.parent_dir, test_case.use_ocdbt,
                   test_case.use_zarr3, test_case.name, test_case.process_index,
                   test_case.metadata_key));
  nlohmann::json expected_json_tspec =
      nlohmann::json::parse(test_case.expected_tspec_json_str);
  EXPECT_EQ(json_tspec, expected_json_tspec);
}

INSTANTIATE_TEST_SUITE_P(
    GetJsonTspecTest, TensorStoreJsonSpecUtilTest,
    testing::ValuesIn<GetJsonTspecTestCase>({
        {
            .test_name = "basic_with_ocdbt",
            .parent_dir = "/tmp/test_dir",
            .use_ocdbt = true,
            .use_zarr3 = false,
            .name = "param_a",
            .process_index = std::nullopt,
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({"driver":"zarr",
                "kvstore": {
                    "base": {
                      "driver":"file",
                      "path":"/tmp/test_dir"
                    },
                    "cache_pool":"cache_pool#ocdbt",
                    "driver":"ocdbt",
                    "experimental_read_coalescing_interval":"1ms",
                    "experimental_read_coalescing_merged_bytes":500000000000,
                    "experimental_read_coalescing_threshold_bytes":1000000,
                    "path":"param_a"
                  },
                  "recheck_cached_data":false,
                  "recheck_cached_metadata":false})",
        },
        {
            .test_name = "basic_without_ocdbt",
            .parent_dir = "/tmp/test_dir",
            .use_ocdbt = false,
            .use_zarr3 = false,
            .name = "param_a",
            .process_index = std::nullopt,
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                 "driver":"zarr",
                 "kvstore":{
                   "driver":"file",
                   "path":"/tmp/test_dir/param_a"
                 },
                 "recheck_cached_data":false,
                 "recheck_cached_metadata":false})",
        },
        {
            .test_name = "basic_with_ocdbt_and_empty_param_name",
            .parent_dir = "/tmp/test_dir",
            .use_ocdbt = true,
            .use_zarr3 = false,
            .name = std::nullopt,
            .process_index = std::nullopt,
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                "driver":"zarr",
                "kvstore":{
                  "base":{
                    "driver":"file",
                    "path":"/tmp/test_dir"
                  },
                "cache_pool":"cache_pool#ocdbt",
                "driver":"ocdbt",
                "experimental_read_coalescing_interval":"1ms",
                "experimental_read_coalescing_merged_bytes":500000000000,
                "experimental_read_coalescing_threshold_bytes":1000000},
                "recheck_cached_data":false,
                "recheck_cached_metadata":false})",
        },
        {
            .test_name = "WithProcessIndex",
            .parent_dir = "/tmp/test_dir",
            .use_ocdbt = true,
            .use_zarr3 = false,
            .name = "1",
            .process_index = 123,
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                "driver":"zarr",
                "kvstore":{
                  "base":{
                    "driver":"file",
                    "path":"/tmp/test_dir/ocdbt.process_123" 
                  },
                  "cache_pool":"cache_pool#ocdbt",
                  "driver":"ocdbt",
                  "experimental_read_coalescing_interval":"1ms",
                  "experimental_read_coalescing_merged_bytes":500000000000,
                  "experimental_read_coalescing_threshold_bytes":1000000,
                  "path":"1"
                },
                "recheck_cached_data":false,
                "recheck_cached_metadata":false})",
        },
        {
            .test_name = "gcs_without_ocdbt",
            .parent_dir = "gs://gcs_bucket/object_path",
            .use_ocdbt = false,
            .use_zarr3 = false,
            .name = "param_aa",
            .process_index = std::nullopt,
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                "driver":"zarr",
                "kvstore":{
                  "bucket":"gcs_bucket",
                  "driver":"gcs",
                  "path":"object_path/param_aa"
                },
                "recheck_cached_data":false,
                "recheck_cached_metadata":false})",
        },
        {
            .test_name = "gcs_with_ocdbt",
            .parent_dir = "gs://gcs_bucket/object_path",
            .use_ocdbt = true,
            .use_zarr3 = false,
            .name = "param_aa",
            .process_index = "proc1",
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                "driver":"zarr",
                "kvstore":{
                  "base":"gs://gcs_bucket/object_path/ocdbt.process_proc1",
                  "cache_pool":"cache_pool#ocdbt",
                  "driver":"ocdbt",
                  "experimental_read_coalescing_interval":"1ms",
                  "experimental_read_coalescing_merged_bytes":500000000000,
                  "experimental_read_coalescing_threshold_bytes":1000000,
                  "path":"param_aa"
                },
                "recheck_cached_data":false,
                "recheck_cached_metadata":false})",
        },
    }),
    [](const testing::TestParamInfo<TensorStoreJsonSpecUtilTest::ParamType>&
           info) { return info.param.test_name; });

struct FormattingTestCase {
  std::string test_name;
  std::string directory;
  std::string param_name;
  bool use_ocdbt;
  int process_id;
  std::string expected_tspec_json_str;
};

using TensorStoreUtilTest = TestWithParam<FormattingTestCase>;

TEST_P(TensorStoreUtilTest, BuildKvstoreTspec) {
  const FormattingTestCase& test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(
      nlohmann::json json_kvstore_spec,
      BuildKvstoreTspec(test_case.directory, test_case.param_name,
                        test_case.use_ocdbt, test_case.process_id));
  nlohmann::json expected_json_kvstore_spec =
      nlohmann::json::parse(test_case.expected_tspec_json_str);
  EXPECT_TRUE(json_kvstore_spec == expected_json_kvstore_spec);
}

INSTANTIATE_TEST_SUITE_P(
    NumbersTestSuiteInstantiation, TensorStoreUtilTest,
    testing::ValuesIn<FormattingTestCase>({
        {.test_name = "local_fs_path",
         .directory = "/tmp/local_path",
         .param_name = "params/a",
         .use_ocdbt = false,
         .process_id = 13,
         .expected_tspec_json_str =
             R"({"driver":"file","path":"/tmp/local_path/params/a"})"},
        {.test_name = "bare_local_path",
         .directory = "/some/local/dir",
         .param_name = "params/a",
         .use_ocdbt = false,
         .process_id = 13,
         .expected_tspec_json_str =
             R"({"driver":"file","path":"/some/local/dir/params/a"})"},
        {.test_name = "regular_gcs_path_with_ocdbt",
         .directory = "gs://gcs_bucket/object_path",
         .param_name = "params/a",
         .use_ocdbt = true,
         .process_id = 666,
         .expected_tspec_json_str =
             R"({
         "driver": "ocdbt",
         "base": "gs://gcs_bucket/object_path/ocdbt.process_666",
         "path": "params/a",
         "experimental_read_coalescing_threshold_bytes": 1000000,
         "experimental_read_coalescing_merged_bytes": 500000000000,
         "experimental_read_coalescing_interval": "1ms",
         "cache_pool": "cache_pool#ocdbt"
         })"},
        {.test_name = "regular_gcs_path_without_ocdbt",
         .directory = "gs://gcs_bucket/object_path",
         .param_name = "params/a",
         .use_ocdbt = false,
         .process_id = 100,
         .expected_tspec_json_str =
             R"({
             "bucket":"gcs_bucket",
             "driver":"gcs",
             "path":"object_path/params/a"
             })"},
    }),
    [](const testing::TestParamInfo<TensorStoreUtilTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(TensorStoreUtilDirectTest, BareLocalPathResolvesToFileDriver) {
  TF_ASSERT_OK_AND_ASSIGN(
      nlohmann::json json_kvstore_spec,
      BuildKvstoreTspec("/some/local/dir", /*name=*/std::nullopt,
                        /*use_ocdbt=*/false, /*process_id=*/std::nullopt));
  EXPECT_EQ(json_kvstore_spec,
            nlohmann::json({{"driver", "file"}, {"path", "/some/local/dir"}}));
}

TEST(TensorStoreUtilDirectTest, RelativePathResolvesToFileDriver) {
  TF_ASSERT_OK_AND_ASSIGN(const nlohmann::json tspec,
                       BuildKvstoreTspec("relative/dir", /*name=*/std::nullopt,
                                         /*use_ocdbt=*/false));
  EXPECT_EQ(tspec["driver"], "file");
  EXPECT_EQ(tspec["path"], "relative/dir");
}

TEST(TensorStoreUtilDirectTest, EmptyDirectoryResolvesToFileDriver) {
  TF_ASSERT_OK_AND_ASSIGN(const nlohmann::json tspec,
                       BuildKvstoreTspec("", /*name=*/std::nullopt,
                                         /*use_ocdbt=*/false));
  EXPECT_EQ(tspec["driver"], "file");
  EXPECT_EQ(tspec["path"], "");
}

TEST(AssembleOpaqueIfrtArrayTest, AssembleArraysSuccess) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                       xla::ifrt::test_util::GetClient());
  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  const void* data_ptr = static_cast<const void*>(data.data());
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  xla::ifrt::Shape shape({static_cast<int64_t>(data.size())});
  xla::ifrt::MemoryKind memory_kind("unpinned_host");
  std::vector<xla::ifrt::ArrayRef> arrays;
  for (xla::ifrt::Device* device : client->addressable_devices()) {
    std::unique_ptr<xla::ifrt::SingleDeviceSharding> sharding =
        xla::ifrt::SingleDeviceSharding::Create(device, memory_kind);
    TF_ASSERT_OK_AND_ASSIGN(
        xla::ifrt::ArrayRef arr,
        client->MakeArrayFromHostBuffer(
            data_ptr, dtype, shape, /*byte_strides=*/std::nullopt,
            xla::ifrt::ShardingRef(std::move(sharding)), /*layout=*/nullptr,
            xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            /*on_done_with_host_buffer=*/[] {}));
    arrays.push_back(std::move(arr));
  }
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef assembled,
      AssembleOpaqueIfrtArray(client.get(), absl::MakeSpan(arrays),
                              {static_cast<int64_t>(data.size())}));
  EXPECT_THAT(assembled->sharding().memory_kind(), Eq(memory_kind));
  EXPECT_THAT(assembled->shape(),
              Eq(xla::ifrt::Shape({static_cast<int64_t>(data.size())})));
  EXPECT_THAT(assembled->dtype(), Eq(dtype));
  const auto* concrete_sharding =
      xla::ifrt::cast<xla::ifrt::ConcreteSharding>(&assembled->sharding());
  EXPECT_THAT(concrete_sharding->shard_shapes(), Each(Eq(shape)));
}

TEST(AssembleOpaqueIfrtArrayTest, FailsWithEmptyInput) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                       xla::ifrt::test_util::GetClient());
  EXPECT_THAT(AssembleOpaqueIfrtArray(client.get(), /*ifrt_arrays=*/{},
                                      /*global_shape=*/{}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Cannot assemble 0 arrays into an array")));
}

TEST(AssembleOpaqueIfrtArrayTest, FailsWithDifferentMemoryKinds) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> client,
                       xla::ifrt::test_util::GetClient());
  std::vector<int32_t> data{1, 2, 3, 4, 5, 6};
  const void* data_ptr = static_cast<const void*>(data.data());
  xla::ifrt::DType dtype(xla::ifrt::DType::kS32);
  xla::ifrt::Shape shape({static_cast<int64_t>(data.size())});
  xla::ifrt::Device* device = client->addressable_devices().front();
  std::vector<xla::ifrt::ArrayRef> arrays;

  xla::ifrt::MemoryKind memory_kind_1("unpinned_host");
  std::unique_ptr<xla::ifrt::SingleDeviceSharding> sharding_1 =
      xla::ifrt::SingleDeviceSharding::Create(device, memory_kind_1);
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef arr_1,
      client->MakeArrayFromHostBuffer(
          data_ptr, dtype, shape, /*byte_strides=*/std::nullopt,
          xla::ifrt::ShardingRef(std::move(sharding_1)), /*layout=*/nullptr,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/[] {}));
  arrays.push_back(std::move(arr_1));

  xla::ifrt::MemoryKind memory_kind_2("pinned_host");
  std::unique_ptr<xla::ifrt::SingleDeviceSharding> sharding_2 =
      xla::ifrt::SingleDeviceSharding::Create(device, memory_kind_2);
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef arr_2,
      client->MakeArrayFromHostBuffer(
          data_ptr, dtype, shape, /*byte_strides=*/std::nullopt,
          xla::ifrt::ShardingRef(std::move(sharding_2)), /*layout=*/nullptr,
          xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
          /*on_done_with_host_buffer=*/[] {}));
  arrays.push_back(std::move(arr_2));

  EXPECT_THAT(AssembleOpaqueIfrtArray(client.get(), absl::MakeSpan(arrays),
                                      {static_cast<int64_t>(data.size())}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("assemble arrays of different memory kind")));
}

struct DTypeTestParam {
  std::string test_name;
  xla::ifrt::DType ifrt_dtype;
  tensorstore::DataType ts_dtype;
};

using ToTensorStoreDTypeTest = TestWithParam<DTypeTestParam>;

TEST_P(ToTensorStoreDTypeTest, ConvertsDType) {
  const DTypeTestParam& test_case = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(tensorstore::DataType result,
                       ToTensorStoreDType(test_case.ifrt_dtype));
  EXPECT_EQ(result, test_case.ts_dtype);
}

INSTANTIATE_TEST_SUITE_P(
    ToTensorStoreDTypeTests, ToTensorStoreDTypeTest,
    testing::ValuesIn<DTypeTestParam>({
        {"Pred", xla::ifrt::DType(xla::ifrt::DType::kPred),
         tensorstore::dtype_v<tensorstore::dtypes::bool_t>},
        {"S2", xla::ifrt::DType(xla::ifrt::DType::kS2),
         tensorstore::dtype_v<tensorstore::dtypes::int2_t>},
        {"S4", xla::ifrt::DType(xla::ifrt::DType::kS4),
         tensorstore::dtype_v<tensorstore::dtypes::int4_t>},
        {"S8", xla::ifrt::DType(xla::ifrt::DType::kS8),
         tensorstore::dtype_v<tensorstore::dtypes::int8_t>},
        {"S16", xla::ifrt::DType(xla::ifrt::DType::kS16),
         tensorstore::dtype_v<tensorstore::dtypes::int16_t>},
        {"S32", xla::ifrt::DType(xla::ifrt::DType::kS32),
         tensorstore::dtype_v<tensorstore::dtypes::int32_t>},
        {"S64", xla::ifrt::DType(xla::ifrt::DType::kS64),
         tensorstore::dtype_v<tensorstore::dtypes::int64_t>},
        {"U8", xla::ifrt::DType(xla::ifrt::DType::kU8),
         tensorstore::dtype_v<tensorstore::dtypes::uint8_t>},
        {"U16", xla::ifrt::DType(xla::ifrt::DType::kU16),
         tensorstore::dtype_v<tensorstore::dtypes::uint16_t>},
        {"U32", xla::ifrt::DType(xla::ifrt::DType::kU32),
         tensorstore::dtype_v<tensorstore::dtypes::uint32_t>},
        {"U64", xla::ifrt::DType(xla::ifrt::DType::kU64),
         tensorstore::dtype_v<tensorstore::dtypes::uint64_t>},
        {"BF16", xla::ifrt::DType(xla::ifrt::DType::kBF16),
         tensorstore::dtype_v<tensorstore::dtypes::bfloat16_t>},
        {"F16", xla::ifrt::DType(xla::ifrt::DType::kF16),
         tensorstore::dtype_v<tensorstore::dtypes::float16_t>},
        {"F32", xla::ifrt::DType(xla::ifrt::DType::kF32),
         tensorstore::dtype_v<tensorstore::dtypes::float32_t>},
        {"F64", xla::ifrt::DType(xla::ifrt::DType::kF64),
         tensorstore::dtype_v<tensorstore::dtypes::float64_t>},
        {"C64", xla::ifrt::DType(xla::ifrt::DType::kC64),
         tensorstore::dtype_v<tensorstore::dtypes::complex64_t>},
        {"C128", xla::ifrt::DType(xla::ifrt::DType::kC128),
         tensorstore::dtype_v<tensorstore::dtypes::complex128_t>},
        {"F8E4M3FN", xla::ifrt::DType(xla::ifrt::DType::kF8E4M3FN),
         tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3fn_t>},
        {"F8E4M3B11FNUZ", xla::ifrt::DType(xla::ifrt::DType::kF8E4M3B11FNUZ),
         tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3b11fnuz_t>},
        {"F8E4M3FNUZ", xla::ifrt::DType(xla::ifrt::DType::kF8E4M3FNUZ),
         tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3fnuz_t>},
        {"F8E5M2", xla::ifrt::DType(xla::ifrt::DType::kF8E5M2),
         tensorstore::dtype_v<tensorstore::dtypes::float8_e5m2_t>},
        {"F8E5M2FNUZ", xla::ifrt::DType(xla::ifrt::DType::kF8E5M2FNUZ),
         tensorstore::dtype_v<tensorstore::dtypes::float8_e5m2fnuz_t>},
        {"F4E2M1FN", xla::ifrt::DType(xla::ifrt::DType::kF4E2M1FN),
         tensorstore::dtype_v<tensorstore::dtypes::float4_e2m1fn_t>},
    }),
    [](const testing::TestParamInfo<ToTensorStoreDTypeTest::ParamType>& info) {
      return info.param.test_name;
    });

TEST(ToTensorStoreDTypeUnsupported, Unimplemented) {
  EXPECT_THAT(ToTensorStoreDType(xla::ifrt::DType(xla::ifrt::DType::kToken)),
              StatusIs(absl::StatusCode::kUnimplemented));
}

}  // namespace
}  // namespace xla_serving

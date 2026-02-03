#include "third_party/neptune/data_loader/ocp_loader/tensorstore_utils.h"

#include <optional>
#include <string>
#include <variant>

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/json/include/nlohmann/json.hpp"
#include "third_party/json/include/nlohmann/json_fwd.hpp"

namespace neptune {
namespace {

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
  ASSERT_OK_AND_ASSIGN(
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
                      "driver":"gfile",
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
                   "driver":"gfile",
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
                    "driver":"gfile",
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
                    "driver":"gfile",
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
        // BEGIN GOOGLE-INTERNAL
        {
            .test_name = "CnsPathWithOcdbt",
            .parent_dir = "/cns/lu-d/home/user_home",
            .use_ocdbt = false,
            .use_zarr3 = false,
            .name = "2",
            .process_index = "process1",
            .metadata_key = std::nullopt,
            .expected_tspec_json_str =
                R"({
                 "driver":"zarr",
                 "kvstore":{
                  "driver":"gfile",
                  "path":"/cns/lu-d/home/user_home/2"
                 },
                "recheck_cached_data":false,
                "recheck_cached_metadata":false})",
        },
        // END GOOGLE-INTERNAL
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
  ASSERT_OK_AND_ASSIGN(
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
             R"({"driver":"gfile","path":"/tmp/local_path/params/a"})"
        },
        // BEGIN GOOGLE-INTERNAL
        {.test_name = "cns_path",
         .directory = "/cns/lu-d/home/user_home/",
         .param_name = "params/a",
         .use_ocdbt = false,
         .process_id = 13,
         .expected_tspec_json_str =
             R"({"driver":"gfile","path":"/cns/lu-d/home/user_home/params/a"})"
        },
        {.test_name = "bigstore_path",
         .directory = "/bigstore/user_bucket/user_object/",
         .param_name = "params/a",
         .use_ocdbt = false,
         .process_id = 13,
         .expected_tspec_json_str =
             R"({
             "driver":"gfile",
             "path":"/bigstore/user_bucket/user_object/params/a"
             })"
        },
        {.test_name = "cns_path_with_ocdbt",
         .directory = "/cns/lu-d/home/user_home/",
         .param_name = "params/a",
         .use_ocdbt = true,
         .process_id = 13,
         .expected_tspec_json_str =
             R"({
              "driver": "ocdbt",
              "base": {
                "driver": "gfile",
                "path": "/cns/lu-d/home/user_home/ocdbt.process_13"
              },
              "path": "params/a",
              "experimental_read_coalescing_threshold_bytes": 1000000,
              "experimental_read_coalescing_merged_bytes": 500000000000,
              "experimental_read_coalescing_interval": "1ms",
              "cache_pool": "cache_pool#ocdbt"
              }  )"
        },
        // END GOOGLE-INTERNAL
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
         })"
        },
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
             })"
        },
    }),
    [](const testing::TestParamInfo<TensorStoreUtilTest::ParamType>& info) {
      return info.param.test_name;
    });

}  // namespace
}  // namespace neptune

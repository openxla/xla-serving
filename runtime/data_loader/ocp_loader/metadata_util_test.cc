#include "third_party/neptune/runtime/data_loader/ocp_loader/metadata_util.h"

#include <string>
#include <vector>

#include "devtools/build/runtime/get_runfiles_dir.h"
#include "file/base/helpers.h"
#include "file/base/options.h"
#include "file/base/path.h"
#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/status_matchers.h"
#include "third_party/json/include/nlohmann/json.hpp"
#include "third_party/json/include/nlohmann/json_fwd.hpp"

namespace neptune {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

constexpr const char* kCheckpointPath =
    "google3/third_party/neptune/runtime/data_loader/ocp_loader/testdata/"
    "checkpoint_with_ocp_metadata";
constexpr const char* kMetadataFileName = "_METADATA";
constexpr const char* kMetadataSkipDeserializeFileName =
    "_METADATA_SKIP_DESERIALIZE";

absl::StatusOr<nlohmann::ordered_json> GetOcpMetadata() {
  std::string _metadata;
  CHECK_OK(file::GetContents(file::JoinPath(devtools_build::GetRunfilesDir(),
                                            kCheckpointPath, kMetadataFileName),
                             &_metadata, file::Defaults()));
  return nlohmann::ordered_json::parse(_metadata);
}

absl::StatusOr<nlohmann::ordered_json> GetOcpMetadataWithSkipDeserialization() {
  std::string _metadata;
  CHECK_OK(file::GetContents(
      file::JoinPath(devtools_build::GetRunfilesDir(), kCheckpointPath,
                     kMetadataSkipDeserializeFileName),
      &_metadata, file::Defaults()));
  return nlohmann::ordered_json::parse(_metadata);
}

TEST(ParamNamesTest, GetParamNames) {
  ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata, GetOcpMetadata());
  ASSERT_OK_AND_ASSIGN(auto result, GetParamNames(metadata));

  EXPECT_THAT(
      result,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));
}

TEST(ParamNamesTest, GetParamNamesWithSkipDeserialize) {
  ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata,
                       GetOcpMetadataWithSkipDeserialization());
  ASSERT_OK_AND_ASSIGN(auto result, GetParamNames(metadata));

  EXPECT_THAT(result,
              ElementsAre("my_param.a/b/c", "my_param.0.a/b", "my_param.0.c/d",
                          "my_param", ".my_param", "Dense_0.bias",
                          "Dense_0.kernel", "Dense_1.bias"));
}

TEST(ParamNamesTest, GetParamNamesForSubTree) {
  ASSERT_OK_AND_ASSIGN(nlohmann::ordered_json metadata, GetOcpMetadata());

  // Test case: empty sub_tree, get all params.
  std::vector<std::string> sub_tree;
  ASSERT_OK_AND_ASSIGN(std::vector<std::string> param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(
      param_params,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));

  // Test case: sub_tree for "Dense_0"
  sub_tree = {"Dense_0"};
  ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params, ElementsAre("Dense_0.bias", "Dense_0.kernel"));

  // Test case: sub_tree for ["my_param", "0"]
  sub_tree = {"my_param", "0"};
  ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params, ElementsAre("my_param.0.a/b", "my_param.0.c/d"));

  // Test case: sub_tree for "my_param"
  sub_tree = {"my_param"};
  ASSERT_OK_AND_ASSIGN(param_params,
                       GetParamNamesForSubTree(metadata, sub_tree));
  EXPECT_THAT(param_params,
              ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                          "my_param.0.c/d", "my_param"));

  // Test case: sub_tree for "Dense", which is invalid and should return error.
  sub_tree = {"Dense"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));

  // Test case: sub_tree for {"Dense_0", "b"}, which is invalid and should
  // return error.
  sub_tree = {"Dense_0", "b"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));

  // Test case: sub_tree for {"Dense_0.bias"}, which is invalid and should
  // return error.
  sub_tree = {"Dense_0.bias"};
  EXPECT_THAT(GetParamNamesForSubTree(metadata, sub_tree),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("No valid param name")));
}

TEST(ParamNamesTest, GetAllWeightNamesFromCheckpoint) {
  ASSERT_OK_AND_ASSIGN(const auto weight_names,
                       GetAllWeightNamesFromCheckpoint(file::JoinPath(
                           devtools_build::GetRunfilesDir(), kCheckpointPath)));
  EXPECT_THAT(
      weight_names,
      ElementsAre("my_param.a.b", "my_param.a/b/c", "my_param.0.a/b",
                  "my_param.0.c/d", "my_param", ".my_param", "Dense_0.bias",
                  "Dense_0.kernel", "Dense_1.bias", "Dense_1.kernel"));
}

}  // namespace
}  // namespace neptune

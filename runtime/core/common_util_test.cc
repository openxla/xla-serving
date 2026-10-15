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

#include "runtime/core/common_util.h"  // from @xla_serving

#include <cstdint>
#include <string>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/constants.h"  // from @xla_serving
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::IsOk;
using ::testing::Not;

class CommonUtilTest : public ::testing::Test {
 protected:
  void SetUp() override {
    tsl::Env* env = tsl::Env::Default();
    std::string temp_dir = testing::TempDir();
    test_dir_ = tsl::io::JoinPath(temp_dir, "common_util_test_dir");
    ASSERT_OK(env->RecursivelyCreateDir(test_dir_));
  }

  void TearDown() override {
    int64_t undeleted_files = 0;
    int64_t undeleted_dirs = 0;
    tsl::Env::Default()
        ->DeleteRecursively(test_dir_, &undeleted_files, &undeleted_dirs)
        .IgnoreError();
  }

  void CreateFile(absl::string_view filename) {
    std::string filepath = tsl::io::JoinPath(test_dir_, filename);
    ASSERT_OK(tsl::WriteStringToFile(tsl::Env::Default(), filepath, "test"));
  }

  std::string test_dir_;
};

TEST_F(CommonUtilTest, IsOrbaxModelDirectoryReturnsErrorForEmptyDirectory) {
  EXPECT_THAT(IsOrbaxModelDirectory(test_dir_), Not(IsOk()));
}

TEST_F(CommonUtilTest, IsOrbaxModelDirectoryRecognizesOrbaxVersionFile) {
  CreateFile(kModelVersionTxt);
  EXPECT_OK(IsOrbaxModelDirectory(test_dir_));
}

TEST_F(CommonUtilTest, IsBundleReturnsTrueForValidOrbaxBundle) {
  CreateFile(kModelVersionTxt);
  CreateFile(kBundleOrchestrationFileName);
  EXPECT_TRUE(IsBundle(test_dir_));
}

TEST_F(CommonUtilTest, IsBundleReturnsFalseForStandaloneModel) {
  CreateFile(kModelVersionTxt);
  CreateFile("manifest.pb");
  EXPECT_FALSE(IsBundle(test_dir_));
}

TEST_F(CommonUtilTest, IsBundleReturnsFalseWhenVersionFileMissing) {
  CreateFile(kBundleOrchestrationFileName);
  EXPECT_FALSE(IsBundle(test_dir_));
}

TEST_F(CommonUtilTest, IsBundleReturnsFalseForNonExistentPath) {
  EXPECT_FALSE(IsBundle(tsl::io::JoinPath(test_dir_, "non_existent_path")));
}

}  // namespace
}  // namespace xla_serving

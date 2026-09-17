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

#include "examples/composite_processor/add_int_from_resource_transform.h"  // from @xla_serving

#include <fstream>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "examples/composite_processor/int_resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune_examples {
namespace composite {
namespace {

using ::absl_testing::StatusIs;

TEST(AddIntFromResourceTransformTest, GetAllResourcesWithNullResource) {
  auto transform_or = AddIntFromResourceTransform::Create(nullptr);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  EXPECT_TRUE(transform->GetAllResources().empty());
}

TEST(AddIntFromResourceTransformTest, GetAllResourcesWithNonNullResource) {
  auto resource = std::make_shared<IntResource>(
      /*id=*/1, "int_resource", std::vector<neptune::data::Asset>{},
      "ExamplesIntResource");
  auto transform_or = AddIntFromResourceTransform::Create(resource);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  auto resources = transform->GetAllResources();
  ASSERT_EQ(resources.size(), 1);
  EXPECT_EQ(resources[0], resource);
}

TEST(AddIntFromResourceTransformTest, ToProtoWithNullResource) {
  auto transform_or = AddIntFromResourceTransform::Create(nullptr);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  auto config_or = transform->ToProto();
  ASSERT_TRUE(config_or.ok());
  auto config = config_or.value();

  EXPECT_EQ(config.op_name(), "AddIntFromResourceTransform");
  EXPECT_TRUE(config.resource_ids().empty());
}

TEST(AddIntFromResourceTransformTest, ToProtoWithNonNullResource) {
  auto resource = std::make_shared<IntResource>(
      /*id=*/42, "int_resource", std::vector<neptune::data::Asset>{},
      "ExamplesIntResource");
  auto transform_or = AddIntFromResourceTransform::Create(resource);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  auto config_or = transform->ToProto();
  ASSERT_TRUE(config_or.ok());
  auto config = config_or.value();

  EXPECT_EQ(config.op_name(), "AddIntFromResourceTransform");
  ASSERT_EQ(config.resource_ids_size(), 1);
  EXPECT_EQ(config.resource_ids(0), 42);
}

TEST(AddIntFromResourceTransformTest, ExecuteWithNullResourceFails) {
  auto transform_or = AddIntFromResourceTransform::Create(nullptr);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  xla::ifrt::Shape shape({2});
  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), shape);
  int* input_data = static_cast<int*>(input.base());
  input_data[0] = 1;
  input_data[1] = 2;

  EXPECT_THAT((*transform)(input),
              StatusIs(absl::StatusCode::kFailedPrecondition,
                       testing::HasSubstr("Resource not loaded")));
}

TEST(AddIntFromResourceTransformTest, ExecuteWithNonNullResourceSucceeds) {
  // Create a temp file with the resource value.
  std::string temp_file =
      absl::StrCat(testing::TempDir(), "/int_resource_value");
  std::ofstream out(temp_file);
  out << "10";
  out.close();

  // Create and load the resource.
  std::vector<neptune::data::Asset> assets = {
      neptune::data::Asset{.name = "val", .path = temp_file}};
  auto resource = std::make_shared<IntResource>(
      /*id=*/1, "int_resource", assets, "ExamplesIntResource");
  ASSERT_TRUE(resource->Load().ok());

  auto transform_or = AddIntFromResourceTransform::Create(resource);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  xla::ifrt::Shape shape({2});
  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32), shape);
  int* input_data = static_cast<int*>(input.base());
  input_data[0] = 1;
  input_data[1] = 2;

  auto output_or = (*transform)(input);
  ASSERT_TRUE(output_or.ok());
  auto output = std::move(output_or).value();

  EXPECT_EQ(output.dtype().kind(), xla::ifrt::DType::Kind::kS32);
  EXPECT_EQ(output.shape(), shape);
  const int* output_data = static_cast<const int*>(output.base());
  EXPECT_EQ(output_data[0], 11);
  EXPECT_EQ(output_data[1], 12);
}

TEST(AddIntFromResourceTransformTest, ExecuteWithWrongDtypeFails) {
  // Create a temp file with the resource value.
  std::string temp_file =
      absl::StrCat(testing::TempDir(), "/int_resource_value_dtype");
  std::ofstream out(temp_file);
  out << "10";
  out.close();

  // Create and load the resource.
  std::vector<neptune::data::Asset> assets = {
      neptune::data::Asset{.name = "val", .path = temp_file}};
  auto resource = std::make_shared<IntResource>(
      /*id=*/1, "int_resource", assets, "ExamplesIntResource");
  ASSERT_TRUE(resource->Load().ok());

  auto transform_or = AddIntFromResourceTransform::Create(resource);
  ASSERT_TRUE(transform_or.ok());
  auto transform = std::move(transform_or).value();

  xla::ifrt::Shape shape({2});
  // Use float instead of int.
  xla_serving::HostBuffer input(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32), shape);
  float* input_data = static_cast<float*>(input.base());
  input_data[0] = 1.0f;
  input_data[1] = 2.0f;

  EXPECT_THAT((*transform)(input),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       testing::HasSubstr("Input buffer must have S32 dtype")));
}

}  // namespace
}  // namespace composite
}  // namespace neptune_examples

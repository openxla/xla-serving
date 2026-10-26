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

#include "runtime/tensorflow/utils/tf_host_buffer_converter.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_converter.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_testutil.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace xla_serving {
namespace {

using ::absl_testing::IsOk;
using ::absl_testing::StatusIs;
using ::tensorflow::test::AsTensor;
using ::testing::AllOf;
using ::testing::HasSubstr;
using ::testing::NotNull;

TEST(TfHostBufferConverterTest, ConvertsTensorWithoutCopy) {
  tensorflow::Tensor tensor =
      AsTensor<float>({1.0, 2.0, 3.0, 4.0}, tensorflow::TensorShape({2, 2}));
  const void* const data = tensor.data();

  absl::StatusOr<HostBuffer> result = ConvertTfTensorToHostBuffer(tensor);

  ASSERT_THAT(result, IsOk());
  EXPECT_EQ(result->base(), data);
  EXPECT_THAT(*result, HostBufferEq(AsHostBuffer<float>(
                           {{1.0, 2.0, 3.0, 4.0}}, xla::ifrt::Shape({2, 2}))));
}

TEST(TfHostBufferConverterTest, HostBufferKeepsTensorMemoryAlive) {
  absl::StatusOr<HostBuffer> result = absl::UnknownError("unset");
  {
    tensorflow::Tensor tensor =
        AsTensor<float>({5.0, 6.0}, tensorflow::TensorShape({2}));
    result = ConvertTfTensorToHostBuffer(tensor);
  }
  ASSERT_THAT(result, IsOk());
  EXPECT_THAT(*result, HostBufferEq(AsHostBuffer<float>(
                           {{5.0, 6.0}}, xla::ifrt::Shape({2}))));
}

TEST(TfHostBufferConverterTest, UnsupportedDtypeFails) {
  tensorflow::Tensor tensor(tensorflow::DT_RESOURCE,
                            tensorflow::TensorShape({}));
  EXPECT_FALSE(ConvertTfTensorToHostBuffer(tensor).ok());
}

// Linking the library registers the converter globally, for
// `tensorflow::Tensor` only.
TEST(TfHostBufferConverterTest, IsRegisteredGlobally) {
  tensorflow::Tensor tensor =
      AsTensor<float>({7.0}, tensorflow::TensorShape({1}));
  std::shared_ptr<const HostBufferConverter> converter =
      FindHostBufferConverter(Value(tensor));
  ASSERT_THAT(converter, NotNull());
  EXPECT_EQ(converter->name(), kTfHostBufferConverterName);

  absl::StatusOr<std::optional<HostBuffer>> result =
      ConvertToHostBuffer(Value(tensor));
  ASSERT_THAT(result, IsOk());
  ASSERT_TRUE(result->has_value());
  EXPECT_EQ((*result)->base(), tensor.data());

  EXPECT_EQ(FindHostBufferConverter(Value(42)), nullptr);
}

TEST(TfHostBufferConverterTest, DuplicateTensorRegistrationIsRejected) {
  EXPECT_THAT(
      RegisterHostBufferConverter<tensorflow::Tensor>(
          "another_tensorflow", ConvertTfTensorToHostBuffer),
      StatusIs(absl::StatusCode::kAlreadyExists,
               AllOf(HasSubstr("'tensorflow'"),
                     HasSubstr("refusing to register 'another_tensorflow'"))));
}

}  // namespace
}  // namespace xla_serving

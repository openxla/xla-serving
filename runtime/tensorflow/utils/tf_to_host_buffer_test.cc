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

#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @xla_serving

#include <sys/stat.h>

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/kernel/kernel_io_data_type.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/test.h"  // from @xla
#include "xla/tsl/platform/types.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_matcher.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_testutil.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tensorflow/core/platform/tstring.h"  // from @org_tensorflow
#include "tsl/platform/ml_dtypes.h"  // from @tsl

namespace xla_serving {
namespace {
using tensorflow::TensorShape;
using tensorflow::test::AsScalar;
using tensorflow::test::AsTensor;
using tensorflow::test::TensorEq;

using Tf2HostBufferTest = ::testing::TestWithParam<tensorflow::Tensor>;
using HostBuffer2TfTest = ::testing::TestWithParam<HostBuffer>;

// Wrapper function to build int4 tensor
tensorflow::Tensor AsInt4Tensor(absl::Span<const int> values,
                                const TensorShape& shape) {
  tensorflow::Tensor tensor(tensorflow::DT_INT4, shape);
  for (int i = 0; i < values.size(); ++i) {
    tensor.flat<tsl::int4>()(i) = static_cast<tsl::int4>(values[i]);
  }
  return tensor;
}

// Test round trip conversion: HostBuffer -> TF Tensor -> HostBuffer
TEST_P(HostBuffer2TfTest, RoundTrip) {
  const HostBuffer& in_host_buffer = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto out_tensor, ToTFTensor(in_host_buffer));
  TF_ASSERT_OK_AND_ASSIGN(auto out_host_buffer, ToHostBuffer(out_tensor));
  EXPECT_THAT(out_host_buffer, HostBufferEq(in_host_buffer));
}

INSTANTIATE_TEST_SUITE_P(
    HostBuffer2TfTests, HostBuffer2TfTest,
    ::testing::ValuesIn<HostBuffer>({
        {
            AsHostBuffer<int32_t>(1),
        },
        {
            AsHostBuffer<int32_t>({1}, xla::ifrt::Shape({1})),
        },
        {AsHostBuffer<uint32_t>({1, 2, 3}, xla::ifrt::Shape({3}))},
        {
            AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({4})),
        },
        {
            AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})),
        },
        {
            AsHostBuffer<tsl::int4>(
                {
                    tsl::int4(1),
                    tsl::int4(2),
                },
                xla::ifrt::Shape({2, 1})),
        },
    }));

TEST_P(Tf2HostBufferTest, RoundTrip) {
  const tensorflow::Tensor& input_tensor = GetParam();
  TF_ASSERT_OK_AND_ASSIGN(auto host_buffer, ToHostBuffer(input_tensor));
  TF_ASSERT_OK_AND_ASSIGN(auto out_tensor, ToTFTensor(host_buffer));
  EXPECT_THAT(out_tensor, TensorEq(input_tensor));
}

INSTANTIATE_TEST_SUITE_P(
    Tf2HostBufferTests, Tf2HostBufferTest,
    ::testing::ValuesIn<tensorflow::Tensor>({
        {
            AsScalar<int32_t>(1),
        },
        {AsTensor<uint32_t>({1, 2, 3}, TensorShape({3}))},
        {
            AsInt4Tensor({1, 2, 3, 4}, TensorShape({4})),
        },
        {
            AsInt4Tensor({1, 2, 3, 4}, TensorShape({2, 2})),
        },
        {
            AsInt4Tensor({1, 2, 3, 4, 5, 6, 7, 8}, TensorShape({4, 2})),
        },
    }));

TEST(HostBufferStringTest, RoundTripOneDimension) {
  std::vector<tensorflow::tstring> strings = {"aaa", "b", "ccccc", "dd"};
  tensorflow::Tensor expected_tensor =
      AsTensor<tensorflow::tstring>(strings, {4});
  // ASSERT_TRUE(expected_tensor.FromProto(tensor_proto));
  TF_ASSERT_OK_AND_ASSIGN(HostBuffer host_buffer,
                          ToHostBuffer(expected_tensor));
  for (int i = 0; i < strings.size(); ++i) {
    EXPECT_EQ(host_buffer.flat<tensorflow::tstring>()(i), strings[i]);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto tensor, ToTFTensor(host_buffer));
  EXPECT_THAT(tensor, TensorEq(expected_tensor));
  EXPECT_EQ(host_buffer.base(), tensor.data());
}

TEST(HostBufferStringTest, RoundTripMultiDimensionWithConcat) {
  std::vector<tensorflow::tstring> strings_to_concat;

  for (int i = 0; i < 8; ++i) {
    strings_to_concat.push_back(absl::StrCat("test_string_", i % 8));
  }
  std::vector<tensorflow::tstring> strings_concated;
  for (int i = 0; i < 16; ++i) {
    strings_concated.push_back(absl::StrCat("test_string_", i % 8));
  }
  tensorflow::Tensor tensor =
      AsTensor<tensorflow::tstring>(strings_to_concat, {2, 2, 2});
  tensorflow::Tensor expected_tensor =
      AsTensor<tensorflow::tstring>(strings_concated, {4, 2, 2});
  TF_ASSERT_OK_AND_ASSIGN(HostBuffer host_buffer, ToHostBuffer(tensor));
  ASSERT_EQ(host_buffer.shape().num_elements(), tensor.NumElements());
  for (int i = 0; i < tensor.NumElements(); ++i) {
    EXPECT_EQ(host_buffer.flat<tensorflow::tstring>()(i), strings_to_concat[i]);
  }
  TF_ASSERT_OK_AND_ASSIGN(auto output,
                          xla_serving::Concat(nullptr, {host_buffer, host_buffer},
                                      DefaultAllocator::GetInstance()));
  TF_ASSERT_OK_AND_ASSIGN(tensor, ToTFTensor(output));
  EXPECT_THAT(tensor, TensorEq(expected_tensor));
}

TEST(ToJsvValueTest, ValueIsTensor) {
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(Value value,
                          ToJsvValue(tensor, KernelIODataType::kTensor));
  EXPECT_TRUE(value.IsType<tensorflow::Tensor>());
}

TEST(ToJsvValueTest, ValueIsHostBuffer) {
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT, {2, 2});
  TF_ASSERT_OK_AND_ASSIGN(Value value,
                          ToJsvValue(tensor, KernelIODataType::kHostBuffer));
  EXPECT_TRUE(value.IsType<HostBuffer>());
}

}  // namespace
}  // namespace xla_serving

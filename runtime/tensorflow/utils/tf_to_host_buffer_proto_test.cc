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

#include "runtime/tensorflow/utils/tf_to_host_buffer_proto.h"  // from @xla_serving

#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "protos/grpc/host_buffer.pb.h"  // from @xla_serving
#include "protos/grpc/model.pb.h"  // from @xla_serving
#include "protos/grpc/predict.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/util/proto/proto_matchers.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::IsOkAndHolds;
using ::tsl::proto_testing::EqualsProto;

TEST(TfToHostBufferProtoTest, FromTfTensorProtoConvertsFloatTensor) {
  // Create a float tensor.
  tensorflow::Tensor tensor(tensorflow::DT_FLOAT,
                            tensorflow::TensorShape({2, 3}));
  auto flat = tensor.flat<float>();
  for (int i = 0; i < 6; ++i) {
    flat(i) = static_cast<float>(i) * 0.5f;
  }
  tensorflow::TensorProto tf_tensor_proto;
  tensor.AsProtoField(&tf_tensor_proto);

  // Expected xla_serving::HostBufferProto.
  xla_serving::HostBuffer expected_host_buffer(
      xla::ifrt::DType(xla::ifrt::DType::Kind::kF32), xla::ifrt::Shape({2, 3}));
  auto host_buffer_flat = expected_host_buffer.flat<float>();
  for (int i = 0; i < 6; ++i) {
    host_buffer_flat(i) = static_cast<float>(i) * 0.5f;
  }
  xla_serving::HostBufferProto expected_host_buffer_proto =
      xla_serving::HostBuffer::ToHostBufferProto(std::move(expected_host_buffer));

  EXPECT_THAT(FromTfTensorProto(tf_tensor_proto),
              IsOkAndHolds(EqualsProto(expected_host_buffer_proto)));
}

TEST(TfToHostBufferProtoTest, FromTfTensorProtoConvertsStringTensor) {
  // Create a string tensor.
  tensorflow::Tensor tensor(tensorflow::DT_STRING,
                            tensorflow::TensorShape({5}));
  auto flat = tensor.flat<tsl::tstring>();
  flat(0) = "hello";
  flat(1) = "world";
  flat(2) = "test";
  flat(3) = "test2";
  flat(4) = "test3";
  tensorflow::TensorProto tf_tensor_proto;
  tensor.AsProtoField(&tf_tensor_proto);

  // Expected xla_serving::HostBufferProto.
  xla_serving::HostBufferProto expected_host_buffer_proto;
  *expected_host_buffer_proto.mutable_dtype() =
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString).ToProto();
  expected_host_buffer_proto.mutable_shape()->add_dims(5);
  expected_host_buffer_proto.mutable_string_data()->add_elements("hello");
  expected_host_buffer_proto.mutable_string_data()->add_elements("world");
  expected_host_buffer_proto.mutable_string_data()->add_elements("test");
  expected_host_buffer_proto.mutable_string_data()->add_elements("test2");
  expected_host_buffer_proto.mutable_string_data()->add_elements("test3");

  EXPECT_THAT(FromTfTensorProto(tf_tensor_proto),
              IsOkAndHolds(EqualsProto(expected_host_buffer_proto)));
}

}  // namespace
}  // namespace xla_serving

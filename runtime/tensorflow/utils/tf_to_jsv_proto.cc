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

#include "runtime/tensorflow/utils/tf_to_jsv_proto.h"  // from @xla_serving

#include <string>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "protos/grpc/host_buffer.pb.h"  // from @xla_serving
#include "protos/grpc/model.pb.h"  // from @xla_serving
#include "protos/grpc/predict.pb.h"  // from @xla_serving
#include "runtime/tensorflow/utils/ifrt_tensor_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.pb.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {

absl::StatusOr<xla_serving::HostBufferProto> FromTfTensorProto(
    const tensorflow::TensorProto& tf_tensor_proto) {
  xla_serving::HostBufferProto jsv_proto;

  *jsv_proto.mutable_shape() =
      xla_serving::ToIfrtShape(tf_tensor_proto.tensor_shape()).ToProto();
  TF_ASSIGN_OR_RETURN(auto dtype, xla_serving::ToIfrtDType(tf_tensor_proto.dtype()));
  *jsv_proto.mutable_dtype() = dtype.ToProto();

  // Tensors may either be stored in the tensor_content field (raw bytes) or in
  // the type-specific *_val fields (e.g., float_val, int_val). To avoid
  // duplicating conversion logic, we can convert a tensor_proto to a Tensor
  // object and then use ToHostBuffer to convert it to a HostBufferProto.
  tensorflow::Tensor tensor;
  if (!tensor.FromProto(tf_tensor_proto)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse tensor proto: ", tf_tensor_proto));
  }

  if (tf_tensor_proto.dtype() == tensorflow::DT_STRING) {
    auto tstrings = tensor.unaligned_flat<tsl::tstring>();
    for (int i = 0; i < tensor.NumElements(); ++i) {
      jsv_proto.mutable_string_data()->add_elements(std::string(tstrings(i)));
    }
  } else {
    jsv_proto.set_raw_data(tensor.tensor_data());
  }
  return std::move(jsv_proto);
}

}  // namespace xla_serving

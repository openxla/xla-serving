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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_JSV_PROTO_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_JSV_PROTO_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "protos/grpc/host_buffer.pb.h"  // from @neptune
#include "protos/grpc/predict.pb.h"  // from @neptune
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow

namespace xla_serving {

absl::StatusOr<xla_serving::HostBufferProto> FromTfTensorProto(
    const tensorflow::TensorProto& tf_tensor_proto);

}

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_JSV_PROTO_H_

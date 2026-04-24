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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_HOST_BUFFER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_HOST_BUFFER_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace neptune {

// Converts a Tensor to a HostBuffer.
absl::StatusOr<HostBuffer> ToHostBuffer(tensorflow::Tensor tensor);

// Converts a HostBuffer to a Tensor.
absl::StatusOr<tensorflow::Tensor> ToTFTensor(const HostBuffer& host_buffer);

// Converts a Tensor to a Value with the given data type.
absl::StatusOr<Value> ToJsvValue(tensorflow::Tensor tensor,
                                 KernelIODataType data_type);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_UTILS_TF_TO_HOST_BUFFER_H_

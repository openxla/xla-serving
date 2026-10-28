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

#ifndef XLA_SERVING_RUNTIME_TENSORFLOW_UTILS_TF_HOST_BUFFER_CONVERTER_H_
#define XLA_SERVING_RUNTIME_TENSORFLOW_UTILS_TF_HOST_BUFFER_CONVERTER_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow

namespace xla_serving {

// The name of the `tensorflow::Tensor` converter in the global
// `HostBufferConverterRegistry`.
inline constexpr char kTfHostBufferConverterName[] = "tensorflow";

// Converts `tensor` to a `HostBuffer` that shares the tensor's memory (no
// copy) and keeps the tensor alive.
//
// Linking this library registers it in the global `HostBufferConverterRegistry`
// for `tensorflow::Tensor` at static initialization, so kernels such as the
// StableHLO runner accept `tensorflow::Tensor` inputs.
absl::StatusOr<HostBuffer> ConvertTfTensorToHostBuffer(
    const tensorflow::Tensor& tensor);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_TENSORFLOW_UTILS_TF_HOST_BUFFER_CONVERTER_H_

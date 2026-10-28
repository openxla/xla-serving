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

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_converter.h"  // from @xla_serving
#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @xla_serving
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow

namespace xla_serving {

absl::StatusOr<HostBuffer> ConvertTfTensorToHostBuffer(
    const tensorflow::Tensor& tensor) {
  // Copying a tensor shares its buffer; `ToHostBuffer` wraps that buffer
  // without copying it, and the `HostBuffer` owns the tensor copy.
  return ToHostBuffer(tensor);
}

namespace {

[[maybe_unused]] const bool kTfHostBufferConverterRegistered =
    RegisterHostBufferConverterOrDie<tensorflow::Tensor>(
        kTfHostBufferConverterName, ConvertTfTensorToHostBuffer);

}  // namespace
}  // namespace xla_serving

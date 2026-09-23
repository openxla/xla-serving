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

#include "data/processor/python/examples/multi_kernel_math_transform.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_type_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace xla_serving::data {
namespace composite {

template <typename T>
absl::StatusOr<std::unique_ptr<MultiKernelMathTransform<T>>>
MultiKernelMathTransform<T>::Create() {
  return std::make_unique<MultiKernelMathTransform<T>>();
}

template <typename T>
absl::StatusOr<xla_serving::HostBuffer> MultiKernelMathTransform<T>::operator()(
    xla_serving::HostBuffer in_buffer) {
  if (in_buffer.dtype().kind() != xla_serving::NativeAsDTypeKind<T>::value) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input buffer dtype mismatch. Expected: ",
                     xla::ifrt::DType(xla_serving::NativeAsDTypeKind<T>::value),
                     ", Got: ", in_buffer.dtype()));
  }
  xla_serving::HostBuffer out(in_buffer.dtype(), in_buffer.shape());
  const T* in_data = static_cast<const T*>(in_buffer.base());
  T* out_data = static_cast<T*>(out.base());
  for (size_t i = 0; i < out.shape().num_elements(); ++i) {
    out_data[i] = in_data[i] + static_cast<T>(10);
  }
  return out;
}

template <typename T>
absl::StatusOr<TransformConfig> MultiKernelMathTransform<T>::ToProto() const {
  TransformConfig config;
  config.set_op_name("MultiKernelMathTransform");
  return config;
}

template <typename T>
absl::Status MultiKernelMathTransform<T>::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 1 || !context.inputs[0].IsSpec()) {
    return absl::InvalidArgumentError("Expected 1 tensor spec input");
  }
  context.outputs.push_back(context.inputs[0]);
  return absl::OkStatus();
}

template class MultiKernelMathTransform<int32_t>;
template class MultiKernelMathTransform<float>;

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<int32_t>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<xla_serving::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<int32_t>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kS32}}});

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<float>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<xla_serving::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<float>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kF32}}});

}  // namespace composite
}  // namespace xla_serving::data

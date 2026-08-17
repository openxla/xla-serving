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

#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MULTI_KERNEL_MATH_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MULTI_KERNEL_MATH_TRANSFORM_H_

#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune::data {
namespace composite {

template <typename T>
class MultiKernelMathTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MultiKernelMathTransform<T>>> Create();

  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer in_buffer);

  absl::StatusOr<TransformConfig> ToProto() const override;

  absl::string_view DebugString() const override {
    return "MultiKernelMathTransform";
  }

 protected:
  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_MULTI_KERNEL_MATH_TRANSFORM_H_

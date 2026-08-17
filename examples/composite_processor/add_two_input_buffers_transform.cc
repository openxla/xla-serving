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

#include "examples/composite_processor/add_two_input_buffers_transform.h"  // from @neptune

#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace neptune_examples {
namespace composite {

absl::StatusOr<std::unique_ptr<AddTwoInputBuffersTransform>>
AddTwoInputBuffersTransform::Create() {
  return std::make_unique<AddTwoInputBuffersTransform>();
}

absl::StatusOr<neptune::HostBuffer> AddTwoInputBuffersTransform::operator()(
    neptune::HostBuffer in_buffer1, neptune::HostBuffer in_buffer2) {
  if (in_buffer1.shape() != in_buffer2.shape()) {
    return absl::InvalidArgumentError("Input buffers must have the same shape");
  }
  if (in_buffer1.dtype() != in_buffer2.dtype() ||
      in_buffer1.dtype().kind() != xla::ifrt::DType::kS32) {
    return absl::InvalidArgumentError("Input buffers must have the same dtype");
  }
  neptune::HostBuffer out_buffer =
      neptune::HostBuffer(in_buffer1.dtype(), in_buffer1.shape());

  int64_t num_elements = in_buffer1.shape().num_elements();
  const int* data1 = static_cast<const int*>(in_buffer1.base());
  const int* data2 = static_cast<const int*>(in_buffer2.base());
  int* out_data = static_cast<int*>(out_buffer.base());
  for (int64_t i = 0; i < num_elements; ++i) {
    out_data[i] = data1[i] + data2[i];
  }
  return out_buffer;
}

absl::StatusOr<neptune::data::composite::TransformConfig>
AddTwoInputBuffersTransform::ToProto() const {
  neptune::data::composite::TransformConfig config;
  config.set_op_name("AddTwoInputBuffersTransform");
  return config;
}

ND_REGISTER_TRANSFORM(AddTwoInputBuffersTransform,
                      "AddTwoInputBuffersTransform")
    .SetDeserializer(
        [](const neptune::data::composite::TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<
                std::unique_ptr<neptune::data::composite::TransformBase>> {
          return std::make_unique<AddTwoInputBuffersTransform>();
        });

}  // namespace composite
}  // namespace neptune_examples

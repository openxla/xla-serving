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

#ifndef XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_HOST_BUFFER_TRANSFORM_H_
#define XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_HOST_BUFFER_TRANSFORM_H_

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {

class AddHostBufferTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddHostBufferTransform>> Create();
  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer in_buffer1,
                                             xla_serving::HostBuffer in_buffer2);
  absl::StatusOr<TransformConfig> ToProto() const override;
  absl::string_view DebugString() const override {
    return "AddHostBufferTransform";
  }
};

class AddHostBufferVectorTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddHostBufferVectorTransform>> Create();
  absl::StatusOr<xla_serving::HostBuffer> operator()(
      std::vector<xla_serving::HostBuffer> in_buffers);
  absl::StatusOr<TransformConfig> ToProto() const override;
  absl::string_view DebugString() const override {
    return "AddHostBufferVectorTransform";
  }
};

}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_PYTHON_EXAMPLES_ADD_HOST_BUFFER_TRANSFORM_H_

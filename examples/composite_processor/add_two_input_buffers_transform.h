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

#ifndef XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_
#define XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace neptune_examples {
namespace composite {

class AddTwoInputBuffersTransform
    : public xla_serving::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddTwoInputBuffersTransform>> Create();
  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer in_buffer1,
                                             xla_serving::HostBuffer in_buffer2);
  absl::StatusOr<xla_serving::data::composite::TransformConfig> ToProto()
      const override;
};

}  // namespace composite
}  // namespace neptune_examples

#endif  // XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_TWO_INPUT_BUFFERS_TRANSFORM_H_

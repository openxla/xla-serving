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

#ifndef XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_
#define XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "examples/composite_processor/int_resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace neptune_examples {
namespace composite {

class AddIntFromResourceTransform
    : public xla_serving::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<AddIntFromResourceTransform>> Create(
      std::shared_ptr<const IntResource> resource);
  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer in_buffer);
  absl::StatusOr<xla_serving::data::composite::TransformConfig> ToProto()
      const override;
  std::vector<std::shared_ptr<xla_serving::data::Resource>> GetAllResources()
      const override;

 private:
  explicit AddIntFromResourceTransform(
      std::shared_ptr<const IntResource> resource);
  std::shared_ptr<const IntResource> resource_;
};

}  // namespace composite
}  // namespace neptune_examples

#endif  // XLA_SERVING_EXAMPLES_COMPOSITE_PROCESSOR_ADD_INT_FROM_RESOURCE_TRANSFORM_H_

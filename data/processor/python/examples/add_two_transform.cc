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

#include "data/processor/python/examples/add_two_transform.h"  // from @xla_serving

#include <memory>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace neptune::data {
namespace composite {

absl::StatusOr<std::unique_ptr<AddTwoTransform>> AddTwoTransform::Create() {
  return std::make_unique<AddTwoTransform>();
}

int AddTwoTransform::operator()(int a) { return a + 2; }

absl::StatusOr<TransformConfig> AddTwoTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("AddTwoTransform");
  return config;
}

ND_REGISTER_TRANSFORM(AddTwoTransform, "AddTwoTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<AddTwoTransform>();
        });

}  // namespace composite
}  // namespace neptune::data

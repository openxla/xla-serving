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

#include "data/processor/python/examples/make_dict_transform.h"  // from @xla_serving

#include <memory>
#include <string>
#include <unordered_map>
#include <utility>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {

absl::StatusOr<std::unique_ptr<MakeDictTransform>> MakeDictTransform::Create() {
  return std::make_unique<MakeDictTransform>();
}

std::unordered_map<std::string, int> MakeDictTransform::operator()(
    std::string s, int i) {
  return {{std::move(s), i}};
}

absl::StatusOr<TransformConfig> MakeDictTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("MakeDictTransform");
  return config;
}

ND_REGISTER_TRANSFORM(MakeDictTransform, "MakeDictTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<xla_serving::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<MakeDictTransform>();
        });

}  // namespace composite
}  // namespace xla_serving::data

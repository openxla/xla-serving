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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_TYPE_CONVERTER_REGISTRY_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_TYPE_CONVERTER_REGISTRY_H_

#include <memory>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_converter.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving

namespace neptune::data {
namespace composite {
namespace core {

// Registry for TypeConverters.
//
// Allows registering converters between types and retrieving them at runtime.
class TypeConverterRegistry {
 public:
  static TypeConverterRegistry& Instance();

  // Registers a converter for InputType -> OutputType.
  // Returns an error if a converter for this pair is already registered.
  absl::Status RegisterConverter(
      std::unique_ptr<api::TypeConverterBase> converter)
      ABSL_LOCKS_EXCLUDED(type_converter_registry_mu_);

  // Looks up a converter for the given types in the registry. Returns an
  // error if not found.
  absl::StatusOr<const api::TypeConverterBase*> LookupConverter(
      api::TypeInfo input_type, api::TypeInfo output_type) const
      ABSL_LOCKS_EXCLUDED(type_converter_registry_mu_);

 private:
  struct TypePair {
    api::TypeInfo input_type;
    api::TypeInfo output_type;

    bool operator==(const TypePair& other) const {
      return input_type == other.input_type && output_type == other.output_type;
    }

    template <typename H>
    friend H AbslHashValue(H h, const TypePair& p) {
      return H::combine(std::move(h), p.input_type, p.output_type);
    }
  };

  mutable absl::Mutex type_converter_registry_mu_;
  absl::flat_hash_map<TypePair, std::unique_ptr<api::TypeConverterBase>>
      registry_ ABSL_GUARDED_BY(type_converter_registry_mu_);
};

}  // namespace core
}  // namespace composite
}  // namespace neptune::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_TYPE_CONVERTER_REGISTRY_H_

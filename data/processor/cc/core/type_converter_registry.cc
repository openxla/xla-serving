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

#include "data/processor/cc/core/type_converter_registry.h"  // from @xla_serving

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_converter.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

TypeConverterRegistry& TypeConverterRegistry::Instance() {
  static absl::NoDestructor<TypeConverterRegistry> instance;
  return *instance;
}

absl::Status TypeConverterRegistry::RegisterConverter(
    std::unique_ptr<api::TypeConverterBase> converter) {
  absl::MutexLock lock(type_converter_registry_mu_);
  TypePair key{converter->InputType(), converter->OutputType()};
  std::unique_ptr<api::TypeConverterBase>& existing_converter = registry_[key];
  if (existing_converter != nullptr) {
    return absl::AlreadyExistsError(absl::StrCat(
        "api::TypeConverter for ", key.input_type.DebugTypeName(), " -> ",
        key.output_type.DebugTypeName(), " already registered"));
  }
  existing_converter = std::move(converter);
  return absl::OkStatus();
}

absl::StatusOr<const api::TypeConverterBase*>
TypeConverterRegistry::LookupConverter(api::TypeInfo input_type,
                                       api::TypeInfo output_type) const {
  absl::MutexLock lock(type_converter_registry_mu_);
  auto it = registry_.find({input_type, output_type});
  if (it == registry_.end()) {
    return absl::NotFoundError(
        absl::StrCat("api::TypeConverter for ", input_type.DebugTypeName(),
                     " -> ", output_type.DebugTypeName(), " not found"));
  }
  return it->second.get();
}

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

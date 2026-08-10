#include "data/processor/cc/core/type_converter_registry.h"  // from @neptune

#include <memory>
#include <utility>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/processor/cc/api/type_converter.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune

namespace neptune::data {
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
}  // namespace neptune::data

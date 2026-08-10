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

#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_INFO_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_INFO_H_

#include <typeinfo>
#include <utility>

#include "absl/base/fast_type_id.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace neptune::data {
namespace composite {
namespace api {

class TypeInfo;

template <typename T>
TypeInfo CreateTypeInfo();

class TypeInfo {
 public:
  // Returns the type id.
  absl::FastTypeIdType type_id() const { return type_id_; }

  // Returns the type name as a string view for debugging purposes.
  // This should not be used in performance-critical paths, as it incurs
  // overhead due to parsing the string to get its length.
  absl::string_view DebugTypeName() const {
    return type_name_ != nullptr ? type_name_ : kEmptyTypeName;
  }

  TypeInfo() = default;
  TypeInfo(absl::FastTypeIdType type_id, const char* type_name)
      : type_id_(type_id), type_name_(type_name) {}

  bool operator==(const TypeInfo& other) const {
    if (type_id_ == other.type_id_) return true;
    return DebugTypeName() == other.DebugTypeName();
  }

  bool operator!=(const TypeInfo& other) const { return !(*this == other); }

  template <typename T>
  bool IsType() const {
    return *this == CreateTypeInfo<T>();
  }

  template <typename H>
  friend H AbslHashValue(H h, const TypeInfo& t) {
    return H::combine(std::move(h), t.DebugTypeName());
  }

 private:
  absl::FastTypeIdType type_id_;

  // Pointer to the platform-dependent debug string representation of the type.
  // We use char* instead of std::string_view to avoid the initial overhead of
  // parsing the string to get the length.
  const char* type_name_ = nullptr;

  static constexpr absl::string_view kEmptyTypeName = "(typename unavailable)";
};

template <typename T>
TypeInfo CreateTypeInfo() {
  return TypeInfo(absl::FastTypeId<T>(), typeid(T).name());
}

}  // namespace api
}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_API_TYPE_INFO_H_

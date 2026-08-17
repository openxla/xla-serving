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

#include "data/processor/cc/utils/default_value_utils.h"  // from @neptune

#include <cstdint>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/processor/cc/utils/default_value.pb.h"  // from @neptune
#include "google/protobuf/descriptor.h"
#include "tsl/platform/tstring.h"  // from @tsl

namespace neptune::data {
namespace composite {
namespace utils {

namespace {

absl::string_view CaseName(DefaultValue::ValueCase value_case) {
  if (value_case == DefaultValue::VALUE_NOT_SET) {
    return "VALUE_NOT_SET";
  }
  const google::protobuf::Descriptor* descriptor = DefaultValue::descriptor();
  const google::protobuf::FieldDescriptor* field_desc =
      descriptor->FindFieldByNumber(value_case);
  return field_desc ? field_desc->name() : "unknown";
}

}  // namespace

// Helper to get default values for each data type.
#define DEFAULT_VALUE(type, value) \
  template <>                      \
  type GetDefaultValue<type>() {   \
    return value;                  \
  }

DEFAULT_VALUE(int32_t, 0)
DEFAULT_VALUE(int64_t, 0)
DEFAULT_VALUE(uint32_t, 0)
DEFAULT_VALUE(uint64_t, 0)
DEFAULT_VALUE(float, 0.0f)
DEFAULT_VALUE(double, 0.0)
DEFAULT_VALUE(bool, false)
DEFAULT_VALUE(std::string, "")
DEFAULT_VALUE(tsl::tstring, "")
#undef DEFAULT_VALUE

#define REGISTER_UNPACK(TYPE, CASE, GETTER)                                    \
  template <>                                                                  \
  absl::StatusOr<TYPE> UnpackDefaultValue<TYPE>(const DefaultValue& proto) {   \
    if (proto.value_case() != DefaultValue::CASE) {                            \
      return absl::InvalidArgumentError(absl::StrCat(                          \
          "Type mismatch for expected type: " #TYPE ". Expected case: " #CASE, \
          ", but got actual case: ", CaseName(proto.value_case())));           \
    }                                                                          \
    return static_cast<TYPE>(proto.GETTER());                                  \
  }

REGISTER_UNPACK(int32_t, kIntValue, int_value)
REGISTER_UNPACK(int64_t, kInt64Value, int64_value)
REGISTER_UNPACK(uint32_t, kIntValue, int_value)
REGISTER_UNPACK(uint64_t, kInt64Value, int64_value)
REGISTER_UNPACK(float, kFloatValue, float_value)
REGISTER_UNPACK(double, kDoubleValue, double_value)
REGISTER_UNPACK(bool, kBoolValue, bool_value)

#undef REGISTER_UNPACK

template <>
absl::StatusOr<std::string> UnpackDefaultValue<std::string>(
    const DefaultValue& proto) {
  if (proto.value_case() == DefaultValue::kStringValue) {
    return std::string(proto.string_value());
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Type mismatch for expected type: std::string. Expected case: "
      "kStringValue, but got actual case: ",
      CaseName(proto.value_case())));
}

template <>
absl::StatusOr<tsl::tstring> UnpackDefaultValue<tsl::tstring>(
    const DefaultValue& proto) {
  if (proto.value_case() == DefaultValue::kStringValue) {
    return tsl::tstring(proto.string_value());
  }
  return absl::InvalidArgumentError(absl::StrCat(
      "Type mismatch for expected type: tsl::tstring. Expected case: "
      "kStringValue, but got actual case: ",
      CaseName(proto.value_case())));
}

}  // namespace utils
}  // namespace composite
}  // namespace neptune::data

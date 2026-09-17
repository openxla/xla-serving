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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_UTILS_DEFAULT_VALUE_UTILS_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_UTILS_DEFAULT_VALUE_UTILS_H_

#include <cstdint>
#include <string>
#include <type_traits>

#include "google/protobuf/any.pb.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "data/processor/cc/utils/default_value.pb.h"  // from @xla_serving
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving::data {
namespace composite {
namespace utils {

// Returns the default value for a given type T.
template <typename T>
T GetDefaultValue();

template <>
int32_t GetDefaultValue<int32_t>();
template <>
int64_t GetDefaultValue<int64_t>();
template <>
uint32_t GetDefaultValue<uint32_t>();
template <>
uint64_t GetDefaultValue<uint64_t>();
template <>
float GetDefaultValue<float>();
template <>
double GetDefaultValue<double>();
template <>
bool GetDefaultValue<bool>();
template <>
std::string GetDefaultValue<std::string>();
template <>
tsl::tstring GetDefaultValue<tsl::tstring>();

// Unpacks a DefaultValue proto into a value of type T.
template <typename T>
absl::StatusOr<T> UnpackDefaultValue(const DefaultValue& proto);

template <>
absl::StatusOr<int32_t> UnpackDefaultValue<int32_t>(const DefaultValue& proto);
template <>
absl::StatusOr<int64_t> UnpackDefaultValue<int64_t>(const DefaultValue& proto);
template <>
absl::StatusOr<uint32_t> UnpackDefaultValue<uint32_t>(
    const DefaultValue& proto);
template <>
absl::StatusOr<uint64_t> UnpackDefaultValue<uint64_t>(
    const DefaultValue& proto);
template <>
absl::StatusOr<float> UnpackDefaultValue<float>(const DefaultValue& proto);
template <>
absl::StatusOr<double> UnpackDefaultValue<double>(const DefaultValue& proto);
template <>
absl::StatusOr<bool> UnpackDefaultValue<bool>(const DefaultValue& proto);
template <>
absl::StatusOr<std::string> UnpackDefaultValue<std::string>(
    const DefaultValue& proto);
template <>
absl::StatusOr<tsl::tstring> UnpackDefaultValue<tsl::tstring>(
    const DefaultValue& proto);

// Unpacks a google::protobuf::Any containing a DefaultValue proto
// into a value of type T.
template <typename T>
absl::StatusOr<T> UnpackAny(const google::protobuf::Any& any) {
  DefaultValue proto;
  if (!any.UnpackTo(&proto)) {
    return absl::InvalidArgumentError("Failed to unpack DefaultValue");
  }
  return UnpackDefaultValue<T>(proto);
}

// Gets the value from a DefaultValue pointer. If the pointer is null or
// value is not set, returns the default value for T.
template <typename T>
absl::Status GetValue(const DefaultValue* proto, T& val) {
  if (proto == nullptr || proto->value_case() == DefaultValue::VALUE_NOT_SET) {
    val = GetDefaultValue<T>();
    return absl::OkStatus();
  }
  auto unpacked_val = UnpackDefaultValue<T>(*proto);
  if (!unpacked_val.ok()) return unpacked_val.status();
  val = *unpacked_val;
  return absl::OkStatus();
}

// Gets the value from a DefaultValue reference.
template <typename T>
absl::Status GetValue(const DefaultValue& proto, T& val) {
  return GetValue(&proto, val);
}

// Gets the value from google::protobuf::Any. If the Any is empty,
// returns the default value for T.
template <typename T>
absl::Status GetValue(const google::protobuf::Any& any, T& val) {
  if (any.type_url().empty()) {
    val = GetDefaultValue<T>();
    return absl::OkStatus();
  }
  DefaultValue proto;
  if (!any.UnpackTo(&proto)) {
    return absl::InvalidArgumentError("Failed to unpack DefaultValue");
  }
  return GetValue(proto, val);
}

template <typename T>
absl::StatusOr<google::protobuf::Any> PackToAny(T val) {
  utils::DefaultValue proto;
  if constexpr (std::is_same_v<T, int> || std::is_same_v<T, int32_t>) {
    proto.set_int_value(val);
  } else if constexpr (std::is_same_v<T, int64_t>) {
    proto.set_int64_value(val);
  } else if constexpr (std::is_same_v<T, uint32_t>) {
    proto.set_int_value(static_cast<int32_t>(val));
  } else if constexpr (std::is_same_v<T, uint64_t>) {
    proto.set_int64_value(static_cast<int64_t>(val));
  } else if constexpr (std::is_same_v<T, float>) {
    proto.set_float_value(val);
  } else if constexpr (std::is_same_v<T, double>) {
    proto.set_double_value(val);
  } else if constexpr (std::is_same_v<T, bool>) {
    proto.set_bool_value(val);
  } else if constexpr (std::is_same_v<T, std::string> ||
                       std::is_same_v<T, tsl::tstring>) {
    proto.set_string_value(val);
  } else {
    return absl::UnimplementedError("Unsupported type for PackToAny");
  }
  google::protobuf::Any any;
  any.PackFrom(proto);
  return any;
}

}  // namespace utils
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_UTILS_DEFAULT_VALUE_UTILS_H_

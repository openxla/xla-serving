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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_

#include <algorithm>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving {

// Returns a new Value containing a buffer reshaped to the shape specified.
// Supports HostBuffer and DeviceBuffer.
absl::StatusOr<Value> Reshape(Value&& value,
                             const xla::ifrt::Shape& shape);

// Returns a new Value containing a buffer reshaped to the shape specified and
// reinterpreted with the dtype specified. Supports HostBuffer and DeviceBuffer.
absl::StatusOr<Value> Bitcast(Value&& value, xla::ifrt::DType dtype,
                             const xla::ifrt::Shape& shape);

namespace internal {

// Creates a vector of `Value` from a vector of objects. The objects will be
// moved.
template <typename T>
std::vector<Value> PackAsValues(std::vector<T>&& objects) {
  std::vector<Value> result;
  result.reserve(objects.size());
  for (auto& object : objects) {
    result.push_back(Value(std::move(object)));
  }
  return result;
}

// Creates a map of `Value` from a map of objects.
template <typename K, typename T>
absl::flat_hash_map<K, Value> PackAsValues(absl::flat_hash_map<K, T> objects) {
  absl::flat_hash_map<K, Value> result;
  result.reserve(objects.size());
  for (auto& [key, value] : objects) {
    result[key] = Value(std::move(value));
  }
  return result;
}

// Unpacks a vector of `Value` into a vector of objects. The objects will be
// moved.
template <typename T>
absl::StatusOr<std::vector<T>> UnpackFromValues(std::vector<Value>&& values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (Value& value : values) {
    if (!value.IsType<T>()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Value is not of type `", typeid(T).name(), "`."));
    }
    result.push_back(std::move(value.Get<T>()));
  }
  return result;
}

// Unpacks a vector of `Value` into a vector of objects. The objects will be
// copied.
template <typename T>
absl::StatusOr<std::vector<T>> UnpackFromValues(
    absl::Span<const Value> values) {
  std::vector<T> result;
  result.reserve(values.size());
  for (const Value& value : values) {
    if (!value.IsType<T>()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Value is not of type `", typeid(T).name(), "`."));
    }
    result.push_back(value.Get<T>());
  }
  return result;
}

}  // namespace internal
}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_

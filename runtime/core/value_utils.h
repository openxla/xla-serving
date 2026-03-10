#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_

#include <algorithm>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/log.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/strings/str_cat.h"  // from @com_google_absl
#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/core/value.h"  // from @neptune

namespace jsv::internal {

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

}  // namespace jsv::internal

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_UTILS_H_

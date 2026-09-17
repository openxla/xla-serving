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

#ifndef XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_
#define XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_

#include <type_traits>

namespace neptune::data {
namespace utils {

// Safely multiplies two integers, storing the result in 'result'.
// Returns true on success (no overflow), false if overflow occurred.
template <typename T, typename U, typename V>
[[nodiscard]] inline bool SafeMultiply(T a, U b, V* result) {
  static_assert(std::is_integral_v<T>,
                "SafeMultiply operands must be integral");
  static_assert(std::is_integral_v<U>,
                "SafeMultiply operands must be integral");
  static_assert(std::is_integral_v<V>,
                "SafeMultiply operands must be integral");
  return !__builtin_mul_overflow(a, b, result);
}

// Safely adds two integers, storing the result in 'result'.
// Returns true on success (no overflow), false if overflow occurred.
template <typename T, typename U, typename V>
[[nodiscard]] inline bool SafeAdd(T a, U b, V* result) {
  static_assert(std::is_integral_v<T>, "SafeAdd operands must be integral");
  static_assert(std::is_integral_v<U>, "SafeAdd operands must be integral");
  static_assert(std::is_integral_v<V>, "SafeAdd operands must be integral");
  return !__builtin_add_overflow(a, b, result);
}

}  // namespace utils
}  // namespace neptune::data

#endif  // XLA_SERVING_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_

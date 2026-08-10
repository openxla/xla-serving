#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_

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

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_SAFE_INT_UTILS_H_

/* Copyright 2015 The TensorFlow Authors. All Rights Reserved.
Copyright 2026 Google LLC. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// This file is branched from the computational functors of TensorFlow's:
// http:///third_party/tensorflow/core/kernels/cwise_ops.h

#ifndef XLA_SERVING_DATA_KERNELS_BITWISE_OP_FUNCTORS_H_
#define XLA_SERVING_DATA_KERNELS_BITWISE_OP_FUNCTORS_H_

#include <climits>
#include <type_traits>

namespace xla_serving::data::kernel::functor {

struct bitwise_and {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T x, T y) const {
    return static_cast<T>(x & y);
  }
};

struct bitwise_or {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T x, T y) const {
    return static_cast<T>(x | y);
  }
};

struct bitwise_xor {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T x, T y) const {
    return static_cast<T>(x ^ y);
  }
};

struct invert {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T a) const {
    return static_cast<T>(~a);
  }
};

struct left_shift {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T x, T y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T, and
    // performs left shifts as unsigned shifts.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > static_cast<T>(sizeof(T) * CHAR_BIT - 1)) {
      y_clamped = static_cast<T>(sizeof(T) * CHAR_BIT - 1);
    }
    using U = std::make_unsigned_t<T>;
    return static_cast<T>(static_cast<U>(x) << static_cast<U>(y_clamped));
  }
};

struct right_shift {
  template <typename T>
    requires(std::is_integral_v<T> && !std::is_same_v<T, bool>)
  T operator()(T x, T y) const {
    // Avoids UB: don't shift by larger than the bitwidth of T.
    T y_clamped = y;
    if (y_clamped < 0) {
      y_clamped = 0;
    } else if (y_clamped > static_cast<T>(sizeof(T) * CHAR_BIT - 1)) {
      y_clamped = static_cast<T>(sizeof(T) * CHAR_BIT - 1);
    }
    return static_cast<T>(x >> y_clamped);
  }
};

}  // namespace xla_serving::data::kernel::functor

#endif  // XLA_SERVING_DATA_KERNELS_BITWISE_OP_FUNCTORS_H_

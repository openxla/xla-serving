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
// http:///third_party/tensorflow/core/kernels/cwise_ops.h;l=948;rcl=966763186

#ifndef XLA_SERVING_DATA_KERNELS_MATH_OP_FUNCTORS_H_
#define XLA_SERVING_DATA_KERNELS_MATH_OP_FUNCTORS_H_

// NOLINTBEGIN(misc-include-cleaner)

#include <complex>
#include <cstdint>
#include <limits>
#include <type_traits>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/SpecialFunctions"
#include "data/kernels/bounds_check.h"  // from @xla_serving
#include "data/kernels/math/special_math_impl.h"  // from @xla_serving
#include "tsl/platform/tstring.h"  // from @tsl

namespace Eigen {
namespace internal {

// Adapted from /third_party/tensorflow/core/kernels/cwise_ops.h to
// return absl::StatusOr instead of setting a bool* error flag.
template <typename Scalar, typename Exponent>
struct safe_scalar_binary_pow_op {
  static_assert(std::is_integral_v<Scalar>, "Integer type expected");
  static_assert(std::is_integral_v<Exponent> && std::is_signed_v<Exponent>,
                "Signed integer type expected");

  EIGEN_DEVICE_FUNC absl::StatusOr<Scalar> operator()(const Scalar& a,
                                                      const Exponent& b) const {
    const Exponent safe_b =
        ::xla_serving::data::kernel::internal::SubtleMustCopy(b);
    if (safe_b >= 0) [[likely]] {
      return static_cast<Scalar>(numext::pow(a, safe_b));
    } else {
      return absl::InvalidArgumentError(
          "Integers to negative integer powers are not allowed");
    }
  }
};

// Eigen evaluates complex powers as exp(exponent * log(base)), which yields
// NaN for 0^0. IEEE 754 requires z^0 == 1, which TensorFlow follows,
// including non-finite bases. Handle zero exponents before the logarithm as a
// local workaround, without depending on an upstream Eigen change.
template <typename T, bool IsComplex = NumTraits<T>::IsComplex>
struct jd_scalar_pow_op;

template <typename T>
struct jd_scalar_pow_op<T, /*IsComplex=*/false> : scalar_pow_op<T, T> {};

template <typename T>
struct functor_traits<jd_scalar_pow_op<T, /*IsComplex=*/false>>
    : functor_traits<scalar_pow_op<T, T>> {};

template <typename T>
struct jd_scalar_pow_op<T, /*IsComplex=*/true> : scalar_pow_op<T, T> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& base,
                                                     const T& exponent) const {
    if (Eigen::numext::is_exactly_zero(exponent)) [[unlikely]] {
      return T(1);
    }
    return scalar_pow_op<T, T>::operator()(base, exponent);
  }
};

template <typename T>
struct functor_traits<jd_scalar_pow_op<T, /*IsComplex=*/true>> {
  enum {
    Cost = functor_traits<scalar_pow_op<T, T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = false,
  };
};

// Adapted from /third_party/tensorflow/core/kernels/cwise_ops.h to
// return absl::StatusOr instead of setting a bool* error flag.
template <typename T, typename DivOrMod>
struct safe_div_or_mod_op {
  static_assert(std::is_integral_v<T>, "Integer type expected");

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE absl::StatusOr<T> operator()(
      const T& a, const T& b) const {
    const T safe_b = ::xla_serving::data::kernel::internal::SubtleMustCopy(b);
    if (safe_b != 0) [[likely]] {
      // Avoid FPE for INT_MIN/-1.
      const T safe_a = ::xla_serving::data::kernel::internal::SubtleMustCopy(a);
      if (std::is_signed_v<T> && safe_a == std::numeric_limits<T>::min() &&
          safe_b == T(-1)) [[unlikely]] {
        // Prefer to overflow 'a' instead of crashing.
        const T neg_a =
            static_cast<T>(0U - static_cast<std::make_unsigned_t<T>>(safe_a));
        return DivOrMod()(neg_a, T(1));
      }
      return DivOrMod()(safe_a, safe_b);
    } else {
      return absl::InvalidArgumentError("Integer division by zero");
    }
  }
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T, typename Binary>
struct no_nan_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b != T(0)) {
      return Binary()(a, b);
    } else {
      return T(0);
    }
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet mask = pcmp_eq(b, pzero(b));
    const Packet quotient = Binary().packetOp(a, b);
    return pandnot(quotient, mask);
  }
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T, bool IsComplex = Eigen::NumTraits<T>::IsComplex>
struct div_no_nan_op;

template <typename T>
struct div_no_nan_op<T, /*IsComplex=*/false>
    : public no_nan_op<T, scalar_quotient_op<T>> {};

template <typename T>
struct functor_traits<div_no_nan_op<T, /*IsComplex=*/false>> {
  enum {
    Cost = functor_traits<scalar_quotient_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = true,
  };
};

// Whether or not complex division produces a NaN depends on the underlying
// implementation. Some compilers (e.g. gcc) use a simple method that divides
// by |b|^2, which may underflow to 0 for b != 0.
template <typename T>
struct div_no_nan_op<T, /*IsComplex=*/true> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    if (b == T(0)) {
      return T(0);
    } else {
      // If the numerator is zero, then the result must be zero even if |b|^2
      // underflows to zero.
      const T numerator =
          scalar_product_op<T>()(a, scalar_conjugate_op<T>()(b));
      if (numerator == T(0)) {
        return T(0);
      }
    }
    return scalar_quotient_op<T>()(a, b);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& a,
                                                        const Packet& b) const {
    const Packet numerator = pmul(a, pconj(b));
    const Packet mask = por(pcmp_eq(b, pzero(a)), pcmp_eq(numerator, pzero(a)));
    const Packet quotient = pdiv(a, b);
    return pandnot(quotient, mask);
  }
};

template <typename T>
struct functor_traits<div_no_nan_op<T, /*IsComplex=*/true>> {
  enum {
    Cost = functor_traits<scalar_quotient_op<T>>::Cost + NumTraits<T>::MulCost,
    PacketAccess = packet_traits<T>::HasMul && packet_traits<T>::HasDiv &&
        packet_traits<T>::HasConj,
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T>
struct mul_no_nan_op : public no_nan_op<T, scalar_product_op<T>> {};

template <typename T>
struct functor_traits<mul_no_nan_op<T>> {
  enum {
    Cost = functor_traits<scalar_product_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = true,
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T, typename Enable = void>
struct google_floor_div {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    const T z = x / y;
    // Subtract one if there is a remainder and if the inputs have opposite
    // signs. This approach avoids unnecessary overflows.
    return z * y != x && (x < T(0) != y < T(0)) ? z - T(1) : z;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet x_mask = pcmp_lt(x, zeros);
    Packet y_mask = pcmp_lt(y, zeros);
    Packet x_div_y = pdiv(x, y);
    Packet x_div_y_times_y = pmul(x_div_y, y);
    return pselect(por(peq(x_div_y_times_y, x), peq(x_mask, y_mask)), x_div_y,
                   psub(x_div_y, pones(x)));
  }
};

template <typename T>
struct google_floor_div<T, std::enable_if_t<std::is_unsigned_v<T>>> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pdiv(x, y);
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
        NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasDiv
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T, typename Enable = void>
struct google_floor_div_real {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    return Eigen::numext::floor(x / y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    return pfloor(pdiv(x, y));
  }
};

template <typename Scalar>
struct functor_traits<google_floor_div_real<Scalar>> {
  enum {
    Cost = 2 * Eigen::internal::scalar_div_cost<
                   Scalar, packet_traits<Scalar>::HasDiv>::value +
        2 * NumTraits<Scalar>::AddCost,
    PacketAccess = packet_traits<Scalar>::HasDiv &&
        packet_traits<Scalar>::HasRound
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T>
struct google_floor_fmod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    T trunc_mod = scalar_fmod_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_fmod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_fmod_op<Scalar>>::Cost +
        NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T>
struct google_floor_mod {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x,
                                                     const T& y) const {
    T trunc_mod = Eigen::internal::scalar_mod2_op<T>()(x, y);
    return trunc_mod != T(0) && (y < T(0) != trunc_mod < T(0)) ? trunc_mod + y
                                                               : trunc_mod;
  }
};

template <typename Scalar>
struct functor_traits<google_floor_mod<Scalar>> {
  enum {
    Cost = functor_traits<Eigen::internal::scalar_mod2_op<Scalar>>::Cost +
        NumTraits<Scalar>::AddCost,
    PacketAccess = false
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T>
struct scalar_erfinv_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& x) const {
    constexpr T half = T(0.5);
    T y = numext::ndtri(half * x + half);
    constexpr T half_sqrt = T(M_SQRT1_2);
    return y * half_sqrt;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    Packet half = pset1<Packet>(T(0.5));
    Packet y = pndtri<Packet>(pmadd(half, x, half));
    Packet half_sqrt = pset1<Packet>(T(M_SQRT1_2));
    return pmul(y, half_sqrt);
  }
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename T>
struct functor_traits<scalar_erfinv_op<T>> {
  enum {
    Cost = functor_traits<scalar_ndtri_op<T>>::Cost + NumTraits<T>::AddCost,
    PacketAccess = packet_traits<T>::HasNdtri,
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename Scalar>
struct digamma_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x) const {
    if (x == Scalar(0.)) {
      return -Eigen::NumTraits<Scalar>::infinity();
    }
    return Eigen::internal::scalar_digamma_op<Scalar>()(x);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    Packet infs = pset1<Packet>(-Eigen::NumTraits<Scalar>::infinity());
    Packet digamma_x = Eigen::internal::scalar_digamma_op<Scalar>().packetOp(x);
    return pselect(mask, infs, digamma_x);
  }
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename Scalar>
struct functor_traits<digamma_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_digamma_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::AddCost,
    PacketAccess = functor_traits<scalar_digamma_op<Scalar>>::PacketAccess
  };
};

// Specialization of erfinv for float. Copied over from
// /third_party/tensorflow/core/kernels/cwise_ops.h
template <>
struct scalar_erfinv_op<float> {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE float operator()(float x) const {
    constexpr int kDegree = 9;
    constexpr float w_less_than_5_constants[kDegree] = {
        2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
        -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
        -0.00417768164f,  0.246640727f,    1.50140941f};
    constexpr float w_greater_than_5_constants[kDegree] = {
        -0.000200214257f, 0.000100950558f, 0.00134934322f,
        -0.00367342844f,  0.00573950773f,  -0.0076224613f,
        0.00943887047f,   1.00167406f,     2.83297682f};
    float w = -numext::log1p(-x * x);
    float p;
    if (w < 5.0f) {
      w -= 2.5f;
      p = w_less_than_5_constants[0];
      for (int i = 1; i < kDegree; ++i) p = w_less_than_5_constants[i] + p * w;
    } else {
      w = numext::sqrt(w) - 3.0f;
      p = w_greater_than_5_constants[0];
      for (int i = 1; i < kDegree; ++i)
        p = w_greater_than_5_constants[i] + p * w;
    }
    float result = p * x;
    const float abs_x = numext::abs(x);
    if (abs_x >= 1.0f) {
      result = (abs_x == 1.0f) ? (x < 0.0f ? -NumTraits<float>::infinity()
                                           : NumTraits<float>::infinity())
                               : NumTraits<float>::quiet_NaN();
    }
    return result;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x) const {
    constexpr int kDegree = 9;
    constexpr float w_less_than_5_constants[kDegree] = {
        2.81022636e-08f,  3.43273939e-07f, -3.5233877e-06f,
        -4.39150654e-06f, 0.00021858087f,  -0.00125372503f,
        -0.00417768164f,  0.246640727f,    1.50140941f};
    constexpr float w_greater_than_5_constants[kDegree] = {
        -0.000200214257f, 0.000100950558f, 0.00134934322f,
        -0.00367342844f,  0.00573950773f,  -0.0076224613f,
        0.00943887047f,   1.00167406f,     2.83297682f};
    Packet w = pnegate(plog1p(pnegate(pmul(x, x))));
    Packet lt = pcmp_lt(w, pset1<Packet>(5.0f));
    w = pselect(lt, psub(w, pset1<Packet>(2.5f)),
                psub(psqrt(w), pset1<Packet>(3.0f)));
    Packet p = pselect(lt, pset1<Packet>(w_less_than_5_constants[0]),
                       pset1<Packet>(w_greater_than_5_constants[0]));
    for (int i = 1; i < kDegree; ++i) {
      Packet c = pselect(lt, pset1<Packet>(w_less_than_5_constants[i]),
                         pset1<Packet>(w_greater_than_5_constants[i]));
      p = pmadd(p, w, c);
    }
    Packet result = pmul(p, x);
    Packet one = pset1<Packet>(1.0f);
    Packet abs_x = pabs(x);
    Packet inf = pset1<Packet>(NumTraits<float>::infinity());
    Packet signed_inf = pselect(pcmp_lt(x, pzero(x)), pnegate(inf), inf);
    result = pselect(pcmp_eq(abs_x, one), signed_inf, result);
    result = pselect(pcmp_lt(one, abs_x),
                     pset1<Packet>(NumTraits<float>::quiet_NaN()), result);
    return result;
  }
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <>
struct functor_traits<scalar_erfinv_op<float>> {
  enum {
    Cost = functor_traits<scalar_log1p_op<float>>::Cost +
           20 * NumTraits<float>::MulCost,
    PacketAccess = packet_traits<float>::HasLog1p,
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename Scalar>
struct xlogy_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x * numext::log(y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    scalar_log_op<Scalar> log_op;
    Packet log_y = log_op.packetOp(y);
    Packet x_log_y = pmul(x, log_y);
    return pselect(mask, x, x_log_y);
  }
};

template <typename Scalar>
struct functor_traits<xlogy_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_log_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::MulCost,
    PacketAccess = functor_traits<scalar_log_op<Scalar>>::PacketAccess
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename Scalar>
struct xlog1py_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x * numext::log1p(y);
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    scalar_log1p_op<Scalar> log1p_op;
    Packet log1p_y = log1p_op.packetOp(y);
    Packet x_log1p_y = pmul(x, log1p_y);
    return pselect(mask, x, x_log1p_y);
  }
};

template <typename Scalar>
struct functor_traits<xlog1py_op<Scalar>> {
  enum {
    Cost = functor_traits<scalar_log1p_op<Scalar>>::Cost +
           Eigen::NumTraits<Scalar>::MulCost,
    PacketAccess = functor_traits<scalar_log1p_op<Scalar>>::PacketAccess
  };
};

// Copied over from /third_party/tensorflow/core/kernels/cwise_ops.h
template <typename Scalar>
struct xdivy_op {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Scalar
  operator()(const Scalar& x, const Scalar& y) const {
    if (x == Scalar(0.)) {
      return Scalar(0.);
    }
    return x / y;
  }
  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet packetOp(const Packet& x,
                                                        const Packet& y) const {
    Packet zeros = pzero(x);
    Packet mask = pcmp_eq(x, zeros);
    Packet x_div_y = pdiv(x, y);
    return pselect(mask, x, x_div_y);
  }
};

template <typename Scalar>
struct functor_traits<xdivy_op<Scalar>> {
  enum {
    Cost =
        Eigen::NumTraits<Scalar>::AddCost +
        Eigen::internal::scalar_div_cost<Scalar,
                                         packet_traits<Scalar>::HasDiv>::value,
    PacketAccess = packet_traits<Scalar>::HasDiv
  };
};

}  // namespace internal
}  // namespace Eigen

namespace xla_serving::data {
namespace kernel {
namespace functor {

// Helper to check if a type T matches any candidate type.
template <typename T, typename... Ts>
inline constexpr bool is_one_of_v = (std::is_same_v<T, Ts> || ...);

template <typename T>
inline constexpr bool is_signed_int_v =
    is_one_of_v<T, int8_t, int16_t, int32_t, int64_t>;

template <typename T>
inline constexpr bool is_unsigned_int_v =
    is_one_of_v<T, uint8_t, uint16_t, uint32_t, uint64_t>;

template <typename T>
inline constexpr bool is_float_v =
    is_one_of_v<T, Eigen::half, Eigen::bfloat16, float, double>;

template <typename T>
inline constexpr bool is_complex_v =
    is_one_of_v<T, std::complex<float>, std::complex<double>>;

// Helper to check if a template Op matches any candidate template.
template <template <typename...> class Target,
          template <typename...> class... Candidates>
inline constexpr bool is_one_of_op_v =
    (std::is_same_v<Target<float>, Candidates<float>> || ...);

// Set of Eigen unary operations supported on std::complex<T>.
template <template <typename...> class Op>
inline constexpr bool is_complex_supported_op_v = is_one_of_op_v<
    Op, Eigen::internal::scalar_abs_op, Eigen::internal::scalar_conjugate_op,
    Eigen::internal::scalar_opposite_op, Eigen::internal::scalar_square_op,
    Eigen::internal::scalar_sqrt_op, Eigen::internal::scalar_rsqrt_op,
    Eigen::internal::scalar_exp_op, Eigen::internal::scalar_expm1_op,
    Eigen::internal::scalar_log_op, Eigen::internal::scalar_log1p_op,
    Eigen::internal::scalar_sin_op, Eigen::internal::scalar_cos_op,
    Eigen::internal::scalar_tan_op, Eigen::internal::scalar_asin_op,
    Eigen::internal::scalar_acos_op, Eigen::internal::scalar_atan_op,
    Eigen::internal::scalar_sinh_op, Eigen::internal::scalar_cosh_op,
    Eigen::internal::scalar_tanh_op, Eigen::internal::scalar_asinh_op,
    Eigen::internal::scalar_acosh_op, Eigen::internal::scalar_atanh_op,
    Eigen::internal::scalar_inverse_op, Eigen::internal::scalar_logistic_op,
    Eigen::internal::scalar_sign_op>;

// Set of Eigen unary operations that support exact integral arithmetic (e.g.,
// abs, negation, square, sign, rounding). For operations not in this set (such
// as transcendental or float-only functions like sin, exp, or sqrt), Eigen's
// functors contain compile-time static_asserts that reject integral types,
// requiring polymorphic_unary_op to cast through double to compile.
template <template <typename...> class Op>
inline constexpr bool is_integer_exact_op_v = is_one_of_op_v<
    Op, Eigen::internal::scalar_abs_op, Eigen::internal::scalar_opposite_op,
    Eigen::internal::scalar_square_op, Eigen::internal::scalar_sign_op,
    Eigen::internal::scalar_rint_op, Eigen::internal::scalar_round_op>;

// Wrapper to adapt Eigen's strongly-typed functors into generic polymorphic
// functors like the ones previously used in Jax Data.
template <template <typename...> class EigenOp>
struct polymorphic_unary_op {
  template <typename T>
    requires(!is_one_of_v<T, bool, tsl::tstring> &&
             (!is_complex_v<T> || is_complex_supported_op_v<EigenOp>))
  auto operator()(T x) const {
    if constexpr (is_complex_v<T>) {
      return EigenOp<T>()(x);
    } else if constexpr (std::is_integral_v<T>) {
      // Allow integer exact math for operations naturally defined on integers.
      if constexpr (is_integer_exact_op_v<EigenOp>) {
        return EigenOp<T>()(x);
      } else {
        // For transcendental / float-only ops, cast to double to prevent Eigen
        // static_assert.
        return static_cast<T>(EigenOp<double>()(static_cast<double>(x)));
      }
    } else {
      return EigenOp<T>()(x);
    }
  }
};

template <typename T>
using scalar_max_nan_op =
    Eigen::internal::scalar_max_op<T, T, Eigen::PropagateNaN>;
template <typename T>
using scalar_min_nan_op =
    Eigen::internal::scalar_min_op<T, T, Eigen::PropagateNaN>;

template <template <typename...> class EigenOp>
struct polymorphic_binary_op {
  template <typename T>
    requires(!std::is_same_v<T, bool> &&
             (!std::is_same_v<T, tsl::tstring> ||
              std::is_same_v<EigenOp<T>, Eigen::internal::scalar_sum_op<T>>) &&
             (!is_complex_v<T> ||
              (!std::is_same_v<EigenOp<T>, scalar_max_nan_op<T>> &&
               !std::is_same_v<EigenOp<T>, scalar_min_nan_op<T>>)))
  T operator()(T a, T b) const {
    return EigenOp<T>()(a, b);
  }
};

template <template <typename...> class EigenOp>
struct float_double_binary_op {
  template <typename T>
    requires(std::is_same_v<T, float> || std::is_same_v<T, double>)
  T operator()(T a, T b) const {
    return EigenOp<T>()(a, b);
  }
};

template <template <typename...> class EigenOp>
struct float_complex_binary_op {
  template <typename T>
    requires(!std::is_integral_v<T> && !std::is_same_v<T, tsl::tstring>)
  T operator()(T a, T b) const {
    return EigenOp<T>()(a, b);
  }
};

// Aliases matching TensorFlow functor naming conventions

// Unary functors
using abs = polymorphic_unary_op<Eigen::internal::scalar_abs_op>;
using conj = polymorphic_unary_op<Eigen::internal::scalar_conjugate_op>;
using negative = polymorphic_unary_op<Eigen::internal::scalar_opposite_op>;
using square = polymorphic_unary_op<Eigen::internal::scalar_square_op>;
using sqrt = polymorphic_unary_op<Eigen::internal::scalar_sqrt_op>;
using rsqrt = polymorphic_unary_op<Eigen::internal::scalar_rsqrt_op>;
using log = polymorphic_unary_op<Eigen::internal::scalar_log_op>;
using sin = polymorphic_unary_op<Eigen::internal::scalar_sin_op>;
using cos = polymorphic_unary_op<Eigen::internal::scalar_cos_op>;
using tan = polymorphic_unary_op<Eigen::internal::scalar_tan_op>;
using asin = polymorphic_unary_op<Eigen::internal::scalar_asin_op>;
using acos = polymorphic_unary_op<Eigen::internal::scalar_acos_op>;
using atan = polymorphic_unary_op<Eigen::internal::scalar_atan_op>;
using sinh = polymorphic_unary_op<Eigen::internal::scalar_sinh_op>;
using cosh = polymorphic_unary_op<Eigen::internal::scalar_cosh_op>;
using tanh = polymorphic_unary_op<Eigen::internal::scalar_tanh_op>;
using asinh = polymorphic_unary_op<Eigen::internal::scalar_asinh_op>;
using acosh = polymorphic_unary_op<Eigen::internal::scalar_acosh_op>;
using atanh = polymorphic_unary_op<Eigen::internal::scalar_atanh_op>;
using ceil = polymorphic_unary_op<Eigen::internal::scalar_ceil_op>;
using expm1 = polymorphic_unary_op<Eigen::internal::scalar_expm1_op>;
using floor = polymorphic_unary_op<Eigen::internal::scalar_floor_op>;
using reciprocal = polymorphic_unary_op<Eigen::internal::scalar_inverse_op>;
using log1p = polymorphic_unary_op<Eigen::internal::scalar_log1p_op>;
using rint = polymorphic_unary_op<Eigen::internal::scalar_rint_op>;
using round = polymorphic_unary_op<Eigen::internal::scalar_rint_op>;
using sign = polymorphic_unary_op<Eigen::internal::scalar_sign_op>;
using sigmoid = polymorphic_unary_op<Eigen::internal::scalar_logistic_op>;
struct logical_not {
  bool operator()(bool a) const {
    return Eigen::internal::scalar_boolean_not_op<bool>()(a);
  }
};

// Binary functors
using add = polymorphic_binary_op<Eigen::internal::scalar_sum_op>;
using sub = polymorphic_binary_op<Eigen::internal::scalar_difference_op>;
using mul = polymorphic_binary_op<Eigen::internal::scalar_product_op>;

struct mul_no_nan {
  template <typename T>
    requires(is_float_v<T> || is_complex_v<T>)
  T operator()(T a, T b) const {
    return Eigen::internal::mul_no_nan_op<T>()(a, b);
  }
};

struct div_no_nan {
  template <typename T>
    requires(is_float_v<T> || is_complex_v<T>)
  T operator()(T a, T b) const {
    return Eigen::internal::div_no_nan_op<T>()(a, b);
  }
};

// Safe division matching TensorFlow's safe_div for integers and div for
// floating-point and complex types.
struct div {
  template <typename T>
    requires(is_signed_int_v<T> || is_unsigned_int_v<T> || is_float_v<T> ||
             is_complex_v<T>)
  std::conditional_t<std::is_integral_v<T>, absl::StatusOr<T>, T> operator()(
      T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      return Eigen::internal::safe_div_or_mod_op<
          T, Eigen::internal::scalar_quotient_op<T>>()(a, b);
    } else {
      return Eigen::internal::scalar_quotient_op<T>()(a, b);
    }
  }
};

// Python 3 true division matching TensorFlow's _truediv_python3 (_TRUEDIV_TABLE
// in math_ops.py): promotes 8/16-bit integers to float32 and 32/64-bit integers
// to float64 before dividing, while preserving floating-point and complex
// types.
struct true_div {
  template <typename T>
    requires(is_signed_int_v<T> || is_unsigned_int_v<T> || is_float_v<T> ||
             is_complex_v<T>)
  auto operator()(T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      using OutT = std::conditional_t<sizeof(T) <= 2, float, double>;
      return Eigen::internal::scalar_quotient_op<OutT>()(static_cast<OutT>(a),
                                                         static_cast<OutT>(b));
    } else {
      return Eigen::internal::scalar_quotient_op<T>()(a, b);
    }
  }
};

// Safe floor division matching TensorFlow's safe_floor_div for integers and
// floor_div_real for floating-point types.
struct floor_div {
  template <typename T>
    requires(is_signed_int_v<T> || is_unsigned_int_v<T> || is_float_v<T>)
  std::conditional_t<std::is_integral_v<T>, absl::StatusOr<T>, T> operator()(
      T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      return Eigen::internal::safe_div_or_mod_op<
          T, Eigen::internal::google_floor_div<T>>()(a, b);
    } else {
      return Eigen::internal::google_floor_div_real<T>()(a, b);
    }
  }
};

// Safe floor modulo matching TensorFlow's safe_floor_mod for integers and
// floor_fmod for floating-point types.
struct floor_mod {
  template <typename T>
    requires(is_signed_int_v<T> || is_unsigned_int_v<T> || is_float_v<T>)
  std::conditional_t<std::is_integral_v<T>, absl::StatusOr<T>, T> operator()(
      T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      return Eigen::internal::safe_div_or_mod_op<
          T, Eigen::internal::google_floor_mod<T>>()(a, b);
    } else {
      return Eigen::internal::google_floor_fmod<T>()(a, b);
    }
  }
};

using exp = polymorphic_unary_op<Eigen::internal::scalar_exp_op>;
using maximum = polymorphic_binary_op<scalar_max_nan_op>;
using minimum = polymorphic_binary_op<scalar_min_nan_op>;
using igamma = float_double_binary_op<Eigen::internal::scalar_igamma_op>;
using igammac = float_double_binary_op<Eigen::internal::scalar_igammac_op>;
using polygamma = float_double_binary_op<Eigen::internal::scalar_polygamma_op>;
using xdivy = float_complex_binary_op<Eigen::internal::xdivy_op>;
using xlog1py = float_complex_binary_op<Eigen::internal::xlog1py_op>;
using xlogy = float_complex_binary_op<Eigen::internal::xlogy_op>;
using zeta = float_double_binary_op<Eigen::internal::scalar_zeta_op>;
struct logical_and {
  bool operator()(bool a, bool b) const {
    return Eigen::internal::scalar_boolean_and_op<bool>()(a, b);
  }
};
struct logical_or {
  bool operator()(bool a, bool b) const {
    return Eigen::internal::scalar_boolean_or_op<bool>()(a, b);
  }
};
struct logical_xor {
  bool operator()(bool a, bool b) const {
    return Eigen::internal::scalar_boolean_xor_op<bool>()(a, b);
  }
};
// Safe exponentiation matching TensorFlow's safe_pow for signed integers and
// pow for floating-point and complex types.
struct custom_pow_op {
  template <typename T>
    requires(is_signed_int_v<T> || is_float_v<T> || is_complex_v<T>)
  std::conditional_t<std::is_integral_v<T>, absl::StatusOr<T>, T> operator()(
      T a, T b) const {
    if constexpr (std::is_integral_v<T>) {
      return Eigen::internal::safe_scalar_binary_pow_op<T, T>()(a, b);
    } else {
      return Eigen::internal::jd_scalar_pow_op<T>()(a, b);
    }
  }
};
using pow = custom_pow_op;

struct atan2 {
  template <typename T>
    requires(is_float_v<T>)
  T operator()(T a, T b) const {
    return Eigen::internal::scalar_atan2_op<T, T>()(a, b);
  }
};

// Logical functors
struct isfinite {
  template <typename T>
    requires(!std::is_same_v<T, tsl::tstring>)
  bool operator()(T x) const {
    if constexpr (is_float_v<T>) {
      return (Eigen::numext::isfinite)(x);
    } else {
      return true;
    }
  }
};

struct isinf {
  template <typename T>
    requires(!std::is_same_v<T, tsl::tstring>)
  bool operator()(T x) const {
    if constexpr (is_float_v<T>) {
      return (Eigen::numext::isinf)(x);
    } else {
      return false;
    }
  }
};

struct isnan {
  template <typename T>
    requires(!std::is_same_v<T, tsl::tstring>)
  bool operator()(T x) const {
    if constexpr (is_float_v<T>) {
      return (Eigen::numext::isnan)(x);
    } else {
      return false;
    }
  }
};

// Special Math / Advanced Functions
using dawsn = polymorphic_unary_op<Eigen::internal::dawsn_op>;
using digamma = polymorphic_unary_op<Eigen::internal::digamma_op>;
using erf = polymorphic_unary_op<Eigen::internal::scalar_erf_op>;
using erfc = polymorphic_unary_op<Eigen::internal::scalar_erfc_op>;
struct erfinv {
  template <typename T>
    requires(is_one_of_v<T, float, double>)
  T operator()(T x) const {
    if constexpr (std::is_same_v<T, float>) {
      return Eigen::internal::scalar_erfinv_op<float>()(x);
    } else {
      return Eigen::internal::scalar_erfinv_op<double>()(x);
    }
  }
};
using expint = polymorphic_unary_op<Eigen::internal::expint_op>;
using fresnel_cos = polymorphic_unary_op<Eigen::internal::fresnel_cos_op>;
using fresnel_sin = polymorphic_unary_op<Eigen::internal::fresnel_sin_op>;
using lgamma = polymorphic_unary_op<Eigen::internal::scalar_lgamma_op>;
using ndtri = polymorphic_unary_op<Eigen::internal::scalar_ndtri_op>;
using spence = polymorphic_unary_op<Eigen::internal::spence_op>;

// Bessel Functions
using bessel_i0 = polymorphic_unary_op<Eigen::internal::scalar_bessel_i0_op>;
using bessel_i0e = polymorphic_unary_op<Eigen::internal::scalar_bessel_i0e_op>;
using bessel_i1 = polymorphic_unary_op<Eigen::internal::scalar_bessel_i1_op>;
using bessel_i1e = polymorphic_unary_op<Eigen::internal::scalar_bessel_i1e_op>;
using bessel_k0 = polymorphic_unary_op<Eigen::internal::scalar_bessel_k0_op>;
using bessel_k0e = polymorphic_unary_op<Eigen::internal::scalar_bessel_k0e_op>;
using bessel_k1 = polymorphic_unary_op<Eigen::internal::scalar_bessel_k1_op>;
using bessel_k1e = polymorphic_unary_op<Eigen::internal::scalar_bessel_k1e_op>;
using bessel_j0 = polymorphic_unary_op<Eigen::internal::scalar_bessel_j0_op>;
using bessel_j1 = polymorphic_unary_op<Eigen::internal::scalar_bessel_j1_op>;
using bessel_y0 = polymorphic_unary_op<Eigen::internal::scalar_bessel_y0_op>;
using bessel_y1 = polymorphic_unary_op<Eigen::internal::scalar_bessel_y1_op>;

}  // namespace functor
}  // namespace kernel
}  // namespace xla_serving::data

// NOLINTEND(misc-include-cleaner)

#endif  // XLA_SERVING_DATA_KERNELS_MATH_OP_FUNCTORS_H_

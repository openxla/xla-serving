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

#ifndef XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_H_
#define XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_H_

// Functor definitions for Reduction ops, must be compilable by nvcc.

#include <cstdint>
#include <type_traits>

#include "data/kernels/util/ttypes.h"  // from @xla_serving
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla_serving::data::kernel {
#pragma GCC visibility push(default)
namespace functor {

template <typename T>
using AndReducer = Eigen::internal::AndReducer;

template <typename T>
using OrReducer = Eigen::internal::OrReducer;

template <typename T>
using MaxReducer = Eigen::internal::MaxReducer<T, Eigen::PropagateNaN>;

template <typename T>
using MinReducer = Eigen::internal::MinReducer<T, Eigen::PropagateNaN>;

template <typename Reducer>
struct ReducerTraits {
  enum { IsScalarIdentity = true };
};

// Dummy class used for template specialization for mean reduction, which is
// accomplished by SumReducer and on-the-fly division by the reduction factor.
template <typename Scalar>
struct MeanReducer {
  Scalar initialize() const { return Scalar(0); }
};

// Dummy class used for template specialization for l2-norm reduction.
template <typename Scalar>
struct EuclideanNormReducer {
  Scalar initialize() const { return Scalar(0); }
};

template <typename Scalar>
struct ReducerTraits<EuclideanNormReducer<Scalar>> {
  enum { IsScalarIdentity = false };
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Reducer>
struct ReduceEigenImpl {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes, const Reducer& reducer) {
    out.device(d) = in.reduce(reduction_axes, reducer);
  }
};

// Specialization for BF16 Reducer to fix accuracy.
// TODO: All BF16 reducers should have specializations to fix accuracy.
#define CASTING_SPECIALIZATION(Reducer, ScalarType, IntermediateType)        \
  template <typename Device, typename OUT_T, typename IN_T,                  \
            typename ReductionAxes>                                          \
  struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,                 \
                         Reducer<ScalarType>> {                              \
    void operator()(const Device& d, OUT_T out, IN_T in,                     \
                    const ReductionAxes& reduction_axes,                     \
                    const Reducer<ScalarType>& reducer) {                    \
      static_assert(std::is_same_v<ScalarType, typename OUT_T::Scalar>, ""); \
      Reducer<IntermediateType> intermediate_reducer;                        \
      auto in_as_intermediate = in.template cast<IntermediateType>();        \
      out.device(d) =                                                        \
          in_as_intermediate.reduce(reduction_axes, intermediate_reducer)    \
              .template cast<ScalarType>();                                  \
    }                                                                        \
  };

CASTING_SPECIALIZATION(Eigen::internal::SumReducer, Eigen::bfloat16, float);
#undef CASTING_SPECIALIZATION

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::MeanReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::MeanReducer<Scalar>& reducer) {
    static_assert(std::is_same_v<Scalar, typename OUT_T::Scalar>, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) = in.reduce(reduction_axes, sum_reducer) /
                    static_cast<Scalar>(in.size() / out.size());
  }
};

// Specialization for which we do the reduction in IntermediateType to
// avoid integer overflow and fix bfloat16 accuracy in some models.
#define CASTING_SPECIALIZATION(ScalarType, IntermediateType)                  \
  template <typename Device, typename OUT_T, typename IN_T,                   \
            typename ReductionAxes>                                           \
  struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,                  \
                         functor::MeanReducer<ScalarType>> {                  \
    void operator()(const Device& d, OUT_T out, IN_T in,                      \
                    const ReductionAxes& reduction_axes,                      \
                    const functor::MeanReducer<ScalarType>& reducer) {        \
      static_assert(std::is_same_v<ScalarType, typename OUT_T::Scalar>, "");  \
      Eigen::internal::SumReducer<IntermediateType> sum_reducer;              \
      out.device(d) = (in.template cast<IntermediateType>().reduce(           \
                           reduction_axes, sum_reducer) /                     \
                       static_cast<IntermediateType>(in.size() / out.size())) \
                          .template cast<ScalarType>();                       \
    }                                                                         \
  }

CASTING_SPECIALIZATION(uint8_t, uint64_t);
CASTING_SPECIALIZATION(uint16_t, uint64_t);
CASTING_SPECIALIZATION(uint32_t, uint64_t);
CASTING_SPECIALIZATION(int8_t, int64_t);
CASTING_SPECIALIZATION(int16_t, int64_t);
CASTING_SPECIALIZATION(int32_t, int64_t);
CASTING_SPECIALIZATION(Eigen::bfloat16, float);
#undef CASTING_SPECIALIZATION

// TODO: Refactor this such that taking the sqrt can be optional
// controlled by an attribute.
template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes, typename Scalar>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<Scalar>> {
  void operator()(const Device& d, OUT_T out, IN_T in,
                  const ReductionAxes& reduction_axes,
                  const functor::EuclideanNormReducer<Scalar>& reducer) {
    static_assert(std::is_same_v<Scalar, typename OUT_T::Scalar>, "");
    Eigen::internal::SumReducer<Scalar> sum_reducer;
    out.device(d) =
        (in * in.conjugate()).reduce(reduction_axes, sum_reducer).sqrt();
  }
};

template <typename Device, typename OUT_T, typename IN_T,
          typename ReductionAxes>
struct ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes,
                       functor::EuclideanNormReducer<Eigen::bfloat16>> {
  void operator()(
      const Device& d, OUT_T out, IN_T in, const ReductionAxes& reduction_axes,
      const functor::EuclideanNormReducer<Eigen::bfloat16>& reducer) {
    static_assert(std::is_same_v<Eigen::bfloat16, typename OUT_T::Scalar>, "");
    Eigen::internal::SumReducer<float> sum_reducer;
    auto in_as_float = in.template cast<float>();
    out.device(d) = (in_as_float * in_as_float.conjugate())
                        .reduce(reduction_axes, sum_reducer)
                        .sqrt()
                        .template cast<Eigen::bfloat16>();
  }
};

// For most reducers, the identity is Reducer::initialize()
template <typename Reducer>
struct Identity {
  static auto identity(const Reducer& reducer)
      -> decltype(reducer.initialize()) {
    return reducer.initialize();
  }
};

// MeanReducer is a special case, since it doesn't technically have an identity.
// Thus, ideally we'd return nan.  However, mean is instantiated for integer
// types as well, so we do the nan override only for floating point types.
#define FIX_MEAN_IDENTITY(T)                            \
  template <>                                           \
  struct Identity<functor::MeanReducer<T>> {            \
    static T identity(const functor::MeanReducer<T>&) { \
      return Eigen::NumTraits<T>::quiet_NaN();          \
    }                                                   \
  };
FIX_MEAN_IDENTITY(Eigen::half)
FIX_MEAN_IDENTITY(Eigen::bfloat16)
FIX_MEAN_IDENTITY(float)
FIX_MEAN_IDENTITY(double)
#undef FIX_MEAN_IDENTITY

template <typename Device, typename OUT_T, typename Reducer>
void FillIdentityEigenImpl(const Device& d, OUT_T out, const Reducer& reducer) {
  MaybeWith32BitIndexing<Device>(
      [&](auto out32) {
        out32.device(d) = out32.constant(Identity<Reducer>::identity(reducer));
      },
      out);
}

template <typename Device, typename Reducer>
struct ReduceFunctor {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const Device& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer);

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out, const Reducer& reducer);
};

}  // namespace functor
#pragma GCC visibility pop
}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_H_

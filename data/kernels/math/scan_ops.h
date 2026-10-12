/* Copyright 2016 The TensorFlow Authors. All Rights Reserved.
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

#ifndef XLA_SERVING_DATA_KERNELS_MATH_SCAN_OPS_H_
#define XLA_SERVING_DATA_KERNELS_MATH_SCAN_OPS_H_

#include <cstdint>
#include <memory>
#include <optional>

#include "data/kernels/util/ttypes.h"  // from @xla_serving
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data::kernel {
#pragma GCC visibility push(default)
namespace functor {

typedef Eigen::Index Index;

// TODO: Needs to be vectorized.
template <typename Device, typename Reducer, typename T>
struct Scan {
  void operator()(const Device& d, typename TTypes<T, 3>::ConstTensor in,
                  typename TTypes<T, 3>::Tensor out, const Reducer& reducer,
                  const bool reverse, const bool exclusive) {
    // Perform the reverse ops directly with Eigen, which avoids copying the
    // tensor twice compared to using individual ops.
    Eigen::array<bool, 3> dims;
    dims[0] = false;
    dims[1] = reverse;
    dims[2] = false;
    MaybeWith32BitIndexing<Device>(
        [&](auto in32, auto out32) {
          out32.device(d) =
              in32.reverse(dims).scan(1, reducer, exclusive).reverse(dims);
        },
        in, out);
  }
};

template <typename T>
struct LogSumExp {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T operator()(const T& a,
                                                     const T& b) const {
    auto mi = Eigen::internal::scalar_min_op<T>()(a, b);
    auto ma = Eigen::internal::scalar_max_op<T>()(a, b);

    auto sub = Eigen::internal::scalar_difference_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto log1p = Eigen::internal::scalar_log1p_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();

    auto logsumexp = add(log1p(exp(sub(mi, ma))), ma);
    // Return ma directly if it is -inf (all inputs -inf) or +inf
    // (avoids inf - inf = NaN in the subtraction above).
    return (cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ||
            cmp_lt(Eigen::NumTraits<T>::highest(), ma))
               ? ma
               : logsumexp;
  }
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T packetOp(const T& a,
                                                   const T& b) const {
    auto mi = Eigen::internal::pmin(a, b);
    auto ma = Eigen::internal::pmax(a, b);
    using Eigen::internal::padd;
    using Eigen::internal::pcmp_lt;
    using Eigen::internal::pexp;
    using Eigen::internal::plog1p;
    using Eigen::internal::por;
    using Eigen::internal::pset1;
    using Eigen::internal::psub;

    auto logsumexp = padd(plog1p(pexp(psub(mi, ma))), ma);
    // Select ma directly if it is -inf or +inf.
    auto is_inf = por(pcmp_lt(ma, pset1(Eigen::NumTraits<T>::lowest())),
                      pcmp_lt(pset1(Eigen::NumTraits<T>::highest()), ma));
    return pselect(is_inf, ma, logsumexp);
  }
};

template <typename T>
struct LogSumExpReducer {
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reduce(const T t, T* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp(*accum, t);
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE void reducePacket(const Packet& p,
                                                          Packet* accum) const {
    LogSumExp<T> logsumexp;
    *accum = logsumexp.packetOp(*accum, p);
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T initialize() const {
    return -Eigen::NumTraits<T>::infinity();
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet initializePacket() const {
    return Eigen::internal::pset1(initialize());
  }

  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T finalize(const T accum) const {
    return accum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE Packet
  finalizePacket(const Packet& vaccum) const {
    return vaccum;
  }

  template <typename Packet>
  EIGEN_DEVICE_FUNC EIGEN_STRONG_INLINE T
  finalizeBoth(const T saccum, const Packet& vaccum) const {
    auto max_reducer = Eigen::internal::MaxReducer<T, Eigen::PropagateNaN>();
    auto sum_reducer = Eigen::internal::SumReducer<T>();
    auto exp = Eigen::internal::scalar_exp_op<T>();
    auto cmp_lt =
        Eigen::internal::scalar_cmp_op<T, T, Eigen::internal::cmp_LT>();
    auto log = Eigen::internal::scalar_log_op<T>();
    auto add = Eigen::internal::scalar_sum_op<T>();

    using Eigen::internal::pexp;
    using Eigen::internal::psub;

    // `ma = max(x1, ..., xn)`
    // If the max of all of the `xi` is `-infinity` then the result is
    // -infinity. If the max is larger than `-infinity` then it's safe to use
    // for normalization even if the other elements are `-infinity`.
    //
    // `logsumexp(x1, ..., xn) = ma + log (exp(x1 - ma) + ... + exp(xn - ma))`
    auto ma = max_reducer.finalizeBoth(saccum, vaccum);
    auto logsumexp = add(log(sum_reducer.finalizeBoth(
                             exp(saccum - ma), pexp(psub(vaccum, pset1(ma))))),
                         ma);
    return cmp_lt(ma, Eigen::NumTraits<T>::lowest()) ? initialize() : logsumexp;
  }
};

}  // namespace functor

// Base class for cumulative scan transforms (Cumsum, Cumprod,
// CumulativeLogsumexp). When `kFloatOnly` is true (e.g., CumulativeLogsumexp),
// input dtype dispatch is restricted to floating-point types
// (`ND_CALL_FLOAT_TYPES`); otherwise, all real numeric types
// (`ND_CALL_REAL_NUMBER_TYPES`) are supported.
template <typename Derived, template <typename> class Reducer,
          bool kFloatOnly = false>
class ScanTransformBase : public ::xla_serving::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<Derived>> Create(
      bool exclusive = false, bool reverse = false,
      std::optional<int64_t> axis = std::nullopt);

  ~ScanTransformBase() override = default;

  absl::StatusOr<xla_serving::HostBuffer> operator()(
      const xla_serving::HostBuffer& input) const;

  absl::StatusOr<xla_serving::HostBuffer> operator()(const xla_serving::HostBuffer& input,
                                             const xla_serving::HostBuffer& axis) const;

  absl::Status Execute(
      ::xla_serving::data::composite::api::ExecuteContext& context) override;

  absl::Span<const ::xla_serving::data::composite::api::TypeInfo> InputTypes()
      const override;
  absl::Span<const ::xla_serving::data::composite::api::TypeInfo> OutputTypes()
      const override;

  absl::StatusOr<::xla_serving::data::composite::TransformConfig> ToProto()
      const override;

 protected:
  explicit ScanTransformBase(bool exclusive, bool reverse,
                             std::optional<int64_t> axis = std::nullopt)
      : exclusive_(exclusive), reverse_(reverse), axis_(axis) {}

  absl::Status PropagateSignatureImpl(
      ::xla_serving::data::composite::api::ShapeInferenceContext& context)
      const override;

  bool exclusive_;
  bool reverse_;
  std::optional<int64_t> axis_;

 private:
  template <typename Tidx>
  absl::Status DispatchByDType(const xla_serving::HostBuffer& input,
                               const xla_serving::HostBuffer& axis,
                               xla_serving::HostBuffer& output) const;
};

class CumsumTransform
    : public ScanTransformBase<CumsumTransform, Eigen::internal::SumReducer> {
 public:
  static constexpr const char* kOpName = "CumsumTransform";

 private:
  friend class ScanTransformBase<CumsumTransform, Eigen::internal::SumReducer>;
  using ScanTransformBase::ScanTransformBase;
};

class CumprodTransform
    : public ScanTransformBase<CumprodTransform, Eigen::internal::ProdReducer> {
 public:
  static constexpr const char* kOpName = "CumprodTransform";

 private:
  friend class ScanTransformBase<CumprodTransform,
                                 Eigen::internal::ProdReducer>;
  using ScanTransformBase::ScanTransformBase;
};

class CumulativeLogsumexpTransform
    : public ScanTransformBase<CumulativeLogsumexpTransform,
                               functor::LogSumExpReducer,
                               /*kFloatOnly=*/true> {
 public:
  static constexpr const char* kOpName = "CumulativeLogsumexpTransform";

 private:
  friend class ScanTransformBase<CumulativeLogsumexpTransform,
                                 functor::LogSumExpReducer,
                                 /*kFloatOnly=*/true>;
  using ScanTransformBase::ScanTransformBase;
};

#pragma GCC visibility pop
}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_MATH_SCAN_OPS_H_

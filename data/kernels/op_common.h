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

#ifndef XLA_SERVING_DATA_KERNELS_OP_COMMON_H_
#define XLA_SERVING_DATA_KERNELS_OP_COMMON_H_

#include <cstdint>
#include <memory>
#include <type_traits>
#include <utility>

#include "data/kernels/macros.h"  // from @xla_serving
#include "data/kernels/util/bcast.h"  // from @xla_serving
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving::data::kernel {

#if defined(__GNUC__) || defined(__clang__)
#define ND_EXPORT_SYMBOL __attribute__((visibility("default")))
#else
#define ND_EXPORT_SYMBOL
#endif

// Shared base class providing common element-wise validation and signature
// propagation logic, mirroring TensorFlow's BinaryOpShared.
class ND_EXPORT_SYMBOL BinaryOpShared
    : public xla_serving::data::composite::TransformBase {
 public:
  static absl::Status ValidateBinaryInputs(const xla_serving::HostBuffer& a,
                                           const xla_serving::HostBuffer& b,
                                           absl::string_view op_name) {
    if (a.dtype() != b.dtype()) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " inputs must have the same dtype. Got ",
                       a.dtype(), " and ", b.dtype()));
    }
    BCast bcast(a.shape(), b.shape());
    if (!bcast.IsValid()) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " inputs are not broadcast-compatible. Got ",
                       a.shape(), " and ", b.shape()));
    }
    return absl::OkStatus();
  }

  static absl::Status PropagateBinarySignature(
      xla_serving::data::composite::api::ShapeInferenceContext& context,
      absl::string_view op_name) {
    if (context.inputs.size() != 2) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " expects 2 inputs"));
    }
    if (!context.inputs[0].IsSpec() || !context.inputs[1].IsSpec()) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " expects tensor inputs"));
    }
    const auto& a = context.inputs[0].spec();
    const auto& b = context.inputs[1].spec();
    if (a.dtype != b.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " inputs must have the same dtype"));
    }
    BCast bcast(a.shape, b.shape);
    if (!bcast.IsValid()) {
      return absl::InvalidArgumentError(absl::StrCat(
          op_name, " inputs are not broadcast-compatible: ", a.shape, " and ",
          b.shape));
    }
    auto out_spec = a;
    out_spec.shape = BCast::ToShape(bcast.output_shape());
    context.outputs.push_back(out_spec);
    return absl::OkStatus();
  }
};

template <typename Functor>
xla::ifrt::DType GetUnaryOutputDType(xla::ifrt::DType in_dtype);

// Shared base class providing common element-wise validation and signature
// propagation logic for unary ops, mirroring TensorFlow's UnaryOp base.
class ND_EXPORT_SYMBOL UnaryOpShared
    : public xla_serving::data::composite::TransformBase {
 public:
  template <typename Functor = void>
  static absl::Status PropagateUnarySignature(
      xla_serving::data::composite::api::ShapeInferenceContext& context,
      absl::string_view op_name) {
    if (context.inputs.size() != 1) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " expects 1 input"));
    }
    if (!context.inputs[0].IsSpec()) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " expects tensor inputs"));
    }
    auto out_spec = context.inputs[0].spec();
    if constexpr (!std::is_void_v<Functor>) {
      out_spec.dtype = GetUnaryOutputDType<Functor>(out_spec.dtype);
    }
    context.outputs.push_back(out_spec);
    return absl::OkStatus();
  }
};

namespace internal {

template <typename T>
struct is_status_or : std::false_type {};

template <typename T>
struct is_status_or<absl::StatusOr<T>> : std::true_type {};

template <typename T>
inline constexpr bool is_status_or_v = is_status_or<std::decay_t<T>>::value;

template <typename T>
struct status_or_value_type {
  using type = T;
};

template <typename T>
struct status_or_value_type<absl::StatusOr<T>> {
  using type = T;
};

template <typename T>
using status_or_value_type_t =
    typename status_or_value_type<std::decay_t<T>>::type;

}  // namespace internal

// Generic element-wise application helpers
template <typename Functor>
xla::ifrt::DType GetUnaryOutputDType(xla::ifrt::DType in_dtype) {
  if constexpr (std::is_void_v<Functor>) {
    return in_dtype;
  } else {
    switch (in_dtype.kind()) {
#define CHECK_OUT_DTYPE(T, KIND)                                             \
  case KIND:                                                                 \
    if constexpr (std::is_invocable_v<Functor, T>) {                         \
      using Ret =                                                            \
          internal::status_or_value_type_t<decltype(std::declval<Functor>()( \
              std::declval<T>()))>;                                          \
      return xla::ifrt::DType(NativeAsDTypeKind<Ret>::value);                \
    }                                                                        \
    break;
      ND_APPLY_ALL_NUMBER_CASE(CHECK_OUT_DTYPE)
      CHECK_OUT_DTYPE(bool, xla::ifrt::DType::Kind::kPred)
      CHECK_OUT_DTYPE(tsl::tstring, xla::ifrt::DType::Kind::kString)
#undef CHECK_OUT_DTYPE
      default:
        break;
    }
    return in_dtype;
  }
}

template <typename T, typename Functor>
absl::Status ApplyUnaryOp(const xla_serving::HostBuffer& in, xla_serving::HostBuffer* out,
                          Functor functor, absl::string_view op_name) {
  if (in.dtype().kind() != NativeAsDTypeKind<T>::value) {
    return absl::InvalidArgumentError(
        absl::StrCat("Operator '", op_name,
                     "' type mismatch for T and in.dtype: ", in.dtype()));
  }
  if constexpr (std::is_invocable_v<Functor, T>) {
    using ReturnType = decltype(functor(std::declval<T>()));
    using OutT = internal::status_or_value_type_t<ReturnType>;
    auto src_flat = in.flat<T>();
    auto dst_flat = out->flat<OutT>();
    const T* src = src_flat.data();
    OutT* dst = dst_flat.data();
    int64_t n = src_flat.size();
    for (int64_t i = 0; i < n; ++i) {
      if constexpr (internal::is_status_or_v<ReturnType>) {
        ASSIGN_OR_RETURN(auto val, functor(src[i]));
        dst[i] = static_cast<OutT>(val);
      } else {
        dst[i] = static_cast<OutT>(functor(src[i]));
      }
    }
    return absl::OkStatus();
  } else {
    return absl::UnimplementedError(absl::StrCat(
        "Operator '", op_name,
        "' does not provide an implementation for dtype: ", in.dtype()));
  }
}

template <typename T, typename Functor>
absl::Status ApplyBinaryOp(const xla_serving::HostBuffer& in1,
                           const xla_serving::HostBuffer& in2, xla_serving::HostBuffer* out,
                           Functor functor, absl::string_view op_name) {
  const xla::ifrt::DType expected_dtype(NativeAsDTypeKind<T>::value);
  if (in1.dtype() != expected_dtype || in2.dtype() != expected_dtype ||
      out->dtype() != expected_dtype) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Operator '", op_name, "' dtype mismatch: expected ", expected_dtype,
        ", but got input 1 as ", in1.dtype(), ", and input 2 as ", in2.dtype(),
        ", and output as ", out->dtype()));
  }
  if constexpr (!std::is_invocable_v<Functor, T, T>) {
    return absl::UnimplementedError(absl::StrCat(
        "Operator '", op_name,
        "' does not provide an implementation for dtype: ", in1.dtype()));
  } else {  // Required by C++17 `if constexpr` to discard the body below.
    using ReturnType = decltype(functor(std::declval<T>(), std::declval<T>()));
    auto src1_flat = in1.flat<T>();
    auto src2_flat = in2.flat<T>();
    auto dst_flat = out->flat<T>();
    const T* src1 = src1_flat.data();
    const T* src2 = src2_flat.data();
    T* dst = dst_flat.data();

    // Fast path: identical shapes. Bypasses BCast construction/coalescing and
    // guarantees compile-time unit stride (+1) for SIMD auto-vectorization.
    if (in1.shape() == in2.shape()) {
      if (out->shape() != in1.shape()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Operator '", op_name, "' output buffer has incorrect shape: ",
            out->shape(), ", expected: ", in1.shape()));
      }
      const int64_t n = src1_flat.size();
      for (int64_t i = 0; i < n; ++i) {
        if constexpr (internal::is_status_or_v<ReturnType>) {
          ASSIGN_OR_RETURN(auto val, functor(src1[i], src2[i]));
          dst[i] = static_cast<T>(val);
        } else {
          dst[i] = static_cast<T>(functor(src1[i], src2[i]));
        }
      }
      return absl::OkStatus();
    }

    // Step 1: Validate broadcast compatibility and coalesce adjacent dimensions
    // that share identical broadcast behavior (e.g. [2, 5, 1] + [2, 5, 8]
    // coalesces to [10, 1] + [10, 8]).
    BCast bcast(in1.shape(), in2.shape());
    if (!bcast.IsValid()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Operator '", op_name, "' inputs are not broadcast-compatible: ",
          in1.shape(), " and ", in2.shape()));
    }
    const auto expected_out_shape = BCast::ToShape(bcast.output_shape());
    if (out->shape() != expected_out_shape) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Operator '", op_name, "' output buffer has incorrect shape: ",
          out->shape(), ", expected: ", expected_out_shape));
    }

    const int64_t out_num_elements = out->shape().num_elements();
    if (out_num_elements == 0) {
      return absl::OkStatus();
    }

    // Step 2: Compute effective row-major strides over BCast's coalesced shapes
    // of rank k. Setting stride to 0 when shape[d] == 1 repeats elements along
    // broadcasted dimensions without duplicating memory.
    const auto& res_shape = bcast.result_shape();
    const auto& x_shape = bcast.x_reshape();
    const auto& y_shape = bcast.y_reshape();
    const int k = static_cast<int>(res_shape.size());

    absl::InlinedVector<int64_t, 4> stride1(k);
    absl::InlinedVector<int64_t, 4> stride2(k);
    int64_t s1 = 1;
    int64_t s2 = 1;
    for (int d = k - 1; d >= 0; --d) {
      stride1[d] = (x_shape[d] == 1) ? 0 : s1;
      stride2[d] = (y_shape[d] == 1) ? 0 : s2;
      s1 *= x_shape[d];
      s2 *= y_shape[d];
    }

    // Step 3: Sweep across the innermost dimension (length inner_len, strides
    // inner_s1/inner_s2 in {0, 1}) and advance outer coordinates like an
    // odometer in O(1) amortized time per row.
    const int64_t inner_len = res_shape[k - 1];
    const int64_t inner_s1 = stride1[k - 1];
    const int64_t inner_s2 = stride2[k - 1];
    const int64_t num_outer = out_num_elements / inner_len;

    absl::InlinedVector<int64_t, 4> coord(k, 0);
    int64_t offset1 = 0;
    int64_t offset2 = 0;
    int64_t dst_idx = 0;

    for (int64_t outer = 0; outer < num_outer; ++outer) {
      const T* p1 = src1 + offset1;
      const T* p2 = src2 + offset2;
      for (int64_t j = 0; j < inner_len; ++j) {
        if constexpr (internal::is_status_or_v<ReturnType>) {
          ASSIGN_OR_RETURN(auto val,
                           functor(p1[j * inner_s1], p2[j * inner_s2]));
          dst[dst_idx++] = static_cast<T>(val);
        } else {
          dst[dst_idx++] =
              static_cast<T>(functor(p1[j * inner_s1], p2[j * inner_s2]));
        }
      }

      for (int d = k - 2; d >= 0; --d) {
        coord[d]++;
        offset1 += stride1[d];
        offset2 += stride2[d];
        if (coord[d] < res_shape[d]) {
          break;
        }
        coord[d] = 0;
        offset1 -= stride1[d] * res_shape[d];
        offset2 -= stride2[d] * res_shape[d];
      }
    }

    return absl::OkStatus();
  }
}

// Type dispatching engine for Jax Data HostBuffers
template <typename Functor>
absl::Status DispatchUnary(const xla_serving::HostBuffer& in, xla_serving::HostBuffer* out,
                           Functor functor, absl::string_view op_name) {
  switch (in.dtype().kind()) {
#define DISPATCH_CASE(T, DTYPE_KIND) \
  case DTYPE_KIND:                   \
    return ApplyUnaryOp<T>(in, out, functor, op_name);
    ND_APPLY_ALL_NUMBER_CASE(DISPATCH_CASE)
    DISPATCH_CASE(bool, xla::ifrt::DType::Kind::kPred)
    DISPATCH_CASE(tsl::tstring, xla::ifrt::DType::Kind::kString)
#undef DISPATCH_CASE
    default:
      return absl::UnimplementedError(
          absl::StrCat("Operator '", op_name,
                       "' generally unsupported dtype: ", in.dtype()));
  }
}

template <typename Functor>
absl::Status DispatchBinary(const xla_serving::HostBuffer& in1,
                            const xla_serving::HostBuffer& in2, xla_serving::HostBuffer* out,
                            Functor functor, absl::string_view op_name) {
  switch (in1.dtype().kind()) {
#define DISPATCH_CASE(T, DTYPE_KIND) \
  case DTYPE_KIND:                   \
    return ApplyBinaryOp<T>(in1, in2, out, functor, op_name);
    ND_APPLY_ALL_NUMBER_CASE(DISPATCH_CASE)
    DISPATCH_CASE(bool, xla::ifrt::DType::Kind::kPred)
    DISPATCH_CASE(tsl::tstring, xla::ifrt::DType::Kind::kString)
#undef DISPATCH_CASE
    default:
      return absl::UnimplementedError(
          absl::StrCat("Operator '", op_name,
                       "' generally unsupported dtype: ", in1.dtype()));
  }
}

// Coefficient-wise binary operations template base.
// Mirrors tensorflow::BinaryOp<Device, Functor>
template <typename Derived, typename Functor>
class ND_EXPORT_SYMBOL BinaryOp : public BinaryOpShared {
 public:
  static absl::StatusOr<std::unique_ptr<Derived>> Create() {
    return std::make_unique<Derived>();
  }

  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer a,
                                             xla_serving::HostBuffer b) {
    RETURN_IF_ERROR(ValidateBinaryInputs(a, b, Derived::kOpName));
    BCast bcast(a.shape(), b.shape());
    xla_serving::HostBuffer out(a.dtype(), BCast::ToShape(bcast.output_shape()));
    RETURN_IF_ERROR(DispatchBinary(a, b, &out, Functor{}, Derived::kOpName));
    return out;
  }

  absl::StatusOr<xla_serving::data::composite::TransformConfig> ToProto()
      const override {
    xla_serving::data::composite::TransformConfig config;
    config.set_op_name(Derived::kOpName);
    return config;
  }

 protected:
  absl::Status PropagateSignatureImpl(
      xla_serving::data::composite::api::ShapeInferenceContext& context)
      const override {
    return PropagateBinarySignature(context, Derived::kOpName);
  }
};

// Coefficient-wise unary operations template base.
// Mirrors tensorflow::UnaryOp<Device, Functor>
template <typename Derived, typename Functor>
class ND_EXPORT_SYMBOL UnaryOp : public UnaryOpShared {
 public:
  static absl::StatusOr<std::unique_ptr<Derived>> Create() {
    return std::make_unique<Derived>();
  }

  absl::StatusOr<xla_serving::HostBuffer> operator()(xla_serving::HostBuffer a) {
    xla_serving::HostBuffer out(GetUnaryOutputDType<Functor>(a.dtype()), a.shape());
    RETURN_IF_ERROR(DispatchUnary(a, &out, Functor{}, Derived::kOpName));
    return out;
  }

  absl::StatusOr<xla_serving::data::composite::TransformConfig> ToProto()
      const override {
    xla_serving::data::composite::TransformConfig config;
    config.set_op_name(Derived::kOpName);
    return config;
  }

 protected:
  absl::Status PropagateSignatureImpl(
      xla_serving::data::composite::api::ShapeInferenceContext& context)
      const override {
    return PropagateUnarySignature<Functor>(context, Derived::kOpName);
  }
};

// Convenience aliases matching transform naming conventions
template <typename Derived, typename Functor>
using BinaryTransformBase = BinaryOp<Derived, Functor>;

template <typename Derived, typename Functor>
using UnaryTransformBase = UnaryOp<Derived, Functor>;

}  // namespace xla_serving::data::kernel

// Registration Macros mirroring TensorFlow's REGISTER_* macros
#define ND_DECLARE_BINARY_OP(ClassName, FunctorType)                           \
  class ND_EXPORT_SYMBOL ClassName                                             \
      : public ::xla_serving::data::kernel::BinaryOp<ClassName, FunctorType> {          \
   public:                                                                     \
    static constexpr const char* kOpName = #ClassName;                         \
    static ::absl::StatusOr<std::unique_ptr<ClassName>> Create() {             \
      return ::xla_serving::data::kernel::BinaryOp<ClassName, FunctorType>::Create();   \
    }                                                                          \
    ::absl::StatusOr<::xla_serving::HostBuffer> operator()(::xla_serving::HostBuffer a,        \
                                                   ::xla_serving::HostBuffer b) {      \
      return ::xla_serving::data::kernel::BinaryOp<ClassName, FunctorType>::operator()( \
          std::move(a), std::move(b));                                         \
    }                                                                          \
  }

#define ND_DECLARE_UNARY_OP(ClassName, FunctorType)                           \
  class ND_EXPORT_SYMBOL ClassName                                            \
      : public ::xla_serving::data::kernel::UnaryOp<ClassName, FunctorType> {          \
   public:                                                                    \
    static constexpr const char* kOpName = #ClassName;                        \
    static ::absl::StatusOr<std::unique_ptr<ClassName>> Create() {            \
      return ::xla_serving::data::kernel::UnaryOp<ClassName, FunctorType>::Create();   \
    }                                                                         \
    ::absl::StatusOr<::xla_serving::HostBuffer> operator()(::xla_serving::HostBuffer a) {     \
      return ::xla_serving::data::kernel::UnaryOp<ClassName, FunctorType>::operator()( \
          std::move(a));                                                      \
    }                                                                         \
  }

#define ND_REGISTER_BINARY_OP(ClassName, ...)                               \
  ND_REGISTER_TRANSFORM(ClassName, ClassName::kOpName)                      \
      .SetDeserializer(                                                     \
          [](const ::xla_serving::data::composite::TransformConfig& config,     \
             ::absl::Span<const std::shared_ptr<::xla_serving::data::Resource>> \
                 resources)                                                 \
              -> ::absl::StatusOr<std::unique_ptr<                          \
                  ::xla_serving::data::composite::TransformBase>> {             \
            return ClassName::Create();                                     \
          })                                                                \
      .SetInputConstraints(                                                 \
          {{.allowed_dtypes = __VA_ARGS__}, {.allowed_dtypes = __VA_ARGS__}})

#define ND_REGISTER_UNARY_OP(ClassName, ...)                                \
  ND_REGISTER_TRANSFORM(ClassName, ClassName::kOpName)                      \
      .SetDeserializer(                                                     \
          [](const ::xla_serving::data::composite::TransformConfig& config,     \
             ::absl::Span<const std::shared_ptr<::xla_serving::data::Resource>> \
                 resources)                                                 \
              -> ::absl::StatusOr<std::unique_ptr<                          \
                  ::xla_serving::data::composite::TransformBase>> {             \
            return ClassName::Create();                                     \
          })                                                                \
      .SetInputConstraints({{.allowed_dtypes = __VA_ARGS__}})

#endif  // XLA_SERVING_DATA_KERNELS_OP_COMMON_H_

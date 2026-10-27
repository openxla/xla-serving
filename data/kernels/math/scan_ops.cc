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

#include "data/kernels/math/scan_ops.h"  // from @xla_serving

#define EIGEN_USE_THREADS

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "data/kernels/bounds_check.h"  // from @xla_serving
#include "data/kernels/macros.h"  // from @xla_serving
#include "data/kernels/math/scan_transform_config.pb.h"  // from @xla_serving
#include "data/kernels/util/shape_utils.h"  // from @xla_serving
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {

typedef Eigen::DefaultDevice CPUDevice;

template <typename Device, class T, typename Reducer, typename Tidx>
absl::Status ScanOp(const Device& d, const xla_serving::HostBuffer& input,
                    const xla_serving::HostBuffer& tensor_axis, xla_serving::HostBuffer& output,
                    bool reverse, bool exclusive) {
  if (!ShapeUtils::IsScalar(tensor_axis.shape())) {
    return absl::InvalidArgumentError(absl::StrCat(
        "ScanOp: axis must be a scalar, not ", tensor_axis.shape()));
  }

  const Tidx axis_arg =
      internal::SubtleMustCopy(*static_cast<const Tidx*>(tensor_axis.base()));
  const int64_t dims = input.shape().dims().size();
  const Tidx axis = (axis_arg < 0) ? dims + axis_arg : axis_arg;
  if (!FastBoundsCheck(axis, dims)) {
    return absl::InvalidArgumentError(
        absl::StrCat("ScanOp: Expected scan axis in the range [", -dims, ", ",
                     dims, "), but got ", axis));
  }

  const auto& output_shape = input.shape();

  // Exit early if there's nothing to compute
  if (output_shape.num_elements() == 0) return absl::OkStatus();

  Reducer reducer;

  // Dim reduction.
  int64_t reduced_shape[3] = {1, 1, 1};
  for (Tidx i = 0; i < axis; ++i) {
    reduced_shape[0] *= input.shape().dims()[i];
  }
  reduced_shape[1] = input.shape().dims()[axis];
  for (Tidx i = axis + 1; i < dims; ++i) {
    reduced_shape[2] *= input.shape().dims()[i];
  }

  ASSIGN_OR_RETURN(auto in_shaped,
                   (input.template shaped<T, 3>(reduced_shape)));
  ASSIGN_OR_RETURN(auto out_shaped,
                   (output.template shaped<T, 3>(reduced_shape)));

  functor::Scan<Device, Reducer, T>()(d, in_shaped, out_shaped, reducer,
                                      reverse, exclusive);
  return absl::OkStatus();
}

static absl::Status PropagateScanSignature(
    ::xla_serving::data::composite::api::ShapeInferenceContext& context,
    absl::string_view op_name, std::optional<int64_t> axis) {
  if (axis.has_value()) {
    if (context.inputs.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          op_name, " expects 1 input, got ", context.inputs.size()));
    }
    if (!context.inputs[0].IsSpec()) {
      return absl::InvalidArgumentError(
          absl::StrCat(op_name, " expects tensor input"));
    }
    context.outputs.push_back(context.inputs[0].spec());
    return absl::OkStatus();
  }
  if (context.inputs.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        op_name, " expects 2 inputs, got ", context.inputs.size()));
  }
  if (!context.inputs[0].IsSpec() || !context.inputs[1].IsSpec()) {
    return absl::InvalidArgumentError(
        absl::StrCat(op_name, " expects tensor inputs"));
  }
  const auto& input_spec = context.inputs[0].spec();
  const auto& axis_spec = context.inputs[1].spec();
  if (axis_spec.dtype.kind() != xla::ifrt::DType::kS32 &&
      axis_spec.dtype.kind() != xla::ifrt::DType::kS64) {
    return absl::InvalidArgumentError(absl::StrCat(
        op_name, ": axis must be int32 or int64, got ", axis_spec.dtype));
  }
  context.outputs.push_back(input_spec);
  return absl::OkStatus();
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::StatusOr<std::unique_ptr<Derived>>
ScanTransformBase<Derived, Reducer, kFloatOnly>::Create(
    bool exclusive, bool reverse, std::optional<int64_t> axis) {
  return absl::WrapUnique(new Derived(exclusive, reverse, axis));
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
template <typename Tidx>
absl::Status ScanTransformBase<Derived, Reducer, kFloatOnly>::DispatchByDType(
    const xla_serving::HostBuffer& input, const xla_serving::HostBuffer& axis,
    xla_serving::HostBuffer& output) const {
  CPUDevice d;
  switch (input.dtype().kind()) {
#define CASE_FLOAT(T, KIND)                                               \
  case KIND:                                                              \
    return ScanOp<CPUDevice, T, Reducer<T>, Tidx>(d, input, axis, output, \
                                                  reverse_, exclusive_);
#define CASE_NON_FLOAT(T, KIND)                                             \
  case KIND:                                                                \
    if constexpr (!kFloatOnly) {                                            \
      return ScanOp<CPUDevice, T, Reducer<T>, Tidx>(d, input, axis, output, \
                                                    reverse_, exclusive_);  \
    }                                                                       \
    break;
    ND_APPLY_FLOAT_CASE(CASE_FLOAT)
    ND_APPLY_INTEGRAL_CASE(CASE_NON_FLOAT)
    ND_APPLY_COMPLEX_CASE(CASE_NON_FLOAT)
#undef CASE_FLOAT
#undef CASE_NON_FLOAT
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      Derived::kOpName, ": unsupported input dtype ", input.dtype()));
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::StatusOr<xla_serving::HostBuffer>
ScanTransformBase<Derived, Reducer, kFloatOnly>::operator()(
    const xla_serving::HostBuffer& input) const {
  int64_t axis_val = axis_.value_or(0);
  xla_serving::HostBuffer axis_hb(xla::ifrt::DType(xla::ifrt::DType::kS64),
                          xla::ifrt::Shape({}));
  *static_cast<int64_t*>(axis_hb.base()) = axis_val;
  return (*this)(input, axis_hb);
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::StatusOr<xla_serving::HostBuffer>
ScanTransformBase<Derived, Reducer, kFloatOnly>::operator()(
    const xla_serving::HostBuffer& input, const xla_serving::HostBuffer& axis) const {
  xla_serving::HostBuffer output(input.dtype(), input.shape());
  if (axis.dtype().kind() == xla::ifrt::DType::kS32) {
    RETURN_IF_ERROR(DispatchByDType<int32_t>(input, axis, output));
  } else if (axis.dtype().kind() == xla::ifrt::DType::kS64) {
    RETURN_IF_ERROR(DispatchByDType<int64_t>(input, axis, output));
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "ScanOp: axis must be int32 or int64, got ", axis.dtype()));
  }
  return output;
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::Status ScanTransformBase<Derived, Reducer, kFloatOnly>::Execute(
    ::xla_serving::data::composite::api::ExecuteContext& context) {
  if (axis_.has_value()) {
    if (context.inputs.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          Derived::kOpName, ": expected 1 input, got ", context.inputs.size()));
    }
    ASSIGN_OR_RETURN(
        auto out, (*this)(context.inputs[0].template Get<xla_serving::HostBuffer>()));
    ::xla_serving::data::composite::api::Value v;
    v.Set(std::move(out));
    context.outputs.push_back(std::move(v));
    return absl::OkStatus();
  }
  if (context.inputs.size() != 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        Derived::kOpName, ": expected 2 inputs, got ", context.inputs.size()));
  }
  ASSIGN_OR_RETURN(auto out,
                   (*this)(context.inputs[0].template Get<xla_serving::HostBuffer>(),
                           context.inputs[1].template Get<xla_serving::HostBuffer>()));
  ::xla_serving::data::composite::api::Value v;
  v.Set(std::move(out));
  context.outputs.push_back(std::move(v));
  return absl::OkStatus();
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::Span<const ::xla_serving::data::composite::api::TypeInfo>
ScanTransformBase<Derived, Reducer, kFloatOnly>::InputTypes() const {
  static const auto* const kOneInput =
      new std::vector<::xla_serving::data::composite::api::TypeInfo>{
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>()};
  static const auto* const kTwoInputs =
      new std::vector<::xla_serving::data::composite::api::TypeInfo>{
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>(),
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>()};
  return axis_.has_value() ? *kOneInput : *kTwoInputs;
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::Span<const ::xla_serving::data::composite::api::TypeInfo>
ScanTransformBase<Derived, Reducer, kFloatOnly>::OutputTypes() const {
  static const auto* const kOneOutput =
      new std::vector<::xla_serving::data::composite::api::TypeInfo>{
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>()};
  return *kOneOutput;
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::Status
ScanTransformBase<Derived, Reducer, kFloatOnly>::PropagateSignatureImpl(
    ::xla_serving::data::composite::api::ShapeInferenceContext& context) const {
  return PropagateScanSignature(context, Derived::kOpName, axis_);
}

template <typename Derived, template <typename> class Reducer, bool kFloatOnly>
absl::StatusOr<::xla_serving::data::composite::TransformConfig>
ScanTransformBase<Derived, Reducer, kFloatOnly>::ToProto() const {
  ::xla_serving::data::composite::TransformConfig config;
  config.set_op_name(Derived::kOpName);
  ScanTransformConfig detail_proto;
  detail_proto.set_exclusive(exclusive_);
  detail_proto.set_reverse(reverse_);
  if (axis_.has_value()) {
    detail_proto.set_axis(*axis_);
  }
  config.mutable_detail()->PackFrom(detail_proto);
  return config;
}

template <typename Transform>
static absl::StatusOr<
    std::unique_ptr<::xla_serving::data::composite::TransformBase>>
DeserializeScanTransform(const ScanTransformConfig& detail_proto) {
  std::optional<int64_t> axis = std::nullopt;
  if (detail_proto.has_axis()) {
    axis = detail_proto.axis();
  }
  return Transform::Create(detail_proto.exclusive(), detail_proto.reverse(),
                           axis);
}

template class ScanTransformBase<CumsumTransform, Eigen::internal::SumReducer>;
template class ScanTransformBase<CumprodTransform,
                                 Eigen::internal::ProdReducer>;
template class ScanTransformBase<CumulativeLogsumexpTransform,
                                 functor::LogSumExpReducer,
                                 /*kFloatOnly=*/true>;

ND_REGISTER_TRANSFORM(CumsumTransform, "CumsumTransform")
    .SetDeserializerWithDetail<ScanTransformConfig>(
        &DeserializeScanTransform<CumsumTransform>);

ND_REGISTER_TRANSFORM(CumprodTransform, "CumprodTransform")
    .SetDeserializerWithDetail<ScanTransformConfig>(
        &DeserializeScanTransform<CumprodTransform>);

ND_REGISTER_TRANSFORM(CumulativeLogsumexpTransform,
                      "CumulativeLogsumexpTransform")
    .SetDeserializerWithDetail<ScanTransformConfig>(
        &DeserializeScanTransform<CumulativeLogsumexpTransform>);

}  // namespace xla_serving::data::kernel

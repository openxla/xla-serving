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

#include "data/kernels/math/reduction_ops_common.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "data/kernels/macros.h"  // from @xla_serving
#include "data/kernels/math/reduction_ops.h"  // from @xla_serving
#include "data/kernels/math/reduction_transform_config.pb.h"  // from @xla_serving
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {

xla::ifrt::Shape ReductionHelper::out_reshape() const {
  return xla::ifrt::Shape(out_reshape_);
}

// The final output shape must be allocated with this shape.
xla::ifrt::Shape ReductionHelper::out_shape() const {
  return xla::ifrt::Shape(out_shape_);
}

xla::ifrt::Shape ReductionHelper::shuffled_shape() {
  const int dims = data_reshape_.size();
  absl::InlinedVector<int64_t, 4> dims_vec;
  for (int i = reduce_first_axis_; i < dims; i += 2) {
    dims_vec.push_back(data_reshape_[i]);
  }
  for (int i = !reduce_first_axis_; i < dims; i += 2) {
    dims_vec.push_back(data_reshape_[i]);
  }
  return xla::ifrt::Shape(dims_vec);
}

absl::InlinedVector<int32_t, 8> ReductionHelper::permutation() {
  const int dims = data_reshape_.size();
  const int unreduced_dims = (dims + !reduce_first_axis_) / 2;
  absl::InlinedVector<int32_t, 8> perm(dims);
  for (int i = 0; i < unreduced_dims; i++) {
    perm[i] = 2 * i + reduce_first_axis_;
  }
  for (int i = unreduced_dims; i < dims; i++) {
    perm[i] = 2 * (i - unreduced_dims) + !reduce_first_axis_;
  }
  return perm;
}

template <typename Tperm>
absl::Status SimplifyHelper(const xla::ifrt::Shape& data_shape,
                            const xla_serving::HostBuffer& axis,
                            absl::InlinedVector<bool, 4>& bitmap) {
  auto axis_vec = axis.flat<Tperm>();
  const int64_t data_dims = data_shape.dims().size();
  for (int64_t i = 0; i < axis.shape().num_elements(); ++i) {
    Tperm index = axis_vec(i);
    if (index < -data_dims || index >= data_dims) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid reduction dimension (", index,
                       " for input with ", data_dims, " dimension(s))"));
    }
    index = (index + data_dims) % data_dims;
    if (bitmap[index]) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Invalid reduction arguments: Axes contains duplicate dimension: ",
          index));
    }
    bitmap[index] = true;
  }
  return absl::OkStatus();
}

absl::Status ReductionHelper::Simplify(const xla_serving::HostBuffer& data,
                                       const xla_serving::HostBuffer& axis,
                                       const bool keep_dims) {
  return Simplify(data.shape(), axis, keep_dims);
}

absl::Status ReductionHelper::Simplify(const xla::ifrt::Shape& data_shape,
                                       const xla_serving::HostBuffer& axis,
                                       const bool keep_dims) {
  if (axis.shape().dims().size() > 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("axis must be at most rank 1, got shape ", axis.shape()));
  }
  const int data_dims = static_cast<int>(data_shape.dims().size());
  // bitmap[i] indicates whether to reduce data along i-th axis.
  absl::InlinedVector<bool, 4> bitmap(data_dims, false);
  if (axis.dtype().kind() == xla::ifrt::DType::kS32) {
    RETURN_IF_ERROR(SimplifyHelper<int32_t>(data_shape, axis, bitmap));
  } else if (axis.dtype().kind() == xla::ifrt::DType::kS64) {
    RETURN_IF_ERROR(SimplifyHelper<int64_t>(data_shape, axis, bitmap));
  } else {
    return absl::InvalidArgumentError(absl::StrCat(
        "Reduction axis must be int32 or int64, got ", axis.dtype()));
  }
  // Output tensor's dim sizes.
  out_shape_.clear();
  for (int i = 0; i < data_dims; ++i) {
    if (!bitmap[i]) {
      // If we are not reducing along dimension i.
      out_shape_.push_back(data_shape.dims()[i]);
    } else if (keep_dims) {
      // We are reducing along dimension i, but we want to keep the
      // same number of dimensions, so we set the dimension of i to
      // '1'.
      out_shape_.push_back(1);
    }
  }

  // Depending on bitmap[i] and bitmap[i-1], we can collapse axis of
  // the input data before doing the reduction on the resulting
  // tensor.  The shape of the reduction is a reshape of the final
  // output.

  // We'll skip the leading 1s.
  int dim_index = 0;
  for (; dim_index < data_dims; ++dim_index) {
    if (data_shape.dims()[dim_index] != 1) break;
  }
  data_reshape_.clear();
  out_reshape_.clear();
  if (dim_index >= data_dims) {
    // Special case. The input is essentially a scalar.
    reduce_first_axis_ = true;
  } else {
    // Starting from the (dim_index)-th dimension, dimensions
    // alternates between runs that need to be reduced and runs that
    // don't.
    //
    // NOTE: If a dimension has size 1, we group it as the current
    // run so that we can minimize the number of runs.
    //
    // E.g., when we want to reduce a tensor of shape [2, 1, 3, 1,
    // 5] by axes = [1, 4], we should treat the tensor as a [6, 5]
    // and reduce by axes = [1] (i.e., the output is shape [6]).
    reduce_first_axis_ = bitmap[dim_index];
    data_reshape_.push_back(data_shape.dims()[dim_index]);
    ++dim_index;
    for (; dim_index < data_dims; ++dim_index) {
      const auto size = data_shape.dims()[dim_index];
      if (size == 1) {
        bitmap[dim_index] = bitmap[dim_index - 1];
      }
      if (bitmap[dim_index - 1] != bitmap[dim_index]) {
        // Starts a new run of reduce or !reduce.
        data_reshape_.push_back(size);
      } else {
        // Continue a run of reduce or !reduce.
        data_reshape_.back() *= size;
      }
    }
    // If reduce_first_axis_ is true (input's dimension 0, 2, 4, etc
    // are reduced), data_reshape_[1, 3, 5, ...]  is out_reshape_,
    // otherwise, data_reshape_[0, 2, 4, ...] is.
    for (size_t i = reduce_first_axis_ ? 1 : 0; i < data_reshape_.size();
         i += 2) {
      out_reshape_.push_back(data_reshape_[i]);
    }
  }

  VLOG(1) << "data reshape: " << absl::StrJoin(data_reshape_, ",");
  VLOG(1) << "out  reshape: " << absl::StrJoin(out_reshape_, ",");
  VLOG(1) << "out    shape: " << absl::StrJoin(out_shape_, ",");
  return absl::OkStatus();
}

namespace {

xla_serving::HostBuffer AxesVectorToHostBuffer(absl::Span<const int64_t> axes) {
  xla_serving::HostBuffer axes_hb(
      xla::ifrt::DType(xla::ifrt::DType::kS64),
      xla::ifrt::Shape({static_cast<int64_t>(axes.size())}));
  int64_t* ptr = static_cast<int64_t*>(axes_hb.base());
  for (size_t i = 0; i < axes.size(); ++i) {
    ptr[i] = axes[i];
  }
  return axes_hb;
}

}  // namespace

template <typename Derived>
absl::StatusOr<std::unique_ptr<Derived>>
ReductionTransformBase<Derived>::Create(std::vector<int64_t> axis,
                                        bool keep_dims) {
  return absl::WrapUnique(new Derived(std::move(axis), keep_dims));
}

template <typename Derived>
absl::StatusOr<xla_serving::HostBuffer> ReductionTransformBase<Derived>::operator()(
    const xla_serving::HostBuffer& data) const {
  xla_serving::HostBuffer axes_hb = AxesVectorToHostBuffer(axis_);
  xla_serving::HostBuffer output;
  RETURN_IF_ERROR(DispatchByDType(data, axes_hb, &output));
  return output;
}

template <typename Derived>
absl::Status ReductionTransformBase<Derived>::Execute(
    ::xla_serving::data::composite::api::ExecuteContext& context) {
  if (context.inputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        Derived::kOpName, ": expected 1 input, got ", context.inputs.size()));
  }
  ASSIGN_OR_RETURN(auto out,
                   (*this)(context.inputs[0].template Get<xla_serving::HostBuffer>()));
  ::xla_serving::data::composite::api::Value v;
  v.Set(std::move(out));
  context.outputs.push_back(std::move(v));
  return absl::OkStatus();
}

template <typename Derived>
absl::Span<const ::xla_serving::data::composite::api::TypeInfo>
ReductionTransformBase<Derived>::InputTypes() const {
  static const auto* const kOneInput =
      new std::vector<::xla_serving::data::composite::api::TypeInfo>{
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>()};
  return *kOneInput;
}

template <typename Derived>
absl::Span<const ::xla_serving::data::composite::api::TypeInfo>
ReductionTransformBase<Derived>::OutputTypes() const {
  static const auto* const kOneOutput =
      new std::vector<::xla_serving::data::composite::api::TypeInfo>{
          ::xla_serving::data::composite::api::CreateTypeInfo<xla_serving::HostBuffer>()};
  return *kOneOutput;
}

template <typename Derived>
absl::Status ReductionTransformBase<Derived>::PropagateSignatureImpl(
    ::xla_serving::data::composite::api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        Derived::kOpName, " expects 1 input, got ", context.inputs.size()));
  }
  if (!context.inputs[0].IsSpec()) {
    return absl::InvalidArgumentError(
        absl::StrCat(Derived::kOpName, " expects tensor input"));
  }
  const auto& input_spec = context.inputs[0].spec();
  xla_serving::HostBuffer axes_hb = AxesVectorToHostBuffer(axis_);
  ReductionHelper helper;
  RETURN_IF_ERROR(helper.Simplify(input_spec.shape, axes_hb, keep_dims_));
  auto out_spec = input_spec;
  out_spec.shape = helper.out_shape();
  context.outputs.push_back(out_spec);
  return absl::OkStatus();
}

template <typename Derived>
absl::StatusOr<::xla_serving::data::composite::TransformConfig>
ReductionTransformBase<Derived>::ToProto() const {
  ::xla_serving::data::composite::TransformConfig config;
  config.set_op_name(Derived::kOpName);
  ReductionTransformConfig detail_proto;
  detail_proto.mutable_axis()->Assign(axis_.begin(), axis_.end());
  detail_proto.set_keep_dims(keep_dims_);
  config.mutable_detail()->PackFrom(detail_proto);
  return config;
}

template <typename Derived, template <typename> class Reducer>
absl::Status BoolReductionTransformBase<Derived, Reducer>::DispatchByDType(
    const xla_serving::HostBuffer& data, const xla_serving::HostBuffer& axes,
    xla_serving::HostBuffer* output) const {
  CPUDevice d;
  if (data.dtype().kind() == xla::ifrt::DType::Kind::kPred) {
    return ReductionOp<CPUDevice, bool, int64_t, Reducer<bool>>::Compute(
        d, data, axes, this->keep_dims_, output);
  }
  return absl::InvalidArgumentError(absl::StrCat(
      Derived::kOpName, ": unsupported input dtype ", data.dtype()));
}

template <typename Derived, template <typename> class Reducer>
absl::Status RealReductionTransformBase<Derived, Reducer>::DispatchByDType(
    const xla_serving::HostBuffer& data, const xla_serving::HostBuffer& axes,
    xla_serving::HostBuffer* output) const {
  CPUDevice d;
  switch (data.dtype().kind()) {
#define CASE_TYPE(T, KIND)                                          \
  case KIND:                                                        \
    return ReductionOp<CPUDevice, T, int64_t, Reducer<T>>::Compute( \
        d, data, axes, this->keep_dims_, output);
    ND_APPLY_NUMBER_CASE(CASE_TYPE)
#undef CASE_TYPE
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      Derived::kOpName, ": unsupported input dtype ", data.dtype()));
}

template <typename Derived, template <typename> class Reducer>
absl::Status AllNumberReductionTransformBase<Derived, Reducer>::DispatchByDType(
    const xla_serving::HostBuffer& data, const xla_serving::HostBuffer& axes,
    xla_serving::HostBuffer* output) const {
  CPUDevice d;
  switch (data.dtype().kind()) {
#define CASE_TYPE(T, KIND)                                          \
  case KIND:                                                        \
    return ReductionOp<CPUDevice, T, int64_t, Reducer<T>>::Compute( \
        d, data, axes, this->keep_dims_, output);
    ND_APPLY_ALL_NUMBER_CASE(CASE_TYPE)
#undef CASE_TYPE
    default:
      break;
  }
  return absl::InvalidArgumentError(absl::StrCat(
      Derived::kOpName, ": unsupported input dtype ", data.dtype()));
}

template <typename Transform>
absl::StatusOr<std::unique_ptr<::xla_serving::data::composite::TransformBase>>
DeserializeReductionTransform(
    const ::xla_serving::data::composite::TransformConfig& config,
    absl::Span<const std::shared_ptr<::xla_serving::data::Resource>>
    /*resources*/) {
  ReductionTransformConfig detail_proto;
  if (config.has_detail()) {
    config.detail().UnpackTo(&detail_proto);
  }
  std::vector<int64_t> axis(detail_proto.axis().begin(),
                            detail_proto.axis().end());
  return Transform::Create(std::move(axis), detail_proto.keep_dims());
}

#define INSTANTIATE_REDUCTION_TRANSFORM(BaseTemplate, Transform, Reducer) \
  template class ReductionTransformBase<Transform>;                       \
  template class BaseTemplate<Transform, Reducer>;                        \
  template absl::StatusOr<                                                \
      std::unique_ptr<::xla_serving::data::composite::TransformBase>>         \
  DeserializeReductionTransform<Transform>(                               \
      const ::xla_serving::data::composite::TransformConfig&,                 \
      absl::Span<const std::shared_ptr<::xla_serving::data::Resource>>)

INSTANTIATE_REDUCTION_TRANSFORM(BoolReductionTransformBase, ReduceAllTransform,
                                functor::AndReducer);
INSTANTIATE_REDUCTION_TRANSFORM(BoolReductionTransformBase, ReduceAnyTransform,
                                functor::OrReducer);
INSTANTIATE_REDUCTION_TRANSFORM(AllNumberReductionTransformBase,
                                ReduceEuclideanNormTransform,
                                functor::EuclideanNormReducer);
INSTANTIATE_REDUCTION_TRANSFORM(RealReductionTransformBase, ReduceMaxTransform,
                                functor::MaxReducer);
INSTANTIATE_REDUCTION_TRANSFORM(AllNumberReductionTransformBase,
                                ReduceMeanTransform, functor::MeanReducer);
INSTANTIATE_REDUCTION_TRANSFORM(RealReductionTransformBase, ReduceMinTransform,
                                functor::MinReducer);
INSTANTIATE_REDUCTION_TRANSFORM(AllNumberReductionTransformBase,
                                ReduceProdTransform,
                                Eigen::internal::ProdReducer);
INSTANTIATE_REDUCTION_TRANSFORM(AllNumberReductionTransformBase,
                                ReduceSumTransform,
                                Eigen::internal::SumReducer);
#undef INSTANTIATE_REDUCTION_TRANSFORM

}  // namespace xla_serving::data::kernel

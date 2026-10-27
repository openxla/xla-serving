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

// This is an internal header file intended to only be included as the
// front-matter in the implementation files of various reduction ops.  It
// is a header file because we split the various reduction ops into their
// own compilation units to get more parallelism in compilation.

#ifndef XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_COMMON_H_
#define XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_COMMON_H_

#define EIGEN_USE_THREADS

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "data/kernels/macros.h"  // from @xla_serving
#include "data/kernels/math/reduction_ops.h"  // from @xla_serving
#include "data/kernels/math/reduction_transform_config.pb.h"  // from @xla_serving
#include "data/kernels/transpose_functor.h"  // from @xla_serving
#include "data/kernels/util/ttypes.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {
#pragma GCC visibility push(default)

typedef Eigen::DefaultDevice CPUDevice;

template <typename Device>
struct Constants {
  // Derive Index type. int (32-bit) or long (64-bit) depending on the
  // compile-time configuration. "float" here is not relevant.
  // TODO: Moves the definition to TTypes.
  typedef TTypes<float>::Tensor::Index Index;
  Eigen::array<Index, 1> zero;
  Eigen::array<Index, 1> one;
  Eigen::array<Index, 2> zero_two;

  Constants() {
    zero[0] = 0;
    one[0] = 1;
    zero_two[0] = 0;
    zero_two[1] = 2;
  }
};

struct ConstantsBase {
  const Eigen::IndexList<Eigen::type2index<0>> zero;
  const Eigen::IndexList<Eigen::type2index<1>> one;
  const Eigen::IndexList<Eigen::type2index<0>, Eigen::type2index<2>> zero_two;
};
template <>
struct Constants<CPUDevice> : ConstantsBase {};

class ReductionHelper {
 public:
  ReductionHelper() : reduce_first_axis_(false) {}

  absl::Status Simplify(const xla_serving::HostBuffer& data,
                        const xla_serving::HostBuffer& axis, bool keep_dims);

  absl::Status Simplify(const xla::ifrt::Shape& data_shape,
                        const xla_serving::HostBuffer& axis, bool keep_dims);

  // We need to do roughly:
  //   tmp_out = allocate(out_reshape())
  //   tmp_out.reshape(out_reshape) = data.reshape(data_reshape).reduce(axes)
  //   out = tmp_out.reshape(out_shape)

  // The reduction result must be allocated with this shape.
  xla::ifrt::Shape out_reshape() const;

  // The final output shape must be allocated with this shape.
  xla::ifrt::Shape out_shape() const;

  // The reduction is on a reshaped tensor of this rank.
  int ndims() const { return data_reshape_.size(); }

  // True if need to reduce the 0-th dimension.
  bool reduce_first_axis() const { return reduce_first_axis_; }

  // The output is reshaped.
  template <typename T, int N>
  absl::StatusOr<typename TTypes<T, N>::Tensor> out(xla_serving::HostBuffer* out) {
    return out->shaped<T, N>(out_reshape_);
  }

  // The input is reshaped.
  template <typename T, int N>
  absl::StatusOr<typename TTypes<T, N>::ConstTensor> in(
      const xla_serving::HostBuffer& data) {
    return data.shaped<T, N>(data_reshape_);
  }

  // Shape of shuffled input
  xla::ifrt::Shape data_reshape() const {
    return xla::ifrt::Shape(data_reshape_);
  }

  // Shape with all reduction dimensions at the end
  xla::ifrt::Shape shuffled_shape();

  // Permutation of reduced dims needed to put reduction dimensions at the end
  absl::InlinedVector<int32_t, 8> permutation();

 private:
  bool reduce_first_axis_;  // True if need to reduce the 0-th dimension.
  absl::InlinedVector<int64_t, 4>
      data_reshape_;                           // Reshape data before reduction.
  absl::InlinedVector<int64_t, 4> out_shape_;  // The final output shape.
  absl::InlinedVector<int64_t, 4>
      out_reshape_;  // Reshape output for reduction.
};

// For operations where the output is a reduction function along some
// dimensions of the input.
template <typename Device, class T, typename Tperm, typename Reducer>
class ReductionOp {
 public:
  static absl::Status Compute(const Device& d, const xla_serving::HostBuffer& data,
                              const xla_serving::HostBuffer& axes, bool keep_dims,
                              xla_serving::HostBuffer* out_buf) {
    ReductionHelper helper;
    RETURN_IF_ERROR(helper.Simplify(data, axes, keep_dims));
    if (helper.ndims() < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Reduction helper returned negative ndims. Input shape: ",
          data.shape()));
    }

    bool is_scalar_identity = functor::ReducerTraits<Reducer>::IsScalarIdentity;
    bool is_trivial = helper.ndims() == 0 ||
                      (helper.ndims() == 1 && !helper.reduce_first_axis());
    if (is_scalar_identity && is_trivial) {
      // Special case. Reduces nothing and does not alter the input values.
      ASSIGN_OR_RETURN(*out_buf, data.Reshape(helper.out_shape()));
      return absl::OkStatus();
    }

    const xla::ifrt::DType out_dtype = data.dtype();
    using Functor = functor::ReduceFunctor<Device, Reducer>;
    Constants<Device> constants;
    Reducer reducer;

    if (data.shape().num_elements() > 0 && is_trivial && !is_scalar_identity) {
      xla_serving::HostBuffer tmp_out(out_dtype,
                              xla::ifrt::Shape({data.shape().num_elements()}));
      ASSIGN_OR_RETURN(auto in_shaped,
                       (data.shaped<T, 2>({1, data.shape().num_elements()})));
      Functor::Reduce(d, tmp_out.flat<T>(), in_shaped, constants.zero, reducer);
      ASSIGN_OR_RETURN(*out_buf, tmp_out.Reshape(helper.out_shape()));
      return absl::OkStatus();
    }

    // A temporary tensor whose size matches the size of the reduced
    // output.
    xla_serving::HostBuffer tmp_out(out_dtype, helper.out_reshape());

    if (tmp_out.shape().num_elements() == 0) {
      // Nothing to do, fall through to final reshaping.
    } else if (data.shape().num_elements() == 0) {
      // Degenerate reduction where the input is empty but the output is
      // nonempty (thus tmp_out.NumElements() > 0), and we must fill the
      // output with identity elements.  Example: tf.reduce_sum(tf.zeros((0,
      // 3)), [0]). Eigen sometimes crashes in this case, so we do it
      // manually.
      Functor::FillIdentity(d, tmp_out.flat<T>(), reducer);
    } else if ((helper.ndims() == 1) && helper.reduce_first_axis()) {
      // Reduce to a scalar.
      ASSIGN_OR_RETURN(auto out_t, (helper.out<T, 0>(&tmp_out)));
      ASSIGN_OR_RETURN(auto in_t, (helper.in<T, 1>(data)));
      Functor::Reduce(d, out_t, in_t, constants.zero, reducer);
    } else if ((helper.ndims() == 2) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 1st dimension.
      ASSIGN_OR_RETURN(auto out_t, (helper.out<T, 1>(&tmp_out)));
      ASSIGN_OR_RETURN(auto in_t, (helper.in<T, 2>(data)));
      Functor::Reduce(d, out_t, in_t, constants.zero, reducer);
    } else if ((helper.ndims() == 2) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a matrix along 2nd dimension.
      ASSIGN_OR_RETURN(auto out_t, (helper.out<T, 1>(&tmp_out)));
      ASSIGN_OR_RETURN(auto in_t, (helper.in<T, 2>(data)));
      Functor::Reduce(d, out_t, in_t, constants.one, reducer);
    } else if ((helper.ndims() == 3) && helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 1st and 3rd
      // dimensions.
      ASSIGN_OR_RETURN(auto out_t, (helper.out<T, 1>(&tmp_out)));
      ASSIGN_OR_RETURN(auto in_t, (helper.in<T, 3>(data)));
      Functor::Reduce(d, out_t, in_t, constants.zero_two, reducer);
    } else if ((helper.ndims() == 3) && !helper.reduce_first_axis()) {
      // Can be viewed as a reduction of a 3D tensor along 2nd dimension.
      ASSIGN_OR_RETURN(auto out_t, (helper.out<T, 2>(&tmp_out)));
      ASSIGN_OR_RETURN(auto in_t, (helper.in<T, 3>(data)));
      Functor::Reduce(d, out_t, in_t, constants.one, reducer);
    } else {
      // If we don't hit one of the cases above, transpose the data so that
      // all reduced dimensions are last and reuse the 2-D -> 1-D case.
      ASSIGN_OR_RETURN(xla_serving::HostBuffer data_reshaped,
                       data.Reshape(helper.data_reshape()));
      xla_serving::HostBuffer shuffled(data.dtype(), helper.shuffled_shape());
      RETURN_IF_ERROR(
          DoTranspose(d, data_reshaped, helper.permutation(), &shuffled));
      const int64_t unreduced = tmp_out.shape().num_elements();
      const int64_t reduced = shuffled.shape().num_elements() / unreduced;
      const xla_serving::HostBuffer& const_shuffled = shuffled;
      ASSIGN_OR_RETURN(auto in_shaped,
                       (const_shuffled.shaped<T, 2>({unreduced, reduced})));
      Functor::Reduce(d, tmp_out.flat<T>(), in_shaped, constants.one, reducer);
    }

    // Set the real output using the contents of the reduction but the
    // real expected output shape.  The number of elements should
    // match between the two shapes.
    ASSIGN_OR_RETURN(*out_buf, tmp_out.Reshape(helper.out_shape()));
    return absl::OkStatus();
  }
};

namespace functor {

template <typename Device, typename Reducer>
struct ReduceFunctorBase {
  template <typename OUT_T, typename IN_T, typename ReductionAxes>
  static void Reduce(const Device& d, OUT_T out, IN_T in,
                     const ReductionAxes& reduction_axes,
                     const Reducer& reducer) {
    ReduceEigenImpl<Device, OUT_T, IN_T, ReductionAxes, Reducer> reducer_impl;
    reducer_impl(d, out, in, reduction_axes, reducer);
  }

  template <typename OUT_T>
  static void FillIdentity(const Device& d, OUT_T out, const Reducer& reducer) {
    FillIdentityEigenImpl(d, out, reducer);
  }
};

template <typename Reducer>
struct ReduceFunctor<CPUDevice, Reducer>
    : ReduceFunctorBase<CPUDevice, Reducer> {};

}  // namespace functor

template <typename Derived>
class ReductionTransformBase
    : public ::xla_serving::data::composite::TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<Derived>> Create(
      std::vector<int64_t> axis = {}, bool keep_dims = false);

  ~ReductionTransformBase() override = default;

  absl::StatusOr<xla_serving::HostBuffer> operator()(const xla_serving::HostBuffer& data) const;

  absl::Status Execute(
      ::xla_serving::data::composite::api::ExecuteContext& context) override;

  absl::Span<const ::xla_serving::data::composite::api::TypeInfo> InputTypes()
      const override;
  absl::Span<const ::xla_serving::data::composite::api::TypeInfo> OutputTypes()
      const override;

  absl::StatusOr<::xla_serving::data::composite::TransformConfig> ToProto()
      const override;

 protected:
  explicit ReductionTransformBase(std::vector<int64_t> axis, bool keep_dims)
      : axis_(std::move(axis)), keep_dims_(keep_dims) {}

  absl::Status PropagateSignatureImpl(
      ::xla_serving::data::composite::api::ShapeInferenceContext& context)
      const override;

  virtual absl::Status DispatchByDType(const xla_serving::HostBuffer& data,
                                       const xla_serving::HostBuffer& axes,
                                       xla_serving::HostBuffer* output) const = 0;

  std::vector<int64_t> axis_;
  bool keep_dims_;
};

template <typename Derived, template <typename> class Reducer>
class BoolReductionTransformBase : public ReductionTransformBase<Derived> {
 protected:
  using ReductionTransformBase<Derived>::ReductionTransformBase;

  absl::Status DispatchByDType(const xla_serving::HostBuffer& data,
                               const xla_serving::HostBuffer& axes,
                               xla_serving::HostBuffer* output) const override;
};

template <typename Derived, template <typename> class Reducer>
class RealReductionTransformBase : public ReductionTransformBase<Derived> {
 protected:
  using ReductionTransformBase<Derived>::ReductionTransformBase;

  absl::Status DispatchByDType(const xla_serving::HostBuffer& data,
                               const xla_serving::HostBuffer& axes,
                               xla_serving::HostBuffer* output) const override;
};

template <typename Derived, template <typename> class Reducer>
class AllNumberReductionTransformBase : public ReductionTransformBase<Derived> {
 protected:
  using ReductionTransformBase<Derived>::ReductionTransformBase;

  absl::Status DispatchByDType(const xla_serving::HostBuffer& data,
                               const xla_serving::HostBuffer& axes,
                               xla_serving::HostBuffer* output) const override;
};

template <typename Transform>
absl::StatusOr<std::unique_ptr<::xla_serving::data::composite::TransformBase>>
DeserializeReductionTransform(const ReductionTransformConfig& detail_proto);

class ReduceAllTransform
    : public BoolReductionTransformBase<ReduceAllTransform,
                                        functor::AndReducer> {
 public:
  static constexpr const char* kOpName = "ReduceAllTransform";

 private:
  friend class ReductionTransformBase<ReduceAllTransform>;
  using BoolReductionTransformBase::BoolReductionTransformBase;
};

class ReduceAnyTransform
    : public BoolReductionTransformBase<ReduceAnyTransform,
                                        functor::OrReducer> {
 public:
  static constexpr const char* kOpName = "ReduceAnyTransform";

 private:
  friend class ReductionTransformBase<ReduceAnyTransform>;
  using BoolReductionTransformBase::BoolReductionTransformBase;
};

class ReduceEuclideanNormTransform
    : public AllNumberReductionTransformBase<ReduceEuclideanNormTransform,
                                             functor::EuclideanNormReducer> {
 public:
  static constexpr const char* kOpName = "ReduceEuclideanNormTransform";

 private:
  friend class ReductionTransformBase<ReduceEuclideanNormTransform>;
  using AllNumberReductionTransformBase::AllNumberReductionTransformBase;
};

class ReduceMaxTransform
    : public RealReductionTransformBase<ReduceMaxTransform,
                                        functor::MaxReducer> {
 public:
  static constexpr const char* kOpName = "ReduceMaxTransform";

 private:
  friend class ReductionTransformBase<ReduceMaxTransform>;
  using RealReductionTransformBase::RealReductionTransformBase;
};

class ReduceMeanTransform
    : public AllNumberReductionTransformBase<ReduceMeanTransform,
                                             functor::MeanReducer> {
 public:
  static constexpr const char* kOpName = "ReduceMeanTransform";

 private:
  friend class ReductionTransformBase<ReduceMeanTransform>;
  using AllNumberReductionTransformBase::AllNumberReductionTransformBase;
};

class ReduceMinTransform
    : public RealReductionTransformBase<ReduceMinTransform,
                                        functor::MinReducer> {
 public:
  static constexpr const char* kOpName = "ReduceMinTransform";

 private:
  friend class ReductionTransformBase<ReduceMinTransform>;
  using RealReductionTransformBase::RealReductionTransformBase;
};

class ReduceProdTransform
    : public AllNumberReductionTransformBase<ReduceProdTransform,
                                             Eigen::internal::ProdReducer> {
 public:
  static constexpr const char* kOpName = "ReduceProdTransform";

 private:
  friend class ReductionTransformBase<ReduceProdTransform>;
  using AllNumberReductionTransformBase::AllNumberReductionTransformBase;
};

class ReduceSumTransform
    : public AllNumberReductionTransformBase<ReduceSumTransform,
                                             Eigen::internal::SumReducer> {
 public:
  static constexpr const char* kOpName = "ReduceSumTransform";

 private:
  friend class ReductionTransformBase<ReduceSumTransform>;
  using AllNumberReductionTransformBase::AllNumberReductionTransformBase;
};

#pragma GCC visibility pop
}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_MATH_REDUCTION_OPS_COMMON_H_

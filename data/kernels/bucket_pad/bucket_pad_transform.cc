// Copyright 2024 The Neptune Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "data/kernels/bucket_pad/bucket_pad_transform.h"  // from @xla_serving

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/bucket_pad/bucket_pad_transform.pb.h"  // from @xla_serving
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/utils/default_value_utils.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune::data::composite::transforms {
namespace {

template <typename T>
absl::Status ExecuteTypedPad(const xla_serving::HostBuffer& input,
                             xla_serving::HostBuffer& output, int norm_axis,
                             int64_t in_dim, int64_t target_dim,
                             const utils::DefaultValue& pad_value_proto) {
  T pad_val = T(0);
  if (pad_value_proto.has_int_value() || pad_value_proto.has_float_value() ||
      pad_value_proto.has_double_value() || pad_value_proto.has_bool_value() ||
      pad_value_proto.has_string_value()) {
    if constexpr (std::is_same_v<T, int8_t> || std::is_same_v<T, uint8_t>) {
      auto unpacked_or = utils::UnpackDefaultValue<int32_t>(pad_value_proto);
      if (unpacked_or.ok()) {
        pad_val = static_cast<T>(*unpacked_or);
      }
    } else {
      auto unpacked_or = utils::UnpackDefaultValue<T>(pad_value_proto);
      if (unpacked_or.ok()) {
        pad_val = *unpacked_or;
      }
    }
  }

  const auto& in_dims = input.shape().dims();
  int rank = in_dims.size();

  int64_t outer_size = 1;
  for (int i = 0; i < norm_axis; ++i) {
    if (in_dims[i] > 0 &&
        outer_size > std::numeric_limits<int64_t>::max() / in_dims[i]) {
      return absl::InvalidArgumentError(
          "Integer overflow in tensor outer dimensions.");
    }
    outer_size *= in_dims[i];
  }

  int64_t inner_size = 1;
  for (int i = norm_axis + 1; i < rank; ++i) {
    if (in_dims[i] > 0 &&
        inner_size > std::numeric_limits<int64_t>::max() / in_dims[i]) {
      return absl::InvalidArgumentError(
          "Integer overflow in tensor inner dimensions.");
    }
    inner_size *= in_dims[i];
  }

  int64_t total_elements = outer_size;
  if (target_dim > 0 &&
      total_elements > std::numeric_limits<int64_t>::max() / target_dim) {
    return absl::InvalidArgumentError(
        "Integer overflow in output shape calculation.");
  }
  total_elements *= target_dim;
  if (inner_size > 0 &&
      total_elements > std::numeric_limits<int64_t>::max() / inner_size) {
    return absl::InvalidArgumentError(
        "Integer overflow in output shape calculation.");
  }
  total_elements *= inner_size;
  if (total_elements >
      std::numeric_limits<int64_t>::max() / static_cast<int64_t>(sizeof(T))) {
    return absl::InvalidArgumentError(
        "Integer overflow in output byte size calculation.");
  }

  const T* in_ptr = static_cast<const T*>(input.base());
  T* out_ptr = static_cast<T*>(output.base());

  if (inner_size == 1) {
    for (int64_t i = 0; i < outer_size; ++i) {
      const T* src = in_ptr + i * in_dim;
      T* dst = out_ptr + i * target_dim;
      std::memcpy(dst, src, in_dim * sizeof(T));
      std::fill_n(dst + in_dim, target_dim - in_dim, pad_val);
    }
  } else {
    for (int64_t i = 0; i < outer_size; ++i) {
      for (int64_t j = 0; j < in_dim; ++j) {
        const T* src = in_ptr + (i * in_dim + j) * inner_size;
        T* dst = out_ptr + (i * target_dim + j) * inner_size;
        std::memcpy(dst, src, inner_size * sizeof(T));
      }
      for (int64_t j = in_dim; j < target_dim; ++j) {
        T* dst = out_ptr + (i * target_dim + j) * inner_size;
        std::fill_n(dst, inner_size, pad_val);
      }
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<std::unique_ptr<BucketPadTransform>> BucketPadTransform::Create(
    std::vector<int64_t> target_dimensions, int32_t axis,
    utils::DefaultValue pad_value) {
  if (target_dimensions.empty()) {
    return absl::InvalidArgumentError(
        "BucketPadTransform requires at least one target dimension.");
  }
  for (int64_t dim : target_dimensions) {
    if (dim <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Target dimensions must be positive, got: ", dim));
    }
  }
  std::sort(target_dimensions.begin(), target_dimensions.end());
  return absl::WrapUnique(new BucketPadTransform(std::move(target_dimensions),
                                                 axis, std::move(pad_value)));
}

absl::StatusOr<std::unique_ptr<BucketPadTransform>>
BucketPadTransform::CreateFromProto(const BucketPadArgs& args) {
  std::vector<int64_t> target_dimensions(args.target_dimensions().begin(),
                                         args.target_dimensions().end());
  return Create(std::move(target_dimensions), args.axis(), args.pad_value());
}

absl::StatusOr<std::unique_ptr<BucketPadTransform>>
BucketPadTransform::CreateFromTransformConfig(
    const ::neptune::data::composite::TransformConfig& config,
    absl::Span<const std::shared_ptr<neptune::data::Resource>> resources) {
  BucketPadArgs args;
  if (!config.detail().UnpackTo(&args)) {
    return absl::InvalidArgumentError(
        "Failed to unpack BucketPadArgs from TransformConfig.");
  }
  return CreateFromProto(args);
}

BucketPadTransform::BucketPadTransform(std::vector<int64_t> target_dimensions,
                                       int32_t axis,
                                       utils::DefaultValue pad_value)
    : target_dimensions_(std::move(target_dimensions)),
      axis_(axis),
      pad_value_(std::move(pad_value)),
      input_types_({api::CreateTypeInfo<xla_serving::HostBuffer>()}),
      output_types_({api::CreateTypeInfo<xla_serving::HostBuffer>()}) {}

absl::StatusOr<xla_serving::HostBuffer> BucketPadTransform::operator()(
    const xla_serving::HostBuffer& input) const {
  const auto& in_dims = input.shape().dims();
  int rank = in_dims.size();
  if (rank == 0) {
    return absl::InvalidArgumentError(
        "BucketPadTransform cannot pad scalar (rank 0) tensors.");
  }

  for (int i = 0; i < rank; ++i) {
    if (in_dims[i] <= 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input dimension at axis ", i,
                       " must be positive, got: ", in_dims[i]));
    }
  }

  int norm_axis = axis_;
  if (norm_axis < 0) {
    norm_axis += rank;
  }
  if (norm_axis < 0 || norm_axis >= rank) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Axis ", axis_, " is out of bounds for tensor of rank ", rank));
  }

  int64_t in_dim = in_dims[norm_axis];
  auto it = std::lower_bound(target_dimensions_.begin(),
                             target_dimensions_.end(), in_dim);
  if (it == target_dimensions_.end()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input dimension ", in_dim, " along axis ", norm_axis,
                     " exceeds maximum target bucket dimension ",
                     target_dimensions_.back()));
  }
  int64_t target_dim = *it;

  std::vector<int64_t> out_dims(in_dims.begin(), in_dims.end());
  out_dims[norm_axis] = target_dim;

  int64_t total_elements = 1;
  for (int64_t dim : out_dims) {
    if (dim > 0 && total_elements > std::numeric_limits<int64_t>::max() / dim) {
      return absl::InvalidArgumentError(
          "Integer overflow in output shape calculation.");
    }
    total_elements *= dim;
  }

  xla::ifrt::Shape out_shape(out_dims);

  xla_serving::HostBuffer output(input.dtype(), out_shape);

  switch (input.dtype().kind()) {
    case xla::ifrt::DType::kF32:
      RETURN_IF_ERROR(ExecuteTypedPad<float>(input, output, norm_axis, in_dim,
                                             target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kF64:
      RETURN_IF_ERROR(ExecuteTypedPad<double>(input, output, norm_axis, in_dim,
                                              target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kS32:
      RETURN_IF_ERROR(ExecuteTypedPad<int32_t>(input, output, norm_axis, in_dim,
                                               target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kS64:
      RETURN_IF_ERROR(ExecuteTypedPad<int64_t>(input, output, norm_axis, in_dim,
                                               target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kU32:
      RETURN_IF_ERROR(ExecuteTypedPad<uint32_t>(
          input, output, norm_axis, in_dim, target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kU64:
      RETURN_IF_ERROR(ExecuteTypedPad<uint64_t>(
          input, output, norm_axis, in_dim, target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kS8:
      RETURN_IF_ERROR(ExecuteTypedPad<int8_t>(input, output, norm_axis, in_dim,
                                              target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kU8:
      RETURN_IF_ERROR(ExecuteTypedPad<uint8_t>(input, output, norm_axis, in_dim,
                                               target_dim, pad_value_));
      break;
    case xla::ifrt::DType::kPred:
      RETURN_IF_ERROR(ExecuteTypedPad<bool>(input, output, norm_axis, in_dim,
                                            target_dim, pad_value_));
      break;
    default:
      return absl::UnimplementedError(
          absl::StrCat("BucketPadTransform does not support dtype kind: ",
                       static_cast<int>(input.dtype().kind())));
  }

  return output;
}

absl::Status BucketPadTransform::Execute(api::ExecuteContext& context) {
  if (context.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("BucketPadTransform expects exactly 1 input, got ",
                     context.inputs.size()));
  }
  const auto& input = context.inputs[0].Get<xla_serving::HostBuffer>();
  auto output_or = (*this)(input);
  if (!output_or.ok()) {
    return output_or.status();
  }
  context.outputs.push_back(api::Value(std::move(*output_or)));
  return absl::OkStatus();
}

absl::Status BucketPadTransform::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 1) {
    return absl::InvalidArgumentError(
        absl::StrCat("BucketPadTransform expects exactly 1 input for signature "
                     "propagation, got ",
                     context.inputs.size()));
  }
  if (!context.inputs[0].IsSpec()) {
    return absl::InvalidArgumentError(
        "BucketPadTransform expected HostBufferSpec input.");
  }
  const auto& in_spec = context.inputs[0].spec();
  const auto& in_dims = in_spec.shape.dims();
  int rank = in_dims.size();
  if (rank == 0) {
    context.outputs.push_back(in_spec);
    return absl::OkStatus();
  }

  int norm_axis = axis_;
  if (norm_axis < 0) {
    norm_axis += rank;
  }
  if (norm_axis < 0 || norm_axis >= rank) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Axis ", axis_, " is out of bounds for tensor of rank ", rank));
  }

  std::vector<int64_t> out_dims(in_dims.begin(), in_dims.end());
  int64_t in_dim = in_dims[norm_axis];
  if (in_dim > 0) {
    auto it = std::lower_bound(target_dimensions_.begin(),
                               target_dimensions_.end(), in_dim);
    if (it != target_dimensions_.end()) {
      out_dims[norm_axis] = *it;
    } else {
      out_dims[norm_axis] = -1;
    }
  } else {
    out_dims[norm_axis] = -1;
  }

  api::HostBufferSpec out_spec;
  out_spec.dtype = in_spec.dtype;
  out_spec.shape = xla::ifrt::Shape(out_dims);
  context.outputs.push_back(out_spec);
  return absl::OkStatus();
}

absl::StatusOr<::neptune::data::composite::TransformConfig>
BucketPadTransform::ToProto() const {
  ::neptune::data::composite::TransformConfig config;
  config.set_op_name("BucketPadTransform");

  BucketPadArgs args;
  args.mutable_target_dimensions()->Reserve(target_dimensions_.size());
  for (int64_t dim : target_dimensions_) {
    args.add_target_dimensions(dim);
  }
  args.set_axis(axis_);
  *args.mutable_pad_value() = pad_value_;

  config.mutable_detail()->PackFrom(args);
  return config;
}

std::string BucketPadTransform::DebugName() const {
  return absl::StrFormat("BucketPadTransform(axis=%d, target_dims_count=%d)",
                         axis_, target_dimensions_.size());
}

ND_REGISTER_TRANSFORM(BucketPadTransform, "BucketPadTransform")
    .SetDeserializer(&BucketPadTransform::CreateFromTransformConfig);

}  // namespace neptune::data::composite::transforms

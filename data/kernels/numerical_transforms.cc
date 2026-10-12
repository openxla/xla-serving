#include "data/kernels/numerical_transforms.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {
namespace {

using ::xla_serving::data::Resource;
using ::xla_serving::data::composite::TransformBase;
using ::xla_serving::data::composite::TransformConfig;
using ::xla_serving::data::composite::api::HostBufferSpec;
using ::xla_serving::data::composite::api::ShapeInferenceContext;

}  // namespace

// --- MatMulTransform ---

absl::StatusOr<std::unique_ptr<MatMulTransform>> MatMulTransform::Create() {
  return std::make_unique<MatMulTransform>();
}

absl::StatusOr<xla_serving::HostBuffer> MatMulTransform::operator()(xla_serving::HostBuffer a,
                                                            xla_serving::HostBuffer b) {
  if (a.dtype() != b.dtype()) {
    return absl::InvalidArgumentError("Inputs must have the same dtype");
  }
  if (a.dtype().kind() != xla::ifrt::DType::kF32) {
    return absl::UnimplementedError("MatMul only supports F32 for now");
  }
  if (a.shape().dims().size() != 2 || b.shape().dims().size() != 2) {
    return absl::InvalidArgumentError("Inputs must be 2D matrices");
  }
  int64_t M = a.shape().dims()[0];
  int64_t K = a.shape().dims()[1];
  if (b.shape().dims()[0] != K) {
    return absl::InvalidArgumentError(
        absl::StrCat("Inner dimensions must match for MatMul. Got ", K, " and ",
                     b.shape().dims()[0]));
  }
  int64_t N = b.shape().dims()[1];

  xla_serving::HostBuffer out(a.dtype(), xla::ifrt::Shape({M, N}));

  // Use Eigen for matrix multiplication.
  auto A_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>(
      static_cast<const float*>(a.base()), M, K);
  auto B_map = Eigen::Map<const Eigen::Matrix<float, Eigen::Dynamic,
                                              Eigen::Dynamic, Eigen::RowMajor>>(
      static_cast<const float*>(b.base()), K, N);
  auto Out_map = Eigen::Map<
      Eigen::Matrix<float, Eigen::Dynamic, Eigen::Dynamic, Eigen::RowMajor>>(
      static_cast<float*>(out.base()), M, N);

  Out_map = A_map * B_map;

  return out;
}

absl::StatusOr<TransformConfig> MatMulTransform::ToProto() const {
  TransformConfig config;
  config.set_op_name("MatMulTransform");
  return config;
}

ND_REGISTER_TRANSFORM(MatMulTransform, "MatMulTransform")
    .SetDeserializer([](const TransformConfig& config,
                        absl::Span<const std::shared_ptr<Resource>> resources)
                         -> absl::StatusOr<std::unique_ptr<TransformBase>> {
      return MatMulTransform::Create();
    })
    .SetInputConstraints(
        {{.allowed_dtypes = {xla::ifrt::DType::kS32, xla::ifrt::DType::kF32},
          .expected_shape = std::vector<std::optional<int64_t>>{std::nullopt,
                                                                std::nullopt}},
         {.allowed_dtypes = {xla::ifrt::DType::kS32, xla::ifrt::DType::kF32},
          .expected_shape = std::vector<std::optional<int64_t>>{
              std::nullopt, std::nullopt}}});

absl::Status MatMulTransform::PropagateSignatureImpl(
    ShapeInferenceContext& context) const {
  if (context.inputs.size() != 2) {
    return absl::InvalidArgumentError("MatMulTransform expects 2 inputs");
  }
  if (!context.inputs[0].IsSpec() || !context.inputs[1].IsSpec()) {
    return absl::InvalidArgumentError("MatMulTransform expects tensor inputs");
  }
  const auto& a = context.inputs[0].spec();
  const auto& b = context.inputs[1].spec();
  if (a.dtype != b.dtype) {
    return absl::InvalidArgumentError("Inputs must have the same dtype");
  }
  if (a.shape.dims().size() != 2 || b.shape.dims().size() != 2) {
    return absl::InvalidArgumentError("Inputs must be 2D matrices");
  }
  int64_t M = a.shape.dims()[0];
  int64_t K = a.shape.dims()[1];
  if (b.shape.dims()[0] != K) {
    return absl::InvalidArgumentError(
        absl::StrCat("Inner dimensions must match for MatMul. Got ", K, " and ",
                     b.shape.dims()[0]));
  }
  int64_t N = b.shape.dims()[1];
  context.outputs.push_back(HostBufferSpec{a.dtype, xla::ifrt::Shape({M, N})});
  return absl::OkStatus();
}

}  // namespace xla_serving::data::kernel

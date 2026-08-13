#include "data/processor/python/examples/multi_kernel_math_transform.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace neptune::data {
namespace composite {

template <typename T>
absl::StatusOr<std::unique_ptr<MultiKernelMathTransform<T>>>
MultiKernelMathTransform<T>::Create() {
  return std::make_unique<MultiKernelMathTransform<T>>();
}

template <typename T>
absl::StatusOr<neptune::HostBuffer> MultiKernelMathTransform<T>::operator()(
    neptune::HostBuffer in_buffer) {
  neptune::HostBuffer out(in_buffer.dtype(), in_buffer.shape());
  const T* in_data = static_cast<const T*>(in_buffer.base());
  T* out_data = static_cast<T*>(out.base());
  for (size_t i = 0; i < out.shape().num_elements(); ++i) {
    out_data[i] = in_data[i] + static_cast<T>(10);
  }
  return out;
}

template <typename T>
absl::StatusOr<TransformConfig> MultiKernelMathTransform<T>::ToProto() const {
  TransformConfig config;
  config.set_op_name("MultiKernelMathTransform");
  return config;
}

template <typename T>
absl::Status MultiKernelMathTransform<T>::PropagateSignatureImpl(
    api::ShapeInferenceContext& context) const {
  if (context.inputs.size() != 1 || !context.inputs[0].IsSpec()) {
    return absl::InvalidArgumentError("Expected 1 tensor spec input");
  }
  context.outputs.push_back(context.inputs[0]);
  return absl::OkStatus();
}

template class MultiKernelMathTransform<int32_t>;
template class MultiKernelMathTransform<float>;

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<int32_t>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<neptune::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<int32_t>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kS32}}});

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<float>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<neptune::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<float>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kF32}}});

}  // namespace composite
}  // namespace neptune::data

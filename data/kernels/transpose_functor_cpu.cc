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

#define EIGEN_USE_THREADS

#include <cstdint>

#include "data/kernels/transpose_functor.h"  // from @xla_serving
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data::kernel {
namespace {

using CPUDevice = Eigen::DefaultDevice;

template <typename T>
absl::InlinedVector<T, 8> ComputeStride(const xla::ifrt::Shape& shape) {
  const int ndims = static_cast<int>(shape.dims().size());
  absl::InlinedVector<T, 8> strides(ndims);
  T stride = 1;
  for (int i = ndims - 1; i >= 0; --i) {
    strides[i] = stride;
    stride *= static_cast<T>(shape.dims()[i]);
  }
  return strides;
}

template <typename T, bool conjugate>
void TransposeSimple(const CPUDevice& /*device*/, const xla_serving::HostBuffer& in,
                     absl::Span<const int32_t> perm, xla_serving::HostBuffer* out) {
  const int ndims = static_cast<int>(in.shape().dims().size());
  absl::InlinedVector<int64_t, 8UL> in_strides =
      ComputeStride<int64_t>(in.shape());
  absl::InlinedVector<int64_t, 8UL> out_strides =
      ComputeStride<int64_t>(out->shape());
  const T* p = reinterpret_cast<const T*>(in.base());
  T* q = reinterpret_cast<T*>(out->base());
  const int64_t num_elements = in.shape().num_elements();
  for (int64_t o_idx = 0; o_idx < num_elements; ++o_idx) {
    int64_t i_idx = 0;
    int64_t t = o_idx;
    for (int i = 0; i < ndims; ++i) {
      const int64_t ratio = t / out_strides[i];
      t -= ratio * out_strides[i];
      i_idx += ratio * in_strides[perm[i]];
    }
    if (conjugate) {
      q[o_idx] = Eigen::numext::conj(p[i_idx]);
    } else {
      q[o_idx] = p[i_idx];
    }
  }
}

}  // namespace

template <typename T, bool conjugate>
struct Transpose<CPUDevice, T, conjugate> {
  static void run(const CPUDevice& d, const xla_serving::HostBuffer& in,
                  absl::Span<const int32_t> perm, xla_serving::HostBuffer* out) {
    switch (in.shape().dims().size()) {
      case 2:
        internal::TransposeUsingEigen<CPUDevice, T, 2>(d, in, perm, conjugate,
                                                       out);
        break;
      case 3:
        internal::TransposeUsingEigen<CPUDevice, T, 3>(d, in, perm, conjugate,
                                                       out);
        break;
      case 4:
        internal::TransposeUsingEigen<CPUDevice, T, 4>(d, in, perm, conjugate,
                                                       out);
        break;
      case 5:
        internal::TransposeUsingEigen<CPUDevice, T, 5>(d, in, perm, conjugate,
                                                       out);
        break;
      case 6:
        internal::TransposeUsingEigen<CPUDevice, T, 6>(d, in, perm, conjugate,
                                                       out);
        break;
      case 7:
        internal::TransposeUsingEigen<CPUDevice, T, 7>(d, in, perm, conjugate,
                                                       out);
        break;
      case 8:
        internal::TransposeUsingEigen<CPUDevice, T, 8>(d, in, perm, conjugate,
                                                       out);
        break;
      default:
        TransposeSimple<T, conjugate>(d, in, perm, out);
        break;
    }
  }
};

#define INSTANTIATE(DEVICE)                                                    \
  template <>                                                                  \
  absl::Status DoTranspose(const DEVICE& device, const xla_serving::HostBuffer& in,    \
                           absl::Span<const int32_t> perm,                     \
                           xla_serving::HostBuffer* out) {                             \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/false,    \
                                     out);                                     \
  }                                                                            \
  template <>                                                                  \
  absl::Status DoConjugateTranspose(                                           \
      const DEVICE& device, const xla_serving::HostBuffer& in,                         \
      absl::Span<const int32_t> perm, xla_serving::HostBuffer* out) {                  \
    return internal::DoTransposeImpl(device, in, perm, /*conjugate=*/true,     \
                                     out);                                     \
  }                                                                            \
  template <>                                                                  \
  absl::Status DoMatrixTranspose(                                              \
      const DEVICE& device, const xla_serving::HostBuffer& in, xla_serving::HostBuffer* out) { \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/false,    \
                                           out);                               \
  }                                                                            \
  template <>                                                                  \
  absl::Status DoConjugateMatrixTranspose(                                     \
      const DEVICE& device, const xla_serving::HostBuffer& in, xla_serving::HostBuffer* out) { \
    return internal::DoMatrixTransposeImpl(device, in, /*conjugate=*/true,     \
                                           out);                               \
  }

INSTANTIATE(CPUDevice)

}  // namespace xla_serving::data::kernel

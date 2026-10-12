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

#ifndef XLA_SERVING_DATA_KERNELS_TRANSPOSE_FUNCTOR_H_
#define XLA_SERVING_DATA_KERNELS_TRANSPOSE_FUNCTOR_H_

#include <complex>
#include <cstdint>
#include <numeric>
#include <utility>
#include <vector>

#include "data/kernels/util/ttypes.h"  // from @xla_serving
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving::data::kernel {

// Transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.shape().dims().size() == out->shape().dims().size()
// REQUIRES: in.shape().dims().size() == perm.size()
// REQUIRES: in.shape().dims()[perm[i]] == out->shape().dims()[i]
template <typename Device>
absl::Status DoTranspose(const Device& device, const xla_serving::HostBuffer& in,
                         absl::Span<const int32_t> perm, xla_serving::HostBuffer* out);

// Conjugate and transpose tensor 'in' into tensor 'out' according to dimension
// permutation 'perm'.
//
// REQUIRES: in.dtype() == out->dtype()
// REQUIRES: in.shape().dims().size() == out->shape().dims().size()
// REQUIRES: in.shape().dims().size() == perm.size()
// REQUIRES: in.shape().dims()[perm[i]] == out->shape().dims()[i]
template <typename Device>
absl::Status DoConjugateTranspose(const Device& device,
                                  const xla_serving::HostBuffer& in,
                                  absl::Span<const int32_t> perm,
                                  xla_serving::HostBuffer* out);

// Convenience versions of DoTranspose that only swap the last (inner) two
// dimensions.
template <typename Device>
absl::Status DoMatrixTranspose(const Device& device, const xla_serving::HostBuffer& in,
                               xla_serving::HostBuffer* out);

// Convenience versions of DoConjugateTranspose that only swap the last (inner)
// two dimensions.
template <typename Device>
absl::Status DoConjugateMatrixTranspose(const Device& device,
                                        const xla_serving::HostBuffer& in,
                                        xla_serving::HostBuffer* out);

// Primary device specific functor to be specialized for each device and type.
template <typename Device, typename T, bool conjugate = false>
struct Transpose {
  static void run(const Device& d, const xla_serving::HostBuffer& in,
                  absl::Span<const int32_t> perm, xla_serving::HostBuffer* out);
};

// Implementation details.
namespace internal {

using TransposeDimsVec = absl::InlinedVector<int64_t, 8UL>;
using TransposePermsVec = absl::InlinedVector<int32_t, 8UL>;

// Helper function that takes a tensor shape, a permutation, combines the
// neighboring shapes if their indices in the permutation are consecutive.
// The function outputs the combined shape and new permutation.
// Example: Tensor shape {2, 3, 4, 5, 120} and permutation {0, 4, 1, 2, 3} will
// produce new shape {2, 60, 120} and new permutation {0, 2, 1}.
inline void ReduceTransposeDimensions(const xla::ifrt::Shape& shape,
                                      absl::Span<const int32_t> perm,
                                      TransposePermsVec* new_perm,
                                      TransposeDimsVec* new_dims) {
  const int ndims = static_cast<int>(shape.dims().size());
  CHECK_EQ(ndims, perm.size());
  if (ndims == 1) {
    // If input dimension is already 1, no need to reduce dimension.
    new_perm->resize(1);
    (*new_perm)[0] = perm[0];
    (*new_dims)[0] = shape.dims()[0];
    return;
  }
  TransposePermsVec new_dim_position(ndims, -1);
  TransposeDimsVec combined_dims(ndims, 0);
  int cur_head = perm[0];
  new_dim_position[cur_head] = 0;
  combined_dims[0] = shape.dims()[cur_head];
  int dim_idx = 0;
  for (int perm_idx = 1; perm_idx < ndims; ++perm_idx) {
    // If two indices in permutation are consecutive numbers, combine their
    // dimensions.
    if (cur_head + 1 == perm[perm_idx]) {
      cur_head = perm[perm_idx];
      combined_dims[dim_idx] *= shape.dims()[cur_head];
    } else {
      // Else start a new dimension.
      cur_head = perm[perm_idx];
      dim_idx++;
      new_dim_position[cur_head] = dim_idx;
      combined_dims[dim_idx] = shape.dims()[cur_head];
    }
  }
  // Compact the new permutations and dimension sizes.
  new_perm->resize(dim_idx + 1);
  new_dims->resize(dim_idx + 1);
  dim_idx = 0;
  for (int i = 0; i < new_dim_position.size(); ++i) {
    if (new_dim_position[i] >= 0) {
      int new_perm_idx = new_dim_position[i];
      (*new_perm)[dim_idx] = new_perm_idx;
      (*new_dims)[dim_idx] = combined_dims[new_perm_idx];
      dim_idx++;
    }
  }
}

// If all non-singleton dimensions remain in ascending order, the shuffled
// singletons can be transposed by a reshape, saving a memory allocation & copy.
// |permutation| must be a permutation of {0, .., input_shape.dims() - 1}.
// That is, for all i, 0 <= perm[i] < input_shape.dims().
// In practice, this is checked in TransposeOp::Compute prior to calling this
// function, and the function sits here to facilitate unit testing.
inline bool NonSingletonDimensionsAlign(
    const xla::ifrt::Shape& input_shape,
    const std::vector<int32_t>& permutation) {
  int last_nonsingleton_perm_dim = -1;
  for (int perm_dim : permutation) {
    if (input_shape.dims()[perm_dim] == 1) {
      continue;
    }
    if (perm_dim < last_nonsingleton_perm_dim) {
      return false;
    }
    last_nonsingleton_perm_dim = perm_dim;
  }
  return true;
}

// Uses Eigen to transpose.
template <typename Device, typename T, int NDIMS>
void TransposeUsingEigen(const Device& d, const xla_serving::HostBuffer& in,
                         absl::Span<const int32_t> perm, bool conjugate,
                         xla_serving::HostBuffer* out) {
  Eigen::array<int, NDIMS> p;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> in_dsizes;
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> out_dsizes;
  for (int i = 0; i < NDIMS; ++i) {
    p[i] = perm[i];
    in_dsizes[i] = in.shape().dims()[i];
    out_dsizes[i] = out->shape().dims()[i];
  }
  auto x = typename TTypes<T, NDIMS>::ConstTensor(
      reinterpret_cast<const T*>(in.base()), in_dsizes);
  auto y = typename TTypes<T, NDIMS>::Tensor(reinterpret_cast<T*>(out->base()),
                                             out_dsizes);
  if (conjugate) {
    y.device(d) = x.conjugate().shuffle(p);
  } else {
    y.device(d) = x.shuffle(p);
  }
}

template <typename Device>
absl::Status DoTransposeImpl(const Device& d, const xla_serving::HostBuffer& in,
                             absl::Span<const int32_t> perm, bool conjugate,
                             xla_serving::HostBuffer* out) {
  CHECK_EQ(in.shape().dims().size(), out->shape().dims().size());
  CHECK_EQ(in.shape().dims().size(), perm.size());
  CHECK_EQ(in.dtype(), out->dtype());
  if (in.shape().num_elements() == 0) {
    return absl::OkStatus();
  }
  switch (in.dtype().kind()) {
    case xla::ifrt::DType::kPred:
    case xla::ifrt::DType::kS8:
    case xla::ifrt::DType::kU8:
      Transpose<Device, uint8_t>::run(d, in, perm, out);
      break;

    case xla::ifrt::DType::kBF16:
    case xla::ifrt::DType::kF16:
    case xla::ifrt::DType::kS16:
    case xla::ifrt::DType::kU16:
      Transpose<Device, uint16_t>::run(d, in, perm, out);
      break;

    case xla::ifrt::DType::kF32:
    case xla::ifrt::DType::kS32:
    case xla::ifrt::DType::kU32:
      Transpose<Device, uint32_t>::run(d, in, perm, out);
      break;

    case xla::ifrt::DType::kF64:
    case xla::ifrt::DType::kS64:
    case xla::ifrt::DType::kU64:
      Transpose<Device, uint64_t>::run(d, in, perm, out);
      break;

    case xla::ifrt::DType::kC64:
      if (conjugate) {
        Transpose<Device, std::complex<float>, /*conjugate=*/true>::run(
            d, in, perm, out);
      } else {
        Transpose<Device, uint64_t>::run(d, in, perm, out);
      }
      break;

    case xla::ifrt::DType::kC128:
      if (conjugate) {
        Transpose<Device, std::complex<double>, /*conjugate=*/true>::run(
            d, in, perm, out);
      } else {
        Transpose<Device, std::complex<double>, /*conjugate=*/false>::run(
            d, in, perm, out);
      }
      break;

    case xla::ifrt::DType::kString:
      Transpose<Device, tsl::tstring>::run(d, in, perm, out);
      break;

    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported dtype on CPU: ", in.dtype()));
  }
  return absl::OkStatus();
}

template <typename Device>
inline absl::Status DoMatrixTransposeImpl(const Device& device,
                                          const xla_serving::HostBuffer& in,
                                          bool conjugate,
                                          xla_serving::HostBuffer* out) {
  const int ndims = static_cast<int>(in.shape().dims().size());
  if (ndims == 0) return absl::OkStatus();
  TransposePermsVec perm(ndims);
  std::iota(perm.begin(), perm.end(), 0);
  std::swap(perm[ndims - 2], perm[ndims - 1]);
  return DoTransposeImpl(device, in, perm, conjugate, out);
}

}  // namespace internal
}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_TRANSPOSE_FUNCTOR_H_

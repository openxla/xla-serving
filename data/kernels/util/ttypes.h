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

#ifndef XLA_SERVING_DATA_KERNELS_UTIL_TTYPES_H_
#define XLA_SERVING_DATA_KERNELS_UTIL_TTYPES_H_

#include <cstddef>
#include <cstdint>
#include <limits>
#include <utility>

#include "absl/log/check.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving::data::kernel {

// Helper to define Eigen Tensor types given that the scalar is of type T.
// Extends ::xla_serving::TTypes to add some missing aliases and
// supporting custom IndexType (e.g. int32_t for GPU kernels).
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes : public ::xla_serving::TTypes<T, NDIMS, IndexType> {
  // Rank-0 scalar aliases.
  using Scalar = ::xla_serving::TTypes<T, 0, IndexType>::Tensor;
  using ConstScalar = ::xla_serving::TTypes<T, 0, IndexType>::ConstTensor;

  // Rank-1 vector aliases (synonyms for Flat / ConstFlat).
  using Vec = ::xla_serving::TTypes<T, 1, IndexType>::Flat;
  using ConstVec = ::xla_serving::TTypes<T, 1, IndexType>::ConstFlat;

  // 32-bit tensor aliases for GPU kernels (mirrors TensorFlow TTypes).
  using Tensor32Bit = ::xla_serving::TTypes<T, NDIMS, int32_t>::Tensor;
  using ConstTensor32Bit = ::xla_serving::TTypes<T, NDIMS, int32_t>::ConstTensor;
  using Flat32Bit = ::xla_serving::TTypes<T, 1, int32_t>::Tensor;
  using ConstFlat32Bit = ::xla_serving::TTypes<T, 1, int32_t>::ConstFlat;
};

// 32-bit index type companion for Tensor32Bit / Flat32Bit.
using Index32 = TTypes<float, 1>::Tensor32Bit::Index;

template <typename Index, int NumDims>
bool SafeFor32BitIndexing(const Eigen::DSizes<Index, NumDims>& in) {
  for (int i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename Index, size_t NumDims>
bool SafeFor32BitIndexing(const Eigen::array<Index, NumDims>& in) {
  for (size_t i = 0; i < NumDims; ++i) {
    if (in[i] > std::numeric_limits<Index32>::max()) return false;
  }
  return true;
}

template <typename TensorType, typename Enable = typename TTypes<
                                   typename TensorType::Scalar,
                                   TensorType::NumIndices, Index32>::Tensor>
bool SafeFor32BitIndexing(TensorType in) {
  return in.size() <= std::numeric_limits<Index32>::max();
}

template <typename Index, int NumDims>
Eigen::DSizes<Index32, NumDims> To32Bit(
    const Eigen::DSizes<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::DSizes<Index32, NumDims> out;
  for (int i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename Index, size_t NumDims>
Eigen::array<Index32, NumDims> To32Bit(const Eigen::array<Index, NumDims>& in) {
  DCHECK(SafeFor32BitIndexing(in));
  Eigen::array<Index32, NumDims> out;
  for (size_t i = 0; i < NumDims; ++i) {
    out[i] = static_cast<Index32>(in[i]);
  }
  return out;
}

template <typename TensorType>
typename TTypes<typename TensorType::Scalar, TensorType::NumIndices,
                Index32>::Tensor
To32Bit(TensorType in) {
  using RetType = typename TTypes<typename TensorType::Scalar,
                                  TensorType::NumIndices, Index32>::Tensor;
  DCHECK(SafeFor32BitIndexing(in));
  return RetType(in.data(), To32Bit(in.dimensions()));
}

namespace internal {

template <typename Device>
struct MaybeWith32BitIndexingImpl {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    func(std::forward<Args>(args)...);
  }
};

template <>
struct MaybeWith32BitIndexingImpl<Eigen::GpuDevice> {
  template <typename Func, typename... Args>
  void operator()(Func func, Args&&... args) const {
    auto all = [](const auto&... bool_vals) {
      for (bool b : {bool_vals...}) {
        if (!b) return false;
      }
      return true;
    };
    if (all(SafeFor32BitIndexing(std::forward<Args>(args))...)) {
      func(To32Bit(std::forward<Args>(args))...);
    } else {
      func(std::forward<Args>(args)...);
    }
  }
};

}  // namespace internal

template <typename Device, typename Func, typename... Args>
void MaybeWith32BitIndexing(Func func, Args&&... args) {
  return internal::MaybeWith32BitIndexingImpl<Device>()(
      func, std::forward<Args>(args)...);
}

}  // namespace xla_serving::data::kernel

#endif  // XLA_SERVING_DATA_KERNELS_UTIL_TTYPES_H_

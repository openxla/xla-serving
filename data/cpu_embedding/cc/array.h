// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_ARRAY_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_ARRAY_H_

#include <any>
#include <cstdint>
#include <cstring>
#include <functional>
#include <initializer_list>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"

namespace neptune::data::cpu_embedding {

enum class DType {
  kUnknown = 0,
  kFloat32 = 1,
  kBfloat16 = 2,
  kInt32 = 3,
  kInt64 = 4,
};

inline int64_t ByteWidth(DType dtype) {
  switch (dtype) {
    case DType::kFloat32:
    case DType::kInt32:
      return 4;
    case DType::kBfloat16:
      return 2;
    case DType::kInt64:
      return 8;
    default:
      return 0;
  }
}

struct Shape {
  absl::InlinedVector<int64_t, 4> dims;

  Shape() = default;
  explicit Shape(absl::InlinedVector<int64_t, 4> d) : dims(std::move(d)) {}
  explicit Shape(std::vector<int64_t> d) : dims(d.begin(), d.end()) {}
  Shape(std::initializer_list<int64_t> d) : dims(d.begin(), d.end()) {}

  int64_t NumElements() const {
    if (dims.empty()) return 0;
    return std::accumulate(dims.begin(), dims.end(), 1LL,
                           std::multiplies<int64_t>());
  }
};

template <typename T>
struct NativeToDType;

template <>
struct NativeToDType<float> {
  static constexpr DType kDType = DType::kFloat32;
};

template <>
struct NativeToDType<Eigen::bfloat16> {
  static constexpr DType kDType = DType::kBfloat16;
};

template <>
struct NativeToDType<int32_t> {
  static constexpr DType kDType = DType::kInt32;
};

template <>
struct NativeToDType<int64_t> {
  static constexpr DType kDType = DType::kInt64;
};

class Array {
 public:
  template <typename T, int Rank>
  using TensorMap = Eigen::TensorMap<Eigen::Tensor<T, Rank, Eigen::RowMajor>>;

  Array() : dtype_(DType::kUnknown) {}

  // Create an array by copying data from a span.
  // Use std::shared_ptr<std::string> as owner to make ReleaseBuffer move-only.
  template <typename T>
  Array(absl::Span<const T> data, Shape shape)
      : dtype_(NativeToDType<T>::kDType), shape_(std::move(shape)) {
    size_t num_elements = shape_.NumElements();
    size_t byte_size = num_elements * sizeof(T);
    auto storage = std::make_shared<std::string>(
        reinterpret_cast<const char*>(data.data()), byte_size);
    buffer_ = absl::MakeSpan(storage->data(), storage->size());
    owner_ = std::move(storage);
  }

  // Create an array and initialize it with a constant value.
  template <typename T>
  Array(T val, Shape shape)
      : dtype_(NativeToDType<T>::kDType), shape_(std::move(shape)) {
    size_t num_elements = shape_.NumElements();
    auto storage =
        std::make_shared<std::string>(num_elements * sizeof(T), '\0');
    T* ptr = reinterpret_cast<T*>(storage->data());
    std::fill(ptr, ptr + num_elements, val);
    buffer_ = absl::MakeSpan(storage->data(), storage->size());
    owner_ = std::move(storage);
  }

  // Create from an owned buffer.
  Array(std::string buffer, Shape shape, DType dtype)
      : dtype_(dtype), shape_(std::move(shape)) {
    auto storage = std::make_shared<std::string>(std::move(buffer));
    buffer_ = absl::MakeSpan(storage->data(), storage->size());
    owner_ = std::move(storage);
  }

  // Wraps an existing buffer without taking ownership.
  template <typename T>
  static Array Wrap(T* data, Shape shape, std::any owner = {}) {
    Array arr;
    arr.dtype_ = NativeToDType<T>::kDType;
    arr.shape_ = std::move(shape);
    size_t byte_size = arr.shape_.NumElements() * sizeof(T);
    arr.buffer_ = absl::MakeSpan(reinterpret_cast<char*>(data), byte_size);
    arr.owner_ = std::move(owner);
    return arr;
  }

  // Wraps an existing buffer with a specified DType without taking ownership.
  static Array Wrap(void* data, Shape shape, DType dtype, std::any owner = {}) {
    Array arr;
    arr.dtype_ = dtype;
    arr.shape_ = std::move(shape);
    size_t byte_size = arr.shape_.NumElements() * ByteWidth(dtype);
    arr.buffer_ = absl::MakeSpan(static_cast<char*>(data), byte_size);
    arr.owner_ = std::move(owner);
    return arr;
  }

  DType GetDType() const { return dtype_; }
  const Shape& GetShape() const { return shape_; }

  template <typename T, int Rank>
  TensorMap<T, Rank> AsTensor() const {
    Eigen::array<Eigen::Index, Rank> eigen_dims;
    for (int i = 0; i < Rank; ++i) {
      eigen_dims[i] = shape_.dims[i];
    }
    return TensorMap<T, Rank>(static_cast<T*>(Data()), eigen_dims);
  }

  absl::Span<const int64_t> ShapeSpan() const {
    return absl::MakeConstSpan(shape_.dims);
  }

  template <typename T>
  absl::Span<T> AsSpan() {
    return absl::MakeSpan(static_cast<T*>(Data()), shape_.NumElements());
  }

  template <typename T>
  absl::Span<const T> AsSpan() const {
    return absl::MakeConstSpan(static_cast<const T*>(Data()),
                               shape_.NumElements());
  }

  void* Data() const { return buffer_.data(); }

  std::string ReleaseBuffer() && {
    if (owner_.type() == typeid(std::shared_ptr<std::string>)) {
      auto storage = std::any_cast<std::shared_ptr<std::string>>(owner_);
      if (storage.use_count() == 1) {
        std::string res = std::move(*storage);
        owner_ = std::any();
        buffer_ = {};
        return res;
      }
    }
    std::string res(buffer_.begin(), buffer_.end());
    owner_ = std::any();
    buffer_ = {};
    return res;
  }

 private:
  DType dtype_;
  Shape shape_;
  std::any owner_;
  absl::Span<char> buffer_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_ARRAY_H_

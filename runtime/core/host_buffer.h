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

#ifndef XLA_SERVING_RUNTIME_HOST_BUFFER_H_
#define XLA_SERVING_RUNTIME_HOST_BUFFER_H_

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/Tensor"
#include "protos/grpc/host_buffer.pb.h"  // from @xla_serving
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "xla/overflow_util.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/tsl/platform/types.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

// IWYU pragma: no_include "Eigen/src/Core/util/ConfigureVectorization.h"

namespace xla_serving {

// Converts xla::ifrt::Shape to Eigen::DSizes.
template <int NDIMS, typename IndexType = Eigen::DenseIndex>
Eigen::DSizes<IndexType, NDIMS> AsEigenDSizesCopy(
    const xla::ifrt::Shape& shape) {
  CHECK(NDIMS == shape.dims().size());
  Eigen::DSizes<IndexType, NDIMS> dsizes;
  for (int d = 0; d < NDIMS; ++d) {
    dsizes[d] = static_cast<IndexType>(shape.dims()[d]);
  }
  return dsizes;
}

// Helper to define eigen Tensor types given that the scalar is of type T.
template <typename T, int NDIMS = 1, typename IndexType = Eigen::DenseIndex>
struct TTypes {
  // Rank-<NDIMS> tensor of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, NDIMS, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Tensor;

  // Rank-1 tensor (vector) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Flat;

  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, NDIMS, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstTensor;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 1, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstFlat;
  typedef Eigen::TensorMap<Eigen::Tensor<T, 1, Eigen::RowMajor, IndexType>>
      UnalignedFlat;

  // Rank-2 tensor (matrix) of scalar type T.
  typedef Eigen::TensorMap<Eigen::Tensor<T, 2, Eigen::RowMajor, IndexType>,
                           Eigen::Aligned>
      Matrix;
  typedef Eigen::TensorMap<
      Eigen::Tensor<const T, 2, Eigen::RowMajor, IndexType>, Eigen::Aligned>
      ConstMatrix;
};

class BufferData {
 public:
  BufferData(void* base, xla::ifrt::DType dtype, xla::ifrt::Shape shape,
             std::optional<absl::InlinedVector<int64_t, 4>> byte_strides,
             std::shared_ptr<void> owner = nullptr)
      : base_(base),
        dtype_(dtype),
        shape_(std::move(shape)),
        byte_strides_(std::move(byte_strides)),
        owner_(std::move(owner)) {}

  void* base() const { return base_; }
  xla::ifrt::DType dtype() const { return dtype_; }
  const xla::ifrt::Shape& shape() const { return shape_; }
  const std::optional<absl::InlinedVector<int64_t, 4>>& byte_strides() const {
    return byte_strides_;
  }
  std::shared_ptr<void> owner() const { return owner_; }
  int ref_count() const { return owner_.use_count(); }

  std::string DebugString() const;

 private:
  void* base_ = nullptr;
  xla::ifrt::DType dtype_;
  xla::ifrt::Shape shape_;
  std::optional<absl::InlinedVector<int64_t, 4>> byte_strides_;
  std::shared_ptr<void> owner_ = nullptr;
};

// HostBuffer represents a shaped buffer in host memory for numeric or string
// data types. It uses BufferData for underlying storage and memory ownership,
// and provides convenient APIs for slicing, concatenation, and type-safe data
// access via Eigen::Tensor wrappers. HostBuffer instances are lightweight views
// that support efficient slicing by sharing ownership of the underlying data
// buffer without copying. HostBuffer does not support variant type or dynamic
// shape. String data is stored using tsl::tstring.
class HostBuffer {
 public:
  HostBuffer() = default;
  explicit HostBuffer(void* base, xla::ifrt::DType dtype,
                      xla::ifrt::Shape shape,
                      std::shared_ptr<void> owner = nullptr);
  explicit HostBuffer(const xla::ifrt::DType& dtype,
                      const xla::ifrt::Shape& shape,
                      Allocator* allocator = DefaultAllocator::GetInstance());
  HostBuffer(void* base, xla::ifrt::DType dtype, xla::ifrt::Shape shape,
             const std::optional<absl::InlinedVector<int64_t, 4>>& byte_strides,
             std::shared_ptr<void> owner);

  // Allocates a new HostBuffer, returning an error if `dtype` or `shape` is
  // invalid, overflows, exceeds `--xla_serving_host_buffer_max_elements`, or if
  // `allocator` fails to allocate memory.
  static absl::StatusOr<HostBuffer> Create(
      xla::ifrt::DType dtype, xla::ifrt::Shape shape,
      Allocator* allocator = DefaultAllocator::GetInstance());

  /// Returns a `string_view` mapping the current host buffer.
  ///
  /// The returned `string_view` may point to memory location on devices
  /// that the CPU cannot address directly.
  ///
  /// NOTE: The underlying buffer is refcounted, so the lifetime
  /// of the contents mapped by the `string_view` matches the lifetime of
  /// the buffer; callers should arrange to make sure the buffer does
  /// not get destroyed while the `string_view` is still used.
  ///
  /// REQUIRES: `DataTypeCanUseMemcpy(dtype())`.
  absl::string_view raw_data() const;

  /// Slice this HostBuffer along the 1st dimension using the given allocator.

  /// I.e., the returned HostBuffer covers the range [start, end).
  /// The returned HostBuffer shares the underlying buffer with this
  /// HostBuffer.
  ///
  /// NOTE: The returned HostBuffer may not satisfy the same alignment
  /// requirement as this HostBuffer depending on the shape. The caller
  /// must check the returned HostBuffer's alignment before calling certain
  /// methods that have alignment requirement (e.g., `flat()`, `tensor()`).
  ///
  /// NOTE: When fed with an N-dimensional HostBuffer, this method returns a
  /// HostBuffer also with N dimensions. If you want to select a sub tensor, see
  /// SubSlice.
  absl::StatusOr<HostBuffer> SliceAlongFirstDimension(
      int64_t start, int64_t end,
      Allocator* allocator = DefaultAllocator::GetInstance()) const;

  // Returns a new HostBuffer reshaped to the shape specified. Returns an error
  // if the total number of elements mismatch.
  absl::StatusOr<HostBuffer> Reshape(const xla::ifrt::Shape& shape) const;

  // Returns a new HostBuffer reshaped to the shape specified, reinterpreted
  // with the dtype specified. Returns an error if the total number of bytes
  // mismatch.
  absl::StatusOr<HostBuffer> Bitcast(xla::ifrt::DType dtype,
                                     const xla::ifrt::Shape& shape) const;

  /// Splits the host buffer into 'sizes.size()' individual buffers, along the
  /// 0th dimension using the given allocator. The ith output buffer has
  /// 0th-dimension size 'sizes[i]'.
  ///
  /// REQUIRES: host buffer must have at least one dimension.
  /// REQUIRES: 0th-dimension size must equal the sum of the elements of
  /// 'sizes'.
  /// REQUIRES: host buffer must point to data stored in CPU memory.
  /// REQUIRES: host buffer must be a memory-aligned type.
  absl::StatusOr<std::vector<HostBuffer>> SplitAlongFirstDimension(
      absl::Span<const int64_t> sizes,
      Allocator* allocator = DefaultAllocator::GetInstance()) const;

  const xla::ifrt::Shape& shape() const {
    CHECK(buffer_data_.has_value());
    return buffer_data_->shape();
  }

  xla::ifrt::DType dtype() const { return dtype_; }

  const std::optional<absl::InlinedVector<int64_t, 4>>& byte_strides() const {
    CHECK(buffer_data_.has_value());
    return buffer_data_->byte_strides();
  }

  xla::ifrt::DType storage_dtype() const {
    CHECK(buffer_data_.has_value());
    return buffer_data_->dtype();
  }

  void* base() const {
    CHECK(buffer_data_.has_value());
    return const_cast<void*>(buffer_data_->base());
  }

  // Returns the total number of bytes in the storage buffer.
  int64_t total_storage_bytes() const;

  int ref_count() const {
    CHECK(buffer_data_.has_value());
    return buffer_data_->ref_count();
  }

  std::shared_ptr<void> owner() const {
    if (buffer_data_.has_value()) {
      return buffer_data_->owner();
    }
    return nullptr;
  }

  // Whether this buffer has storage. A default-constructed `HostBuffer` has
  // none, and most other accessors must not be called on it.
  bool has_data() const { return buffer_data_.has_value(); }

  std::string DebugString() const {
    if (buffer_data_) {
      return buffer_data_->DebugString();
    }
    return "Empty host buffer";
  }

  static absl::StatusOr<HostBuffer> FromProto(const HostBufferProto& proto);
  static absl::StatusOr<HostBuffer> FromProto(const HostBufferProto& proto,
                                              Allocator* allocator);

  // Serializes a HostBuffer to a HostBufferProto, copying the underlying data.
  static HostBufferProto ToHostBufferProto(const HostBuffer& host_buffer);

  template <int NDIMS>
  Eigen::DSizes<Eigen::DenseIndex, NDIMS> ShapeAsEigenDSizesCopy() const {
    return AsEigenDSizesCopy<NDIMS, Eigen::DenseIndex>(shape());
  }

  bool IsAligned(size_t alignment) const {
    if (dtype_.kind() == xla::ifrt::DType::Kind::kString) return true;
    if (shape().num_elements() == 0) return true;
    return reinterpret_cast<uintptr_t>(base<uint8_t>()) % alignment == 0;
  }

  template <typename T, size_t NDIMS>
  typename TTypes<T, NDIMS>::Tensor tensor() const {
    // Check underlying buffer data meet alignment requirement for eigen tensor.
    CHECK(IsAligned(EIGEN_MAX_ALIGN_BYTES));
    return typename TTypes<T, NDIMS>::Tensor(
        base<T>(), AsEigenDSizesCopy<NDIMS, Eigen::DenseIndex>(shape()));
  }

  template <typename T, size_t NDIMS>
  absl::StatusOr<typename TTypes<T, NDIMS>::Tensor> shaped(
      absl::Span<const int64_t> new_shape) {
    // Check underlying buffer data meet alignment requirement for eigen tensor.
    CHECK(IsAligned(EIGEN_MAX_ALIGN_BYTES));
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    TF_RETURN_IF_ERROR(FillDimsAndValidateCompatibleShape(new_shape, &dims));
    return typename TTypes<T, NDIMS>::Tensor(base<T>(), dims);
  }

  template <typename T, size_t NDIMS>
  absl::StatusOr<typename TTypes<T, NDIMS>::ConstTensor> shaped(
      absl::Span<const int64_t> new_shape) const {
    // Check underlying buffer data meet alignment requirement for eigen tensor.
    CHECK(IsAligned(EIGEN_MAX_ALIGN_BYTES));
    Eigen::array<Eigen::DenseIndex, NDIMS> dims;
    TF_RETURN_IF_ERROR(FillDimsAndValidateCompatibleShape(new_shape, &dims));
    return typename TTypes<T, NDIMS>::ConstTensor(base<T>(), dims);
  }

  template <typename T>
  typename TTypes<T>::Flat flat() {
    // Check underlying buffer data meet alignment requirement for eigen tensor.
    CHECK(IsAligned(EIGEN_MAX_ALIGN_BYTES));
    Eigen::array<Eigen::DenseIndex, 1> dims;
    dims[0] = shape().num_elements();
    return typename TTypes<T, 1>::Tensor(base<T>(), dims);
  }

  template <typename T>
  typename TTypes<T>::ConstFlat flat() const {
    CHECK(IsAligned(EIGEN_MAX_ALIGN_BYTES));
    Eigen::array<Eigen::DenseIndex, 1> dims;
    dims[0] = shape().num_elements();
    return typename TTypes<T, 1>::ConstTensor(base<T>(), dims);
  }

  template <typename T>
  typename TTypes<T>::UnalignedFlat unaligned_flat() const {
    Eigen::array<Eigen::DenseIndex, 1> dims;
    dims[0] = shape().num_elements();
    return typename TTypes<T, 1>::UnalignedFlat(base<T>(), dims);
  }

 private:
  friend class HostBufferSplitEasyCasesTest;

  bool IsInnerDimsSizeAligned() const;

  absl::Status SplitEasyCases(Allocator* allocator,
                              absl::Span<const int64_t> sizes,
                              std::vector<HostBuffer>& output,
                              bool& easy_cases_done) const;

  absl::Status SplitAlongFirstDimensionInternal(std::vector<HostBuffer>* output,
                                                absl::Span<const int64_t> sizes,
                                                Allocator* allocator) const;

  absl::StatusOr<HostBuffer> SliceInDim(int dim, int64_t start,
                                        int64_t end) const;

  template <size_t NDIMS>
  absl::Status FillDimsAndValidateCompatibleShape(
      absl::Span<const int64_t> new_shape,
      Eigen::array<Eigen::DenseIndex, NDIMS>* dims) const {
    if (NDIMS != new_shape.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("NDIMS does not match shape.size(): ", NDIMS, " vs. ",
                       new_shape.size()));
    }
    int64_t new_num_elements = 1;
    for (size_t d = 0; d < NDIMS; d++) {
      if (new_shape[d] < 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Invalid negative dimension in shape: ", new_shape[d]));
      }
      auto [next_num_elements, overflow] =
          xla::OverflowSafeMultiply(new_num_elements, new_shape[d]);
      if (overflow) {
        return absl::InvalidArgumentError("Shape overflow in shaped()");
      }
      new_num_elements = next_num_elements;
      (*dims)[d] = new_shape[d];
    }
    if (new_num_elements != shape().num_elements()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "New num elements does not match original num elements: ",
          new_num_elements, " vs. ", shape().num_elements()));
    }
    return absl::OkStatus();
  }
  // Dtype of the host buffer can be different from the storage dtype of the
  // underlying buffer data.
  xla::ifrt::DType dtype_ = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  std::optional<BufferData> buffer_data_;

  // Returns the base pointer of the host buffer. Use with caution as there is
  // no check on the requested type is the same as the underlying buffer data.
  template <typename T>
  T* base() const {
    return reinterpret_cast<T*>(base());
  }
};

template <typename T>
void ConcatCPUImpl(
    tsl::thread::ThreadPool* threadpool,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    int64_t element_byte_size, typename TTypes<T, 2>::Matrix* output) {
  size_t num_inputs = inputs.size();

  std::vector<ptrdiff_t> sizes;
  sizes.reserve(num_inputs);
  int64_t row_size = 0;
  for (const auto& input : inputs) {
    sizes.push_back(input->dimension(1));
    row_size += sizes.back();
  }

  // Max number of threads to use. Copied from TF impl.
  int num_threads = 4;
  if (threadpool) {
    num_threads = std::min(num_threads, threadpool->NumThreads());
  }
  const int64_t estimated_total_cost = output->size() * element_byte_size;
  num_threads = static_cast<int>(
      std::min<int64_t>(num_threads, estimated_total_cost / 16384));

  // This assumes the inputs are all have shape (1, num_elements). It copied
  // data to the output[start, end) range.
  auto work = [&sizes, &inputs, &output, &num_inputs, &element_byte_size](
                  int64_t start, int64_t end) {
    T* out = output->data();
    T* out_start = output->data() + start;
    T* out_end = output->data() + end;
    for (size_t j = 0; j < num_inputs; ++j) {
      ptrdiff_t size = sizes[j];
      ptrdiff_t offset = out_start - out;
      if (size <= offset) {
        out += size;
        continue;
      }
      const T* inp = &(*inputs[j])(0, 0);
      if (offset > 0) {
        out += offset;
        inp += offset;
        size -= offset;
      }
      size = std::min(size, out_end - out);
      if (size <= 0) break;
      if (std::is_same_v<T, tsl::tstring>) {
        for (ptrdiff_t i = 0; i < size; ++i) {
          out[i] = inp[i];
        }
      } else {
        memcpy(out, inp, size * element_byte_size);  // NOLINT
      }
      out += size;
    }
    // If this fails, concat is wrong, crash is ok.
    CHECK((out == out_end));
  };
  if (!threadpool || num_threads <= 1) {
    work(0, output->size());
    return;
  }

  threadpool->ParallelFor(
      output->size(),
      tsl::thread::ThreadPool::SchedulingParams::Adaptive(element_byte_size),
      work);
}

template <typename T>
void ConcatCPU(
    tsl::thread::ThreadPool* threadpool,
    const std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>>&
        inputs,
    int64_t element_byte_size, typename TTypes<T, 2>::Matrix* output) {
  ConcatCPUImpl<T>(threadpool, inputs, element_byte_size, output);
}

// Concatenates `inputs` into a single HostBuffer along the zeroth dimension
// using the given allocator. Requires that all elements of 'inputs' have
// element type T.
template <typename T>
absl::StatusOr<HostBuffer> Concat(
    tsl::thread::ThreadPool* threadpool,
    const absl::Span<const HostBuffer> inputs,
    Allocator* allocator = DefaultAllocator::GetInstance()) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError("inputs is empty");
  }
  const int input_dims = inputs[0].shape().dims().size();
  const xla::ifrt::Shape& input_shape = inputs[0].shape();
  const xla::ifrt::DType& input_dtype = inputs[0].dtype();

  // Short circuit if there is only one input.
  if (inputs.size() == 1) {
    return inputs[0];
  }

  // Note that we reduce the concat of k-dimensional HostBuffer into a two
  // dimensional concat. Assuming the dimensions of any input buffers are
  // {y0, y1,...,ym-1}, we flatten it to {1, y}, where y = Prod_i(yi).
  std::vector<std::unique_ptr<typename TTypes<T, 2>::ConstMatrix>> inputs_flat;
  inputs_flat.reserve(inputs.size());
  int64_t output_dim0 = 0;
  for (size_t i = 0; i < inputs.size(); ++i) {
    const HostBuffer& input = inputs[i];
    if (input.shape().dims().size() != input_dims) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Ranks of all input buffers should match: shape[0] = ", input_shape,
          " vs. shape[", i, "] = ", input.shape()));
    }
    for (int j = 1; j < input_dims; ++j) {
      if (input.shape().dims()[j] != input_shape.dims()[j]) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Dimensions of inputs should match: shape[0] = ", input_shape,
            " vs. shape[", i, "] = ", input.shape()));
      }
    }
    auto reshaped = input.shaped<T, 2>({1, input.shape().num_elements()});
    if (!reshaped.ok()) {
      return reshaped.status();
    }
    if (input.shape().num_elements() > 0) {
      inputs_flat.emplace_back(new
                               typename TTypes<T, 2>::ConstMatrix(*reshaped));
    }
    std::optional<int64_t> next_dim0 =
        xla::OverflowSafeAdd(output_dim0, input.shape().dims()[0]);
    if (!next_dim0.has_value()) {
      return absl::InvalidArgumentError("Concat output dimension overflow");
    }
    output_dim0 = *next_dim0;
  }

  std::vector<int64_t> output_shape_vec = {input_shape.dims().begin(),
                                           input_shape.dims().end()};

  output_shape_vec[0] = output_dim0;
  TF_ASSIGN_OR_RETURN(
      HostBuffer output_buffer,
      HostBuffer::Create(input_dtype, xla::ifrt::Shape(output_shape_vec),
                         allocator));
  if (!output_buffer.storage_dtype().byte_size().has_value() &&
      output_buffer.dtype().kind() != xla::ifrt::DType::Kind::kString) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unexpected storage dtype without byte size: ",
                     output_buffer.storage_dtype()));
  }
  int64_t cost_per_unit = 0;
  if (output_buffer.dtype().kind() == xla::ifrt::DType::Kind::kString) {
    cost_per_unit = sizeof(tsl::tstring);
  } else {
    cost_per_unit = *output_buffer.storage_dtype().byte_size();
  }
  if (output_buffer.shape().num_elements() > 0) {
    auto output_flat =
        output_buffer.shaped<T, 2>({1, output_buffer.shape().num_elements()});
    if (!output_flat.ok()) {
      return output_flat.status();
    }
    ConcatCPU<T>(threadpool, inputs_flat, cost_per_unit, &(*output_flat));
  }

  return output_buffer;
}

// Same as `Concat` above, but handles buffer dtype deduction automatically.
// If `threadpool` is nullptr, the concatenation will be done in a single
// thread.
absl::StatusOr<HostBuffer> Concat(
    tsl::thread::ThreadPool* threadpool, absl::Span<const HostBuffer> inputs,
    Allocator* allocator = DefaultAllocator::GetInstance());

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_HOST_BUFFER_H_

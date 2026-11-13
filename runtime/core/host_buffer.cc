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

#include "runtime/core/host_buffer.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <new>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer_type_utils.h"  // from @xla_serving
#include "runtime/core/host_buffer_utils.h"  // from @xla_serving
#include "xla/overflow_util.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

ABSL_FLAG(int64_t, xla_serving_host_buffer_max_elements, int64_t{1} << 32,
          "Maximum number of elements allowed in a HostBuffer to prevent "
          "OOM/DoS.");

namespace xla_serving {
namespace {
bool DataTypeCanUseMemCpy(xla::ifrt::DType dtype) {
  switch (dtype.kind()) {
    case xla::ifrt::DType::Kind::kS1:
    case xla::ifrt::DType::Kind::kU1:
    case xla::ifrt::DType::Kind::kS2:
    case xla::ifrt::DType::Kind::kU2:
    case xla::ifrt::DType::Kind::kS4:
    case xla::ifrt::DType::Kind::kU4:
    case xla::ifrt::DType::Kind::kF4E2M1FN:
    case xla::ifrt::DType::Kind::kPred:
    case xla::ifrt::DType::Kind::kS8:
    case xla::ifrt::DType::Kind::kU8:
    case xla::ifrt::DType::Kind::kF8E3M4:
    case xla::ifrt::DType::Kind::kF8E4M3:
    case xla::ifrt::DType::Kind::kF8E8M0FNU:
    // The following types are https://arxiv.org/abs/2209.05433
    case xla::ifrt::DType::Kind::kF8E4M3FN:
    case xla::ifrt::DType::Kind::kF8E4M3B11FNUZ:
    case xla::ifrt::DType::Kind::kF8E4M3FNUZ:
    case xla::ifrt::DType::Kind::kF8E5M2:
    case xla::ifrt::DType::Kind::kF8E5M2FNUZ:
    case xla::ifrt::DType::Kind::kS16:
    case xla::ifrt::DType::Kind::kU16:
    case xla::ifrt::DType::Kind::kF16:
    case xla::ifrt::DType::Kind::kBF16:
    case xla::ifrt::DType::Kind::kS32:
    case xla::ifrt::DType::Kind::kU32:
    case xla::ifrt::DType::Kind::kF32:
    case xla::ifrt::DType::Kind::kS64:
    case xla::ifrt::DType::Kind::kU64:
    case xla::ifrt::DType::Kind::kF64:
    case xla::ifrt::DType::Kind::kC64:
    case xla::ifrt::DType::Kind::kC128:
      return true;
    case xla::ifrt::DType::Kind::kToken:
    case xla::ifrt::DType::Kind::kOpaque:
    case xla::ifrt::DType::Kind::kInvalid:
    case xla::ifrt::DType::Kind::kString:
      return false;
  }
}

absl::StatusOr<int64_t> ValidateAndGetStorageBytes(
    xla::ifrt::DType dtype, const xla::ifrt::Shape& shape) {
  if (!DataTypeCanUseMemCpy(dtype) &&
      dtype.kind() != xla::ifrt::DType::Kind::kString) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported HostBuffer dtype: ", dtype));
  }

  int64_t num_elements = 1;
  for (int64_t d : shape.dims()) {
    if (d < 0) {
      return absl::InvalidArgumentError(
          absl::StrFormat("Invalid negative dimension in shape: %d", d));
    }
    auto [next_num_elements, overflow] =
        xla::OverflowSafeMultiply(num_elements, d);
    if (overflow) {
      return absl::InvalidArgumentError(
          absl::StrCat("HostBuffer shape overflow: ", shape));
    }
    num_elements = next_num_elements;
  }

  // Limit check to prevent OOM/DoS.
  int64_t max_elements =
      absl::GetFlag(FLAGS_xla_serving_host_buffer_max_elements);
  if (num_elements > max_elements) {
    return absl::InvalidArgumentError(
        absl::StrFormat("HostBuffer shape too large: %d elements (max %d)",
                        num_elements, max_elements));
  }

  // Also check byte size overflow.
  int64_t element_size = 0;
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    element_size = sizeof(tsl::tstring);
  } else if (dtype.byte_size().has_value()) {
    element_size = *dtype.byte_size();
  } else {
    element_size = 1;  // Coerced to kU8 in constructor
  }

  auto [byte_size, overflow] =
      xla::OverflowSafeMultiply(element_size, num_elements);
  if (overflow ||
      static_cast<uint64_t>(byte_size) > std::numeric_limits<size_t>::max()) {
    return absl::InvalidArgumentError(
        absl::StrCat("HostBuffer byte size overflow for shape: ", shape));
  }

  return byte_size;
}

absl::Status ValidateHostBufferShape(xla::ifrt::DType dtype,
                                     const xla::ifrt::Shape& shape) {
  return ValidateAndGetStorageBytes(dtype, shape).status();
}

}  // namespace

HostBuffer::HostBuffer(void* base, xla::ifrt::DType dtype,
                       xla::ifrt::Shape shape, std::shared_ptr<void> owner)
    : dtype_(dtype) {
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    buffer_data_.emplace(base, dtype, shape,
                         CreateDefaultStrides(shape.dims(), dtype),
                         std::move(owner));
    return;
  }

  xla::ifrt::DType storage_dtype = dtype;
  if (!dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }
  buffer_data_.emplace(base, storage_dtype, shape,
                       CreateDefaultStrides(shape.dims(), storage_dtype),
                       std::move(owner));
}

HostBuffer::HostBuffer(const xla::ifrt::DType& dtype,
                       const xla::ifrt::Shape& shape, Allocator* allocator)
    : dtype_(dtype) {
  if (allocator == nullptr) {
    allocator = DefaultAllocator::GetInstance();
  }
  int64_t num_elements = 1;
  for (int64_t d : shape.dims()) {
    CHECK_GE(d, 0) << "Invalid negative dimension in shape";
    auto [next_num_elements, dim_overflow] =
        xla::OverflowSafeMultiply(num_elements, d);
    // Prefer HostBuffer::Create() when constructing from untrusted shapes to
    // return absl::Status instead of crashing.
    CHECK(!dim_overflow) << "Integer overflow in shape dimensions";
    num_elements = next_num_elements;
  }

  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    auto [size, overflow] =
        xla::OverflowSafeMultiply(sizeof(tsl::tstring), num_elements);
    CHECK(!overflow &&
          static_cast<uint64_t>(size) <= std::numeric_limits<size_t>::max())
        << "Integer overflow in buffer allocation";
    void* aligned_raw_ptr =
        allocator->Allocate(static_cast<size_t>(size), Eigen::AlignedMax);
    CHECK(size == 0 || aligned_raw_ptr != nullptr)
        << "Failed to allocate buffer of size " << size;
    if (num_elements > 0) {
      // Initialize empty tstring.
      new (aligned_raw_ptr) tsl::tstring[num_elements];
    }
    buffer_data_.emplace(
        aligned_raw_ptr, dtype, shape,
        CreateDefaultStrides(shape.dims(), dtype),
        std::shared_ptr<void>(
            aligned_raw_ptr, [n = num_elements, allocator](void* ptr) {
              if (ptr == nullptr) return;
              tsl::tstring* tstring_ptr = reinterpret_cast<tsl::tstring*>(ptr);
              for (int64_t i = 0; i < n; ++i) {
                tstring_ptr[i].~tstring();
              }
              allocator->Free(ptr, Eigen::AlignedMax);
            }));
    return;
  }

  xla::ifrt::DType storage_dtype = dtype;
  if (!dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }

  auto [size, overflow] =
      xla::OverflowSafeMultiply(*storage_dtype.byte_size(), num_elements);
  CHECK(!overflow &&
        static_cast<uint64_t>(size) <= std::numeric_limits<size_t>::max())
      << "Integer overflow in buffer allocation";
  void* aligned_raw_ptr =
      allocator->Allocate(static_cast<size_t>(size), Eigen::AlignedMax);
  CHECK(size == 0 || aligned_raw_ptr != nullptr)
      << "Failed to allocate buffer of size " << size;
  auto owner = std::shared_ptr<void>(aligned_raw_ptr, [allocator](void* ptr) {
    if (ptr == nullptr) return;
    allocator->Free(ptr, Eigen::AlignedMax);
  });
  auto strides = CreateDefaultStrides(shape.dims(), storage_dtype);
  buffer_data_.emplace(aligned_raw_ptr, storage_dtype, shape, strides,
                       std::move(owner));
}

absl::StatusOr<HostBuffer> HostBuffer::Create(xla::ifrt::DType dtype,
                                              xla::ifrt::Shape shape,
                                              Allocator* allocator) {
  if (allocator == nullptr) {
    allocator = DefaultAllocator::GetInstance();
  }
  TF_ASSIGN_OR_RETURN(int64_t size, ValidateAndGetStorageBytes(dtype, shape));
  const int64_t num_elements = shape.num_elements();

  xla::ifrt::DType storage_dtype = dtype;
  if (dtype.kind() != xla::ifrt::DType::Kind::kString &&
      !dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }
  auto strides = CreateDefaultStrides(shape.dims(), storage_dtype);
  if (!strides.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to compute default strides for ", dtype, shape));
  }

  void* aligned_raw_ptr =
      allocator->Allocate(static_cast<size_t>(size), Eigen::AlignedMax);
  if (size > 0 && aligned_raw_ptr == nullptr) {
    return absl::ResourceExhaustedError(absl::StrCat(
        "Failed to allocate ", size, " bytes for HostBuffer ", dtype, shape));
  }

  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    if (num_elements > 0) {
      new (aligned_raw_ptr) tsl::tstring[num_elements];
    }
    auto owner = std::shared_ptr<void>(
        aligned_raw_ptr, [n = num_elements, allocator](void* ptr) {
          if (ptr == nullptr) return;
          tsl::tstring* tstring_ptr = reinterpret_cast<tsl::tstring*>(ptr);
          for (int64_t i = 0; i < n; ++i) {
            tstring_ptr[i].~tstring();
          }
          allocator->Free(ptr, Eigen::AlignedMax);
        });
    return HostBuffer(aligned_raw_ptr, dtype, std::move(shape), strides,
                      std::move(owner));
  }

  auto owner = std::shared_ptr<void>(aligned_raw_ptr, [allocator](void* ptr) {
    if (ptr == nullptr) return;
    allocator->Free(ptr, Eigen::AlignedMax);
  });
  return HostBuffer(aligned_raw_ptr, dtype, std::move(shape), strides,
                    std::move(owner));
}

HostBuffer::HostBuffer(
    void* base, xla::ifrt::DType dtype, xla::ifrt::Shape shape,
    const std::optional<absl::InlinedVector<int64_t, 4>>& byte_strides,
    std::shared_ptr<void> owner)
    : dtype_(dtype) {
  xla::ifrt::DType storage_dtype = dtype;
  if (dtype.kind() != xla::ifrt::DType::Kind::kString &&
      !dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }
  buffer_data_.emplace(base, storage_dtype, shape, byte_strides,
                       std::move(owner));
}

int64_t HostBuffer::total_storage_bytes() const {
  // For string, the HostBuffer only saves `tsl::tstring` which is a reference
  // representation of the actual string data. Its underlying data structure is
  // a union. sizeof(tsl::tstring) is the union here http://shortn/_qtz1wqAj1O.
  if (dtype_ == xla::ifrt::DType(xla::ifrt::DType::Kind::kString)) {
    return sizeof(tsl::tstring) * shape().num_elements();
  }
  CHECK(storage_dtype().byte_size().has_value());
  return *(storage_dtype().byte_size()) * shape().num_elements();
}

absl::StatusOr<HostBuffer> HostBuffer::SliceInDim(int dim, int64_t start,
                                                  int64_t end) const {
  if (!buffer_data_.has_value()) {
    return absl::InvalidArgumentError(
        "Cannot slice an uninitialized host buffer.");
  }
  if (dim < 0 || dim >= shape().dims().size() || start > end ||
      end > shape().dims()[dim]) {
    return absl::InvalidArgumentError(
        absl::StrFormat("Invalid slice of %v%v dim: %d st: %d limit: %d",
                        dtype(), shape(), dim, start, end));
  }
  std::vector<int64_t> output_dims{shape().dims().begin(),
                                   shape().dims().end()};
  output_dims[dim] = end - start;
  if (!buffer_data_->byte_strides().has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot slice a host buffer without byte strides. Buffer data dtype: ",
        buffer_data_->dtype()));
  }
  char* new_base = static_cast<char*>(buffer_data_->base()) +
                   start * buffer_data_->byte_strides().value().at(dim);
  HostBuffer sliced_host_buffer(
      new_base, dtype(), xla::ifrt::Shape(output_dims),
      buffer_data_->byte_strides(), std::make_shared<HostBuffer>(*this));
  return sliced_host_buffer;
}

absl::StatusOr<HostBuffer> HostBuffer::SliceAlongFirstDimension(
    int64_t start, int64_t end, Allocator* allocator) const {
  if (shape().dims().empty()) {
    return absl::InvalidArgumentError(
        "Host buffer must have at least one dimension for slicing.");
  }
  if (start < 0) {
    return absl::InvalidArgumentError("Start index must be non-negative.");
  }
  if (end < 0) {
    return absl::InvalidArgumentError("End index must be non-negative.");
  }
  if (start > end) {
    return absl::InvalidArgumentError("Start index must not larger than end.");
  }
  if (end > shape().dims()[0]) {
    return absl::InvalidArgumentError(
        "end index must be less than or equal to the first dimension size.");
  }
  int64_t dim0_size = shape().dims()[0];
  if ((start == 0) && (end == dim0_size)) {
    return *this;
  }
  if (dim0_size == 0 || !buffer_data_)
    return HostBuffer::Create(dtype(), shape(), allocator);

  if (dtype() == xla::ifrt::DType(xla::ifrt::DType::Kind::kString)) {
    std::vector<int64_t> output_dims{shape().dims().begin(),
                                     shape().dims().end()};
    output_dims[0] = end - start;
    int64_t elements_per_dim0 = 1;
    for (int i = 1; i < shape().dims().size(); ++i) {
      elements_per_dim0 *= output_dims[i];
    }
    HostBuffer sliced_host_buffer(
        reinterpret_cast<tsl::tstring*>(buffer_data_->base()) +
            start * elements_per_dim0,
        xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
        xla::ifrt::Shape(output_dims), buffer_data_->byte_strides(),
        std::make_shared<HostBuffer>(*this));
    return sliced_host_buffer;
  } else {
    return SliceInDim(/*dim=*/0, start, end);
  }
}

absl::StatusOr<HostBuffer> HostBuffer::Reshape(
    const xla::ifrt::Shape& shape) const {
  if (!buffer_data_) {
    return absl::FailedPreconditionError("HostBuffer is uninitialized.");
  }
  TF_RETURN_IF_ERROR(ValidateHostBufferShape(dtype(), shape));
  if (this->shape().num_elements() != shape.num_elements()) {
    return absl::InvalidArgumentError("Number of elements mismatch");
  }
  return HostBuffer(base(), dtype(), shape, owner());
}

absl::StatusOr<HostBuffer> HostBuffer::Bitcast(
    xla::ifrt::DType dtype, const xla::ifrt::Shape& shape) const {
  if (!buffer_data_) {
    return absl::FailedPreconditionError("HostBuffer is uninitialized.");
  }
  TF_RETURN_IF_ERROR(ValidateHostBufferShape(dtype, shape));
  if (!dtype_.byte_size().has_value() ||
      !storage_dtype().byte_size().has_value() ||
      !dtype.byte_size().has_value()) {
    return absl::InvalidArgumentError(
        "Cannot bitcast datatypes that do not have a fixed byte size.");
  }
  int64_t in_bytes =
      *storage_dtype().byte_size() * this->shape().num_elements();
  int64_t out_bytes = *dtype.byte_size() * shape.num_elements();
  if (in_bytes != out_bytes) {
    return absl::InvalidArgumentError("Total byte size mismatch");
  }
  return HostBuffer(base(), dtype, shape, owner());
}

absl::string_view HostBuffer::raw_data() const {
  CHECK(buffer_data_.has_value());
  CHECK(dtype_ != xla::ifrt::DType(xla::ifrt::DType::Kind::kString))
      << "Should not call raw_data() for string data type.";
  return absl::string_view(reinterpret_cast<const char*>(buffer_data_->base()),
                           total_storage_bytes());
}

// Returns true iff each row of dim0 is a multiple of Eigen::AlignedMax bytes.
bool HostBuffer::IsInnerDimsSizeAligned() const {
  const xla::ifrt::Shape& s = shape();
  const absl::Span<const int64_t> shape_dims = s.dims();
  if (shape_dims.empty()) {
    return false;
  }
  const int64_t dim0_size = shape_dims[0];
  if (dim0_size == 0) {
    return false;
  }
  if constexpr (Eigen::AlignedMax == 0) {
    return true;
  } else {
    int dtype_byte_size;
    xla::ifrt::DType dtype = storage_dtype();
    std::optional<int> dtype_byte_size_opt = dtype.byte_size();
    if (dtype_byte_size_opt.has_value()) {
      dtype_byte_size = *dtype_byte_size_opt;
    } else if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
      dtype_byte_size = sizeof(tsl::tstring);
    } else {
      // Storage dtype should never by sub-byte types.
      // Non-fixed, non-string types are not supported for now.
      return false;
    }
    const int64_t bytes_per_dim0_row =
        (s.num_elements() / dim0_size) * dtype_byte_size;
    return bytes_per_dim0_row % static_cast<int64_t>(Eigen::AlignedMax) == 0;
  }
}

absl::Status HostBuffer::SplitEasyCases(Allocator* allocator,
                                        absl::Span<const int64_t> sizes,
                                        std::vector<HostBuffer>& output,
                                        bool& easy_cases_done) const {
  easy_cases_done = false;
  // Special case 0: trivial 1-way split.
  if (sizes.size() == 1 && sizes[0] == shape().dims()[0]) {
    output.push_back(*this);
    easy_cases_done = true;
    return absl::OkStatus();
  }
  // Special case 1: each row of dim0 is aligned.
  if (IsInnerDimsSizeAligned()) {
    int64_t position = 0;
    for (const int64_t size : sizes) {
      TF_ASSIGN_OR_RETURN(
          HostBuffer sliced,
          SliceAlongFirstDimension(position, position + size, allocator));
      output.emplace_back(sliced);
      position += size;
    }
    easy_cases_done = true;
    return absl::OkStatus();
  }
  return absl::OkStatus();
}

absl::Status HostBuffer::SplitAlongFirstDimensionInternal(
    std::vector<HostBuffer>* output, absl::Span<const int64_t> sizes,
    Allocator* allocator) const {
  if (!DataTypeCanUseMemCpy(dtype()) &&
      dtype() != xla::ifrt::DType(xla::ifrt::DType::Kind::kString)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Host buffer type does not support split: ", dtype()));
  }
  if (dtype() == xla::ifrt::DType(xla::ifrt::DType::Kind::kString)) {
    auto from_strings = flat<tsl::tstring>();
    int64_t offset = 0;
    for (int64_t size : sizes) {
      std::vector<int64_t> output_dims{shape().dims().begin(),
                                       shape().dims().end()};
      output_dims[0] = size;
      TF_ASSIGN_OR_RETURN(
          HostBuffer split,
          HostBuffer::Create(dtype(), xla::ifrt::Shape(output_dims),
                             allocator));
      tsl::tstring* to_strings = split.unaligned_flat<tsl::tstring>().data();
      int64_t num_elements_in_split = split.shape().num_elements();
      CHECK_LE(offset + num_elements_in_split, shape().num_elements());
      for (int64_t i = 0; i < num_elements_in_split; ++i) {
        to_strings[i] = from_strings(offset + i);
      }
      offset += num_elements_in_split;
      output->push_back(std::move(split));
    }
    return absl::OkStatus();
  }
  absl::string_view from_data = raw_data();
  int64_t offset = 0;
  for (int64_t size : sizes) {
    std::vector<int64_t> output_dims{shape().dims().begin(),
                                     shape().dims().end()};
    output_dims[0] = size;
    TF_ASSIGN_OR_RETURN(
        HostBuffer split,
        HostBuffer::Create(dtype(), xla::ifrt::Shape(output_dims), allocator));

    // We use string_view as a convenient map over the tensor buffer,
    // but we cast the type to get to the underlying buffer to do the
    // copy.
    absl::string_view to_data = split.raw_data();
    if (offset + to_data.size() > from_data.size()) {
      return absl::InternalError(
          "Splitting host buffer failed: to_data out of range.");
    }
    if (!to_data.empty()) {
      memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,
             to_data.size());
    }

    offset += to_data.size();
    output->push_back(std::move(split));
  }
  return absl::OkStatus();
}

absl::StatusOr<std::vector<HostBuffer>> HostBuffer::SplitAlongFirstDimension(
    absl::Span<const int64_t> sizes, Allocator* allocator) const {
  if (shape().dims().empty()) {
    return absl::InvalidArgumentError(
        "Host buffer must have at least one dimension for splitting.");
  }
  int64_t total_size = 0;
  for (int64_t size : sizes) {
    if (size < 0) {
      return absl::InvalidArgumentError(
          "Split sizes must be non-negative.");
    }
    std::optional<int64_t> next_total = xla::OverflowSafeAdd(total_size, size);
    if (!next_total.has_value()) {
      return absl::InvalidArgumentError("Split sizes overflow.");
    }
    total_size = *next_total;
  }
  if (total_size != shape().dims()[0]) {
    return absl::InvalidArgumentError(
        "The values in 'sizes' do not sum to the zeroth-dimension size of "
        "'shape'");
  }
  std::vector<HostBuffer> output;
  output.reserve(sizes.size());
  bool easy_cases_done = false;
  TF_RETURN_IF_ERROR(SplitEasyCases(allocator, sizes, output, easy_cases_done));
  if (!easy_cases_done) {
    TF_RETURN_IF_ERROR(
        SplitAlongFirstDimensionInternal(&output, sizes, allocator));
  }
  return output;
}

absl::StatusOr<HostBuffer> Concat(tsl::thread::ThreadPool* threadpool,
                                  absl::Span<const HostBuffer> inputs,
                                  Allocator* allocator) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError("inputs is empty");
  }
  if (!inputs[0].has_data()) {
    return absl::InvalidArgumentError(
        "Input buffer at index 0 is uninitialized");
  }
  xla::ifrt::DType::Kind type = inputs[0].dtype().kind();
  switch (type) {
#define CASE_TYPE(T, DTYPE_KIND)                            \
  case DTYPE_KIND: {                                        \
    auto output = Concat<T>(threadpool, inputs, allocator); \
    if (!output.ok()) {                                     \
      return output.status();                               \
    }                                                       \
    return *output;                                         \
  } break;
    APPLY_POD_CASE(CASE_TYPE)
#undef CASE_TYPE
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported data type of ", type));
  }
}

absl::StatusOr<HostBuffer> HostBuffer::FromProto(const HostBufferProto& proto) {
  return FromProto(proto, DefaultAllocator::GetInstance());
}

absl::StatusOr<HostBuffer> HostBuffer::FromProto(const HostBufferProto& proto,
                                                 Allocator* allocator) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype,
                      xla::ifrt::DType::FromProto(proto.dtype()));
  TF_ASSIGN_OR_RETURN(xla::ifrt::Shape shape,
                      xla::ifrt::Shape::FromProto(proto.shape()));
  TF_ASSIGN_OR_RETURN(int64_t expected_bytes,
                      ValidateAndGetStorageBytes(dtype, shape));
  const int64_t num_elements = shape.num_elements();

  // Validate payload variant and size before allocating host memory so a small
  // malformed proto claiming a huge shape cannot trigger multi-GB allocations.
  switch (proto.data_case()) {
    case HostBufferProto::kRawData: {
      if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
        return absl::InvalidArgumentError(
            "Proto has raw_data but dtype is string.");
      }
      if (static_cast<uint64_t>(expected_bytes) != proto.raw_data().size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Mismatched raw data size: expected ", expected_bytes,
                         " but got ", proto.raw_data().size()));
      }
      TF_ASSIGN_OR_RETURN(
          HostBuffer host_buffer,
          HostBuffer::Create(dtype, std::move(shape), allocator));
      absl::string_view raw_data = host_buffer.raw_data();
      if (!raw_data.empty()) {
        memcpy(const_cast<char*>(raw_data.data()), proto.raw_data().data(),
               raw_data.size());
      }
      return host_buffer;
    }
    case HostBufferProto::kStringData: {
      if (dtype.kind() != xla::ifrt::DType::Kind::kString) {
        return absl::InvalidArgumentError(
            "Proto has string_data but dtype is not string.");
      }
      if (num_elements != proto.string_data().elements_size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Mismatched number of string elements: expected ",
                         num_elements, " but got ",
                         proto.string_data().elements_size()));
      }
      TF_ASSIGN_OR_RETURN(
          HostBuffer host_buffer,
          HostBuffer::Create(dtype, std::move(shape), allocator));
      if (num_elements > 0) {
        auto tstrings = host_buffer.unaligned_flat<tsl::tstring>();
        for (int64_t i = 0; i < num_elements; ++i) {
          tstrings(i) = proto.string_data().elements(i);
        }
      }
      return host_buffer;
    }
    case HostBufferProto::DATA_NOT_SET:
      if (num_elements != 0) {
        return absl::InvalidArgumentError(
            "HostBuffer is non-empty but has no data.");
      }
      return HostBuffer::Create(dtype, std::move(shape), allocator);
  }
  return absl::InvalidArgumentError("Unknown HostBufferProto data_case.");
}

HostBufferProto HostBuffer::ToHostBufferProto(const HostBuffer& host_buffer) {
  HostBufferProto proto;
  xla::ifrt::DType dtype = host_buffer.dtype();
  xla::ifrt::Shape shape = host_buffer.shape();
  *proto.mutable_dtype() = dtype.ToProto();
  *proto.mutable_shape() = shape.ToProto();
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    auto tstrings = host_buffer.unaligned_flat<tsl::tstring>();
    for (int64_t i = 0; i < shape.num_elements(); ++i) {
      proto.mutable_string_data()->add_elements(std::string(tstrings(i)));
    }
  } else {
    absl::string_view raw_data_view = host_buffer.raw_data();
    proto.set_raw_data(raw_data_view);
  }
  return proto;
}

std::string BufferData::DebugString() const {
  std::string out = absl::StrCat(dtype_, shape_);
  if (byte_strides_.has_value()) {
    AppendRecursive(&out, dtype_, base_, 0, *byte_strides_, shape_.dims());
  }
  return out;
}

}  // namespace xla_serving

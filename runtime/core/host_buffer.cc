#include "third_party/neptune/runtime/core/host_buffer.h"

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <memory>
#include <new>
#include <numeric>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/australis/google/ifrt/host_array.h"
#include "third_party/australis/google/ifrt/host_array_ops.h"
#include "third_party/eigen3/Eigen/Core"  // IWYU pragma: keep, go/lsc-fix-eigen-includes
#include "third_party/neptune/runtime/core/allocator.h"
#include "third_party/neptune/runtime/core/default_allocator.h"
#include "third_party/neptune/runtime/core/host_buffer_type_utils.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/errors.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"
#include "third_party/tensorflow/tsl/platform/tstring.h"

// go/lsc-fix-eigen-includes
// IWYU pragma: no_include "third_party/eigen3/Eigen/src/Core/util/ConfigureVectorization.h"

namespace jsv {
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

}  // namespace

HostBuffer::HostBuffer(void* base, xla::ifrt::DType dtype,
                       xla::ifrt::Shape shape, std::shared_ptr<void> owner)
    : dtype_(dtype) {
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    BufferData buffer_data(base, dtype, shape, owner);
    string_data_ = StringData{.buffer_data = std::move(buffer_data)};
    return;
  }

  xla::ifrt::DType storage_dtype = dtype;
  if (!dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }
  host_array_ = tsl::MakeRef<aux::MutableHostArray>(
      base, shape, storage_dtype,
      aux::CreateDefaultStrides(shape.dims(), storage_dtype), owner);
}

HostBuffer::HostBuffer(const xla::ifrt::DType& dtype,
                       const xla::ifrt::Shape& shape, Allocator* allocator)
    : dtype_(dtype) {
  if (allocator == nullptr) {
    allocator = DefaultAllocator::GetInstance();
  }
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    size_t size = sizeof(tsl::tstring) * shape.num_elements();
    void* aligned_raw_ptr = allocator->Allocate(size, EIGEN_MAX_ALIGN_BYTES);
    // Initialize empty tstring.
    new (aligned_raw_ptr) tsl::tstring[shape.num_elements()];
    BufferData buffer_data(
        aligned_raw_ptr, dtype, shape,
        std::shared_ptr<void>(
            aligned_raw_ptr, [n = shape.num_elements(), allocator](void* ptr) {
              tsl::tstring* tstring_ptr = reinterpret_cast<tsl::tstring*>(ptr);
              for (int i = 0; i < n; ++i) {
                tstring_ptr[i].~tstring();
              }
              allocator->Free(ptr, EIGEN_MAX_ALIGN_BYTES);
            }));
    string_data_ = StringData{.buffer_data = std::move(buffer_data)};
    return;
  }

  xla::ifrt::DType storage_dtype = dtype;
  if (!dtype.byte_size().has_value()) {
    storage_dtype = xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
  }

  size_t size = *storage_dtype.byte_size() * shape.num_elements();
  void* aligned_raw_ptr = allocator->Allocate(size, EIGEN_MAX_ALIGN_BYTES);
  host_array_ = aux::MutableHostArray::Create(
      absl::MakeSpan(reinterpret_cast<uint8_t*>(aligned_raw_ptr), size),
      shape.dims(), storage_dtype,
      std::shared_ptr<void>(aligned_raw_ptr, [allocator](void* ptr) {
        allocator->Free(ptr, EIGEN_MAX_ALIGN_BYTES);
      }));
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
  if (dim0_size == 0 || (!host_array_ && !string_data_.has_value()))
    return HostBuffer(dtype(), shape(), allocator);

  if (host_array_) {
    TF_ASSIGN_OR_RETURN(
        auto new_host_array,
        aux::ops::SliceInDim(host_array_, /*dim=*/0, start, end));
    // This copy is ok because the underlying HostArray is ref counted and this
    // HostBuffer copy is cheap.
    return HostBuffer(const_cast<void*>(new_host_array->raw_data()), dtype(),
                      new_host_array->shape(),
                      /*owner=*/std::make_shared<HostBuffer>(*this));
  }

  // Handle string data.
  std::vector<int64_t> output_dims{shape().dims().begin(),
                                   shape().dims().end()};
  int64_t elements_per_dim0 = 1;
  for (int i = 1; i < shape().dims().size(); ++i) {
    elements_per_dim0 *= output_dims[i];
  }
  output_dims[0] = end - start;
  HostBuffer sliced_host_buffer(
      reinterpret_cast<tsl::tstring*>(string_data_->buffer_data.base()) +
          start * elements_per_dim0,
      xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
      xla::ifrt::Shape(output_dims), std::make_shared<HostBuffer>(*this));

  return sliced_host_buffer;
}

absl::string_view HostBuffer::raw_data() const {
  CHECK(dtype_ != xla::ifrt::DType(xla::ifrt::DType::Kind::kString))
      << "Should not call raw_data() for string data type.";
  return absl::string_view(
      reinterpret_cast<const char*>(host_array_->raw_data()),
      total_storage_bytes());
}

// Returns true iff each row of dim0 is a multiple of EIGEN_MAX_ALIGN_BYTES
// bytes.
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
#if EIGEN_MAX_ALIGN_BYTES == 0
  return true;
#else
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
  const int64_t bytes_per_dim0_row = (s.num_elements() / dim0_size) *
                                     dtype_byte_size;
  return bytes_per_dim0_row % EIGEN_MAX_ALIGN_BYTES == 0;
#endif
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
    return absl::InvalidArgumentError(absl::StrCat(
        "Host buffer type does not support split: ", dtype().DebugString()));
  }
  if (dtype() == xla::ifrt::DType(xla::ifrt::DType::Kind::kString)) {
    auto from_strings = flat<tsl::tstring>();
    int offset = 0;
    for (int64_t size : sizes) {
      std::vector<int64_t> output_dims{shape().dims().begin(),
                                       shape().dims().end()};
      output_dims[0] = size;
      HostBuffer split(dtype(), xla::ifrt::Shape(output_dims), allocator);
      tsl::tstring* to_strings = split.unaligned_flat<tsl::tstring>().data();
      int num_elements_in_split = split.shape().num_elements();
      CHECK_LE(offset + num_elements_in_split, shape().num_elements());
      for (int i = 0; i < num_elements_in_split; ++i) {
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
    HostBuffer split(dtype(), xla::ifrt::Shape(output_dims), allocator);

    // We use string_view as a convenient map over the tensor buffer,
    // but we cast the type to get to the underlying buffer to do the
    // copy.
    absl::string_view to_data = split.raw_data();
    if (offset + to_data.size() > from_data.size()) {
      return absl::InternalError(
          "Splitting host buffer failed: to_data out of range.");
    }
    memcpy(const_cast<char*>(to_data.data()), from_data.data() + offset,
           to_data.size());

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
  int64_t total_size = std::accumulate(sizes.begin(), sizes.end(), 0ll);
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
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype,
                      xla::ifrt::DType::FromProto(proto.dtype()));
  TF_ASSIGN_OR_RETURN(xla::ifrt::Shape shape,
                      xla::ifrt::Shape::FromProto(proto.shape()));
  HostBuffer host_buffer(dtype, shape);
  switch (proto.data_case()) {
    case HostBufferProto::kRawData: {
      if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
        return absl::InvalidArgumentError(
            "Proto has raw_data but dtype is string.");
      }
      absl::string_view raw_data = host_buffer.raw_data();
      if (raw_data.size() != proto.raw_data().size()) {
        return absl::InternalError(
            absl::StrCat("Mismatched raw data size: expected ", raw_data.size(),
                         " but got ", proto.raw_data().size()));
      }
      memcpy(const_cast<char*>(raw_data.data()), proto.raw_data().data(),
             raw_data.size());
      return host_buffer;
    }
    case HostBufferProto::kStringData: {
      if (dtype.kind() != xla::ifrt::DType::Kind::kString) {
        return absl::InvalidArgumentError(
            "Proto has string_data but dtype is not string.");
      }
      if (shape.num_elements() != proto.string_data().elements_size()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Mismatched number of string elements: expected ",
                         shape.num_elements(), " but got ",
                         proto.string_data().elements_size()));
      }
      auto tstrings = host_buffer.unaligned_flat<tsl::tstring>();
      for (int i = 0; i < shape.num_elements(); ++i) {
        tstrings(i) = proto.string_data().elements(i);
      }
      return host_buffer;
    }
    case HostBufferProto::DATA_NOT_SET:
      if (shape.num_elements() != 0) {
        return absl::InvalidArgumentError(
            "HostBuffer is non-empty but has no data.");
      }
      return host_buffer;
  }
}

HostBufferProto HostBuffer::ToHostBufferProto(const HostBuffer& host_buffer) {
  HostBufferProto proto;
  xla::ifrt::DType dtype = host_buffer.dtype();
  xla::ifrt::Shape shape = host_buffer.shape();
  *proto.mutable_dtype() = dtype.ToProto();
  *proto.mutable_shape() = shape.ToProto();
  if (dtype.kind() == xla::ifrt::DType::Kind::kString) {
    auto tstrings = host_buffer.unaligned_flat<tsl::tstring>();
    for (int i = 0; i < shape.num_elements(); ++i) {
      proto.mutable_string_data()->add_elements(std::string(tstrings(i)));
    }
  } else {
    absl::string_view raw_data_view = host_buffer.raw_data();
    proto.set_raw_data(raw_data_view);
  }
  return proto;
}

}  // namespace jsv

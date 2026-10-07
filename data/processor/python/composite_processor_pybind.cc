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

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <functional>
#include <iterator>
#include <map>
#include <memory>
#include <numeric>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "google/protobuf/any.pb.h"
#include "absl/base/fast_type_id.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "include/dlpack/dlpack.h"  // from @dlpack
#include "absl/status/status_macros.h"
#include "include/nanobind/nanobind.h"  // from @nanobind_bazel
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/composite_processor.h"  // from @xla_serving
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/cc/utils/buffer_utils.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/processor/python/transform_pybind.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "include/pybind11/gil.h"  // from @pybind11_bazel
#include "include/pybind11/numpy.h"  // from @pybind11_bazel
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel
#include "include/pybind11/pytypes.h"  // from @pybind11_bazel
#include "include/pybind11/stl.h"  // from @pybind11_bazel
#include "pybind11_abseil/status_casters.h"  // from @pybind11_abseil
#include "pybind11_protobuf/native_proto_caster.h"  // from @pybind11_protobuf
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/types.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

#include "xla/tsl/python/lib/core/numpy.h"

namespace xla_serving::data {
namespace composite {
namespace {

namespace py = pybind11;

// Helper struct to hold inputs because api::ExecuteContext::inputs is a span.
struct PyExecuteContext : public api::ExecuteContext {
  std::vector<api::Value> inputs_storage;
};

struct PyShapeInferenceContext : public api::ShapeInferenceContext {
  std::vector<api::ShapeInferenceValue> inputs_storage;
};

// Helper struct for holding PyObject reference and its associated Python buffer
// protocol lock (via py::buffer_info) for the lifetime of a tsl::tstring.
struct PyBufferOwner {
  py::handle py_handle;
  std::optional<py::buffer_info> info;

  PyBufferOwner(py::handle handle, py::buffer_info&& buffer_info)
      : py_handle(std::move(handle)), info(std::move(buffer_info)) {
    py::gil_scoped_acquire acquire;
    py_handle.inc_ref();
  }

  PyBufferOwner(PyBufferOwner&& other) noexcept
      : py_handle(std::move(other.py_handle)), info(std::move(other.info)) {
    other.py_handle = nullptr;
  }

  PyBufferOwner(const PyBufferOwner&) = delete;
  PyBufferOwner& operator=(const PyBufferOwner&) = delete;

  ~PyBufferOwner() {
    py::gil_scoped_acquire acquire;
    info.reset();
    if (py_handle) {
      py_handle.dec_ref();
      py_handle = nullptr;
    }
  }
};

class NumpyFlatIterator {
 public:
  using iterator_category = std::input_iterator_tag;
  using value_type = py::handle;
  using difference_type = std::ptrdiff_t;
  using pointer = const py::handle*;
  using reference = py::handle;

  static absl::StatusOr<NumpyFlatIterator> Create(py::array arr,
                                                  size_t start_index) {
    if (start_index > arr.size()) {
      return absl::OutOfRangeError(
          absl::StrCat("Iterator start index ", start_index,
                       " out of bounds for array of size ", arr.size(),
                       ". This should never happen; if you see this error "
                       "message, please report this as a bug."));
    }
    return NumpyFlatIterator(arr, start_index);
  }

  NumpyFlatIterator(NumpyFlatIterator&&) = default;
  NumpyFlatIterator& operator=(NumpyFlatIterator&&) = default;
  NumpyFlatIterator(const NumpyFlatIterator&) = default;
  NumpyFlatIterator& operator=(const NumpyFlatIterator&) = default;

  reference operator*() const {
    ssize_t offset = 0;
    for (int j = 0; j < ndim_; ++j) {
      offset += index_[j] * strides_[j];
    }
    return py::handle(*reinterpret_cast<PyObject* const*>(data_ + offset));
  }

  NumpyFlatIterator& operator++() {
    flat_index_++;
    for (int j = ndim_ - 1; j >= 0; --j) {
      index_[j]++;
      if (index_[j] < shape_[j]) {
        break;
      }
      index_[j] = 0;
    }
    return *this;
  }

  bool operator==(const NumpyFlatIterator& other) const {
    return flat_index_ == other.flat_index_;
  }
  bool operator!=(const NumpyFlatIterator& other) const {
    return !(*this == other);
  }

 private:
  NumpyFlatIterator(py::array arr, size_t start_index)
      : data_(reinterpret_cast<const char*>(arr.data())),
        shape_(arr.shape()),
        strides_(arr.strides()),
        ndim_(arr.ndim()),
        flat_index_(start_index),
        index_(ndim_, 0) {
    if (start_index > 0 && start_index < arr.size()) {
      ssize_t temp = start_index;
      for (int j = ndim_ - 1; j >= 0; --j) {
        index_[j] = temp % shape_[j];
        temp /= shape_[j];
      }
    }
  }

  const char* data_;
  const ssize_t* shape_;
  const ssize_t* strides_;
  ssize_t ndim_;
  size_t flat_index_;
  std::vector<ssize_t> index_;
};

bool IsBufferContiguous(const py::buffer_info& info) {
  if (info.size == 0 || info.ndim == 0) {
    return true;
  }
  if (info.strides.size() != info.ndim || info.shape.size() != info.ndim) {
    return false;
  }
  ssize_t expected_stride = info.itemsize;
  for (ssize_t i = info.ndim - 1; i >= 0; --i) {
    if (info.strides[i] != expected_stride) [[unlikely]] {
      return false;
    }
    expected_stride *= info.shape[i];
  }
  return true;
}

template <typename InputIterator>
absl::StatusOr<xla_serving::HostBuffer> CreateSharedStringHostBuffer(
    InputIterator begin, InputIterator end, const xla::ifrt::Shape& shape) {
  xla_serving::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kString), shape);
  tsl::tstring* ptr = reinterpret_cast<tsl::tstring*>(hb.base());

  auto it = begin;
  size_t num_elements = shape.num_elements();
  for (size_t i = 0; i < num_elements; ++i, ++ptr) {
    if (it == end) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Too few elements, expected ", num_elements, ", got ", i));
    }
    py::handle item = *it;
    if (item.is_none()) {
      // Default constructed tstring is empty.
    } else if (py::isinstance<py::buffer>(item)) {
      // Extract core C buffer directly from any buffer-like object
      py::buffer_info info = py::cast<py::buffer>(item).request();
      if (!IsBufferContiguous(info)) {
        return absl::InvalidArgumentError(
            "Buffer object must be C-contiguous.");
      }
      const char* data_ptr = static_cast<const char*>(info.ptr);
      size_t data_size = info.size * info.itemsize;
      tsl::tstring::owner<PyBufferOwner>* owner =
          new tsl::tstring::owner<PyBufferOwner>(
              PyBufferOwner(item, std::move(info)));
      ptr->assign_as_shared_view(absl::string_view(data_ptr, data_size), owner);
      owner->Unref();
    } else if (py::isinstance<py::str>(item)) {
      *ptr = py::cast<std::string>(item);
    } else {
      return absl::InvalidArgumentError(
          absl::StrCat("Elements must be str, buffer, or None, got ",
                       item.get_type().attr("__name__").cast<std::string>()));
    }
    ++it;
  }
  if (it != end) {
    return absl::InvalidArgumentError(
        absl::StrCat("Too many elements, expected ", num_elements));
  }
  return hb;
}

absl::StatusOr<DLDataType> IfrtDTypeToDLDataType(xla::ifrt::DType dt) {
  switch (dt.kind()) {
    case xla::ifrt::DType::Kind::kPred:
      return DLDataType{kDLBool, 8, 1};
    case xla::ifrt::DType::Kind::kS8:
      return DLDataType{kDLInt, 8, 1};
    case xla::ifrt::DType::Kind::kS16:
      return DLDataType{kDLInt, 16, 1};
    case xla::ifrt::DType::Kind::kS32:
      return DLDataType{kDLInt, 32, 1};
    case xla::ifrt::DType::Kind::kS64:
      return DLDataType{kDLInt, 64, 1};
    case xla::ifrt::DType::Kind::kU8:
      return DLDataType{kDLUInt, 8, 1};
    case xla::ifrt::DType::Kind::kU16:
      return DLDataType{kDLUInt, 16, 1};
    case xla::ifrt::DType::Kind::kU32:
      return DLDataType{kDLUInt, 32, 1};
    case xla::ifrt::DType::Kind::kU64:
      return DLDataType{kDLUInt, 64, 1};
    case xla::ifrt::DType::Kind::kF16:
      return DLDataType{kDLFloat, 16, 1};
    case xla::ifrt::DType::Kind::kF32:
      return DLDataType{kDLFloat, 32, 1};
    case xla::ifrt::DType::Kind::kF64:
      return DLDataType{kDLFloat, 64, 1};
    case xla::ifrt::DType::Kind::kBF16:
      return DLDataType{kDLBfloat, 16, 1};
    case xla::ifrt::DType::Kind::kC64:
      return DLDataType{kDLComplex, 64, 1};
    case xla::ifrt::DType::Kind::kC128:
      return DLDataType{kDLComplex, 128, 1};
    default:
      return absl::UnimplementedError("IFRT DType has no DLPack equivalent");
  }
}

absl::StatusOr<xla::ifrt::DType> DLDataTypeToIfrtDType(DLDataType dtype) {
  if (dtype.lanes != 1) {
    return absl::InvalidArgumentError(
        "Vectorized DLPack types are not supported");
  }
  switch (dtype.code) {
    case kDLBool:
      if (dtype.bits != 8) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported kDLBool bits: ", dtype.bits, ". Expected 8 bits."));
      }
      return xla::ifrt::DType(xla::ifrt::DType::Kind::kPred);
    case kDLInt:
      switch (dtype.bits) {
        case 8:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kS8);
        case 16:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kS16);
        case 32:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kS32);
        case 64:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kS64);
        default:
          return absl::InvalidArgumentError("Unsupported integer bits");
      }
    case kDLUInt:
      switch (dtype.bits) {
        case 8:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kU8);
        case 16:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kU16);
        case 32:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kU32);
        case 64:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kU64);
        default:
          return absl::InvalidArgumentError(
              "Unsupported unsigned integer bits");
      }
    case kDLFloat:
      switch (dtype.bits) {
        case 16:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kF16);
        case 32:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kF32);
        case 64:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kF64);
        default:
          return absl::InvalidArgumentError("Unsupported float bits");
      }
    case kDLBfloat:
      if (dtype.bits == 16) {
        return xla::ifrt::DType(xla::ifrt::DType::Kind::kBF16);
      }
      return absl::InvalidArgumentError("Unsupported bfloat bits");
    case kDLComplex:
      switch (dtype.bits) {
        case 64:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kC64);
        case 128:
          return xla::ifrt::DType(xla::ifrt::DType::Kind::kC128);
        default:
          return absl::InvalidArgumentError("Unsupported complex bits");
      }
    default:
      return absl::InvalidArgumentError("Unsupported DLPack type code");
  }
}

template <bool IsVersioned>
class DLPackHostBufferBuilder {
  using MT = std::conditional_t<IsVersioned, DLManagedTensorVersioned,
                                DLManagedTensor>;

 public:
  explicit DLPackHostBufferBuilder(py::capsule cap, std::optional<bool> copy)
      : cap_(cap), copy_(copy) {}
  xla_serving::HostBuffer ConstructHostBuffer() {
    // Extract the tensor.
    auto* dl_managed_tensor = static_cast<MT*>(cap_.get_pointer());
    if (!dl_managed_tensor) {
      throw py::value_error(absl::StrCat("PyCapsule does not contain a valid ",
                                         GetDLManagedTensorTypeName()));
    }
    auto dl_tensor = &dl_managed_tensor->dl_tensor;
    bool producer_made_copy = false;
    if constexpr (IsVersioned) {
      if (dl_managed_tensor->flags & DLPACK_FLAG_BITMASK_IS_COPIED) {
        producer_made_copy = true;
      }
    }

    auto dtype = DLDataTypeToIfrtDType(dl_tensor->dtype);
    if (!dtype.ok()) {
      throw py::value_error(std::string(dtype.status().message()));
    }

    // Construct the shape vector for the HostBuffer.
    std::vector<int64_t> shape_vec(dl_tensor->ndim);
    for (int i = 0; i < dl_tensor->ndim; ++i) {
      shape_vec[i] = dl_tensor->shape[i];
    }
    xla::ifrt::Shape shape(shape_vec);

    // Check contiguous strides.
    // When the strides field in the DLTensor struct is NULL, it
    // indicates a row-major compact array. However, strides can only be
    // null when ndims = 0.
    bool is_contiguous = true;
    if (dl_tensor->strides != nullptr) {
      int64_t expected_stride = 1;
      for (int i = dl_tensor->ndim - 1; i >= 0; --i) {
        if (dl_tensor->strides[i] != expected_stride) {
          is_contiguous = false;
          break;
        }
        expected_stride *= dl_tensor->shape[i];
      }
    }

    // Check if we need to copy the data, and throw an error if a copy
    // is needed but copy=False was requested.
    void* data =
        reinterpret_cast<char*>(dl_tensor->data) + dl_tensor->byte_offset;

    bool needs_alignment_copy = !utils::IsBufferEigenAligned(data);
    if ((copy_.has_value() && !*copy_)) {
      if (needs_alignment_copy) {
        throw py::value_error(
            "copy=False requested, but a copy is required due to "
            "non-eigen-aligned data");
      }
      if (!is_contiguous) {
        throw py::value_error(
            "copy=False requested, but a copy is required due to "
            "non-contiguous data");
      }
      if (producer_made_copy) {
        throw py::value_error(
            "copy=False requested, but a copy is required due to "
            "producer making a copy");
      }
    }
    // Consume the capsule now that we've verified no errors will occur.
    // The shared_ptr 'owner' will go out of scope and safely call the
    // DLPack deleter on the consumed capsule.
    auto owner = std::make_shared<DLManagedTensorOwner>(dl_managed_tensor);

    // DLPack requires us to always consume the tensor.
    ConsumeDLManagedTensorCapsule();

    // If no copy is needed, return the buffer directly.
    if (!needs_alignment_copy && is_contiguous &&
        (!copy_.has_value() || !*copy_)) {
      return xla_serving::HostBuffer(data, *dtype, shape, owner);
    }

    // Perform a copy.
    const int elem_size = dl_tensor->dtype.bits / 8;
    xla_serving::HostBuffer hb(*dtype, shape);
    if (is_contiguous) {
      std::memcpy(hb.base(), data, shape.num_elements() * elem_size);
      return hb;
    }
    // Copy manually via strides.
    std::vector<int64_t> dst_strides(dl_tensor->ndim);
    int64_t curr_stride = elem_size;
    for (int i = dl_tensor->ndim - 1; i >= 0; --i) {
      dst_strides[i] = curr_stride;
      curr_stride *= dl_tensor->shape[i];
    }
    // Recursively copy element by element.
    auto recursive_copy = [&](auto& self, int dim, const char* src,
                              char* dst) -> void {
      if (dim >= dl_tensor->ndim) {
        std::memcpy(dst, src, elem_size);
        return;
      }
      int64_t src_stride = dl_tensor->strides[dim] * elem_size;
      int64_t dst_stride = dst_strides[dim];
      int64_t dim_size = dl_tensor->shape[dim];
      for (int64_t i = 0; i < dim_size; ++i) {
        self(self, dim + 1, src + i * src_stride, dst + i * dst_stride);
      }
    };
    recursive_copy(recursive_copy, 0, static_cast<const char*>(data),
                   static_cast<char*>(hb.base()));
    return hb;
  }

 private:
  struct DLManagedTensorOwner {
    MT* absl_nonnull t;
    explicit DLManagedTensorOwner(MT* t) : t(t) {}
    ~DLManagedTensorOwner() {
      if (t && t->deleter) {
        t->deleter(t);
      }
    }
  };

  py::capsule cap_;
  std::optional<bool> copy_;

  static absl::string_view GetDLManagedTensorTypeName() {
    if constexpr (IsVersioned) {
      return "DLManagedTensorVersioned";
    }
    return "DLManagedTensor";
  }

  static absl::string_view GetConsumedCapsuleName() {
    if constexpr (IsVersioned) {
      return "used_dltensor_versioned";
    }
    return "used_dltensor";
  }

  void ConsumeDLManagedTensorCapsule() {
    if (PyCapsule_SetName(cap_.ptr(), GetConsumedCapsuleName().data()) != 0) {
      throw py::error_already_set();
    }
  }
};

xla_serving::HostBuffer HostBufferFromDLPack(py::object obj, std::optional<bool> copy) {
  if (!py::hasattr(obj, "__dlpack__") ||
      !py::hasattr(obj, "__dlpack_device__")) {
    throw py::value_error(
        "The array passed to from_dlpack must have __dlpack__ and "
        "__dlpack_device__ methods.");
  }
  py::tuple device = obj.attr("__dlpack_device__")();
  if (device[0].cast<int>() != kDLCPU) {
    throw py::buffer_error("HostBuffer only supports DLPack tensors on CPU");
  }

  py::object dlpack_obj = obj.attr("__dlpack__")();
  py::capsule cap;
  try {
    cap = py::cast<py::capsule>(dlpack_obj);
  } catch (const py::cast_error&) {
    throw py::type_error("Expected a PyCapsule returned by __dlpack__().");
  }
  if (cap.name() == nullptr) {
    throw py::value_error("Capsule has no name");
  }
  absl::string_view name(cap.name());
  if (name == "dltensor_versioned") {
    return DLPackHostBufferBuilder<true>(dlpack_obj, copy)
        .ConstructHostBuffer();
  } else if (name == "dltensor") {
    return DLPackHostBufferBuilder<false>(dlpack_obj, copy)
        .ConstructHostBuffer();
  } else {
    throw py::value_error("Invalid DLPack capsule name");
  }
}

absl::StatusOr<xla::ifrt::DType> DtypeToIfRtDTypeWithStrings(
    py::handle dtype_obj) {
  char kind = 0;
  if (py::hasattr(dtype_obj, "kind")) {
    kind = dtype_obj.attr("kind").cast<char>();
  }
  // Handle string types that xla::DtypeToIfRtDType does not support.
  if (kind == 'O' || kind == 'U' || kind == 'S' || kind == 'T') {
    return xla::ifrt::DType(xla::ifrt::DType::Kind::kString);
  }
  xla::nb_dtype xla_dtype = nanobind::borrow<xla::nb_dtype>(dtype_obj.ptr());
  return xla::DtypeToIfRtDType(xla_dtype);
}

py::object IfrtDTypeToPyObject(xla::ifrt::DType dtype) {
  auto nb_dtype = xla::IfrtDtypeToNbDtype(dtype);
  if (!nb_dtype.ok()) {
    throw py::value_error(nb_dtype.status().ToString());
  }
  return py::reinterpret_borrow<py::object>(nb_dtype->ptr());
}

py::tuple IfrtShapeToPyTuple(const xla::ifrt::Shape& shape) {
  py::tuple py_shape(shape.dims().size());
  for (size_t i = 0; i < shape.dims().size(); ++i) {
    py_shape[i] = shape.dims()[i];
  }
  return py_shape;
}

py::object BuildNestedStringList(const tsl::tstring* ptr,
                                 absl::Span<const int64_t> dims, int dim_idx,
                                 int& element_idx) {
  if (dim_idx == dims.size()) {
    // HostBuffer allocates memory in a contiguous space.
    return py::str(std::string(ptr[element_idx++]));
  }
  int size = dims[dim_idx];
  py::list l(size);
  for (int64_t i = 0; i < size; ++i) {
    l[i] = BuildNestedStringList(ptr, dims, dim_idx + 1, element_idx);
  }
  return l;
}

py::object BuildNestedBytesList(const tsl::tstring* ptr,
                                absl::Span<const int64_t> dims, int dim_idx,
                                int& element_idx) {
  if (dim_idx == dims.size()) {
    const tsl::tstring& s = ptr[element_idx++];
    return py::bytes(s.data(), s.size());
  }
  int size = dims[dim_idx];
  py::list l(size);
  for (int64_t i = 0; i < size; ++i) {
    l[i] = BuildNestedBytesList(ptr, dims, dim_idx + 1, element_idx);
  }
  return l;
}

template <bool IsVersioned>
struct DLPackHostBuffer {
  using MT = std::conditional_t<IsVersioned, DLManagedTensorVersioned,
                                DLManagedTensor>;

  xla_serving::HostBuffer hb;
  std::vector<int64_t> shape;
  std::vector<int64_t> strides;
  MT tensor{};

  static constexpr const char* CapsuleName() {
    if constexpr (IsVersioned) {
      return "dltensor_versioned";
    } else {
      return "dltensor";
    }
  }

  static absl::StatusOr<std::unique_ptr<DLPackHostBuffer<IsVersioned>>> Create(
      const xla_serving::HostBuffer& host_buffer) {
    auto type = IfrtDTypeToDLDataType(host_buffer.dtype());
    if (!type.ok()) {
      return type.status();
    }
    auto pack =
        absl::WrapUnique(new DLPackHostBuffer<IsVersioned>(host_buffer));
    const xla::ifrt::Shape& hb_shape = pack->hb.shape();
    pack->shape.assign(hb_shape.dims().begin(), hb_shape.dims().end());

    // Calculate strides using the default row-major order.
    // xla_serving::HostBuffer doesn't provides strides, so we need to calculate them.
    pack->strides.resize(pack->shape.size());
    int64_t stride = 1;
    for (int i = pack->shape.size() - 1; i >= 0; --i) {
      pack->strides[i] = stride;
      stride *= pack->shape[i];
    }

    // Initialize the DLPack tensor.
    DLTensor& dt = pack->tensor.dl_tensor;
    dt.data = const_cast<void*>(pack->hb.base());
    dt.device = DLDevice{kDLCPU, 0};
    dt.ndim = pack->shape.size();
    dt.dtype = *type;
    dt.shape = pack->shape.data();
    dt.strides = pack->strides.data();
    dt.byte_offset = 0;

    // Set the manager context and deleter for the DLPack tensor.
    if constexpr (IsVersioned) {
      pack->tensor.version =
          DLPackVersion{DLPACK_MAJOR_VERSION, DLPACK_MINOR_VERSION};
      pack->tensor.flags = DLPACK_FLAG_BITMASK_READ_ONLY;
    }
    pack->tensor.manager_ctx = pack.get();
    pack->tensor.deleter = DLPackHostBuffer<IsVersioned>::Deleter;
    return pack;
  }

  static py::capsule CreateCapsule(const xla_serving::HostBuffer& host_buffer) {
    auto pack = Create(host_buffer);
    if (!pack.ok()) {
      throw std::runtime_error(pack.status().ToString());
    }
    return py::capsule(
        &pack->release()->tensor, CapsuleName(), [](PyObject* obj) {
          auto* t =
              reinterpret_cast<MT*>(PyCapsule_GetPointer(obj, CapsuleName()));
          if (t && t->deleter) {
            t->deleter(t);
          } else {
            PyErr_Clear();
          }
        });
  }

 private:
  explicit DLPackHostBuffer(const xla_serving::HostBuffer& host_buffer)
      // Copy the host buffer to avoid lifetime issues.
      : hb(host_buffer) {}

  static void Deleter(MT* dlmt) {
    if (dlmt) {
      delete static_cast<DLPackHostBuffer<IsVersioned>*>(dlmt->manager_ctx);
    }
  }
};

std::string HostBufferRepr(const xla_serving::HostBuffer& hb) {
  return absl::StrCat("<HostBuffer: dtype=", hb.dtype(), " shape=", hb.shape(),
                      ">");
}

std::string ValueRepr(const api::Value& value) { return value.DebugString(); }

PYBIND11_MODULE(composite_processor_pybind, m) {
  m.doc() = "Composite Processor Python bindings";

#if NB_VERSION_MAJOR >= 3
  if (!nanobind::register_module(nanobind::handle(m.ptr()))) {
    throw py::error_already_set();
  }
#endif

  tsl::ImportNumpy();

  try {
    py::google::ImportStatusModule();
  } catch (py::error_already_set& e) {
    if (e.matches(PyExc_ModuleNotFoundError)) {
      // Fallback for isolated bazel tests where package redefinitions might not
      // be accessible due to a hermetic build.
      py::module_::import("pybind11_abseil.status");
    } else {
      throw;
    }
  }
  pybind11_protobuf::ImportNativeProtoCasters();

  // Internal API only. Do not export.
  py::class_<api::TypeInfo>(m, "TypeInfo")
      .def("debug_type_name", &api::TypeInfo::DebugTypeName)
      .def("__repr__",
           [](const api::TypeInfo& self) {
             return absl::StrCat("<TypeInfo: ", self.DebugTypeName(), ">");
           })
      .def("__eq__", [](const api::TypeInfo& self,
                        const api::TypeInfo& other) { return self == other; })
      .def_static("int", []() { return api::CreateTypeInfo<int>(); })
      .def_static("float", []() { return api::CreateTypeInfo<float>(); })
      .def_static("string", []() { return api::CreateTypeInfo<std::string>(); })
      .def_static("host_buffer",
                  []() { return api::CreateTypeInfo<xla_serving::HostBuffer>(); })
      .def_static(
          "host_buffer_vector",
          []() { return api::CreateTypeInfo<std::vector<xla_serving::HostBuffer>>(); })
      .def_static("protobuf_any", []() {
        return api::CreateTypeInfo<google::protobuf::Any>();
      });

  // Internal API only. Do not export.
  py::class_<xla_serving::HostBuffer>(m, "HostBuffer")
      .def(py::init<xla_serving::HostBuffer>())
      .def("get_ref_count", &xla_serving::HostBuffer::ref_count)
      .def_static(
          "create_int64_array",
          [](std::vector<int64_t> vals,
             std::optional<std::vector<int64_t>> shape) {
            if (shape.has_value() &&
                std::accumulate(shape.value().begin(), shape.value().end(), 1,
                                std::multiplies<int64_t>()) != vals.size()) {
              throw py::value_error(
                  "Shape does not match the number of elements.");
            }
            xla::ifrt::Shape ifrt_shape(
                shape.has_value()
                    ? shape.value()
                    : std::vector<int64_t>{static_cast<int64_t>(vals.size())});
            xla_serving::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                               ifrt_shape);
            int64_t* ptr = reinterpret_cast<int64_t*>(hb.base());
            for (int64_t val : vals) {
              *ptr = val;
              ++ptr;
            }
            return hb;
          },
          py::arg("vals"), py::arg("shape") = std::nullopt)
      .def_static(
          "create_string_array",
          [](std::vector<std::string> vals,
             std::optional<std::vector<int64_t>> shape) {
            if (shape.has_value() &&
                std::accumulate(shape.value().begin(), shape.value().end(), 1,
                                std::multiplies<int64_t>()) != vals.size()) {
              throw py::value_error(
                  "Shape does not match the number of elements.");
            }
            xla::ifrt::Shape ifrt_shape(
                shape.has_value()
                    ? shape.value()
                    : std::vector<int64_t>{static_cast<int64_t>(vals.size())});
            xla_serving::HostBuffer hb(
                xla::ifrt::DType(xla::ifrt::DType::Kind::kString), ifrt_shape);
            tsl::tstring* ptr = reinterpret_cast<tsl::tstring*>(hb.base());
            for (std::string& val : vals) {
              *ptr = std::move(val);
              ++ptr;
            }
            return hb;
          },
          py::arg("vals"), py::arg("shape") = std::nullopt)
      .def_static(
          "create_from_bytes_object_ndarray",
          [](py::array arr) -> xla_serving::HostBuffer {
            py::dtype dt = arr.dtype();
            if (dt.kind() != 'O') {
              throw py::value_error("Expected an object array.");
            }
            std::vector<int64_t> shape_vec(arr.shape(),
                                           arr.shape() + arr.ndim());
            xla::ifrt::Shape ifrt_shape(shape_vec);

            auto arr_begin = NumpyFlatIterator::Create(arr, 0);
            if (!arr_begin.ok()) {
              throw py::value_error(std::string(arr_begin.status().message()));
            }
            auto arr_end = NumpyFlatIterator::Create(arr, arr.size());
            if (!arr_end.ok()) {
              throw py::value_error(std::string(arr_end.status().message()));
            }
            auto hb =
                CreateSharedStringHostBuffer(*arr_begin, *arr_end, ifrt_shape);
            if (!hb.ok()) {
              throw py::value_error(absl::StrCat(
                  "Failed to create host buffer: ", hb.status().message()));
            }
            return *hb;
          },
          py::arg("arr"))
      .def_static("create_int64",
                  [](int64_t val) {
                    xla_serving::HostBuffer hb(
                        xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                        xla::ifrt::Shape({}));
                    int64_t* ptr = reinterpret_cast<int64_t*>(hb.base());
                    *ptr = val;
                    return hb;
                  })
      .def_static("create_int32",
                  [](int64_t val) {
                    xla_serving::HostBuffer hb(
                        xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({}));
                    int32_t* ptr = reinterpret_cast<int32_t*>(hb.base());
                    *ptr = val;
                    return hb;
                  })
      .def_static("create_string",
                  [](std::string val) {
                    xla_serving::HostBuffer hb(
                        xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
                        xla::ifrt::Shape({}));
                    tsl::tstring* ptr =
                        reinterpret_cast<tsl::tstring*>(hb.base());
                    *ptr = std::move(val);
                    return hb;
                  })
      .def_static(
          "create_bytes_array",
          [](py::list vals, std::optional<std::vector<int64_t>> shape) {
            if (shape.has_value() &&
                std::accumulate(shape.value().begin(), shape.value().end(), 1,
                                std::multiplies<int64_t>()) != vals.size()) {
              throw py::value_error(
                  "Shape does not match the number of elements.");
            }
            xla::ifrt::Shape ifrt_shape(
                shape.has_value()
                    ? shape.value()
                    : std::vector<int64_t>{static_cast<int64_t>(vals.size())});

            auto hb = CreateSharedStringHostBuffer(vals.begin(), vals.end(),
                                                   ifrt_shape);
            if (!hb.ok()) {
              // Error is only thrown is mismatched shape and vals is passed in,
              // but this should never happen because we construct the shape.
              throw py::value_error(absl::StrCat(
                  "Unexpected error in CreateSharedStringHostBuffer:",
                  hb.status().message()));
            }
            return *hb;
          },
          py::arg("vals"), py::arg("shape") = std::nullopt)
      .def_static(
          "from_dlpack",
          [](py::object obj, std::optional<bool> copy) -> xla_serving::HostBuffer {
            return HostBufferFromDLPack(std::move(obj), copy);
          },
          py::arg("obj"), py::arg("copy") = py::none())
      .def("__repr__", &HostBufferRepr)
      .def("to_int64",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kS64 ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const int64_t* ptr = reinterpret_cast<const int64_t*>(self.base());
             return *ptr;
           })
      .def("to_int32",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kS32 ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const int32_t* ptr = reinterpret_cast<const int32_t*>(self.base());
             return *ptr;
           })
      .def("to_float",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kF32 ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const float* ptr = reinterpret_cast<const float*>(self.base());
             return *ptr;
           })
      .def("to_double",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kF64 ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const double* ptr = reinterpret_cast<const double*>(self.base());
             return *ptr;
           })
      .def("to_string",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kString ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const tsl::tstring* ptr =
                 reinterpret_cast<const tsl::tstring*>(self.base());
             return std::string(*ptr);
           })
      .def("to_string_array",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kString) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const tsl::tstring* ptr =
                 reinterpret_cast<const tsl::tstring*>(self.base());
             int element_idx = 0;
             return BuildNestedStringList(ptr, self.shape().dims(), 0,
                                          element_idx);
           })
      .def("to_bytes",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kString ||
                 self.shape().num_elements() <= 0) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const tsl::tstring* ptr =
                 reinterpret_cast<const tsl::tstring*>(self.base());
             return py::bytes(ptr->data(), ptr->size());
           })
      .def("to_bytes_array",
           [](const xla_serving::HostBuffer& self) {
             if (self.dtype().kind() != xla::ifrt::DType::Kind::kString) {
               throw py::value_error("HostBuffer is not of expected type");
             }
             const tsl::tstring* ptr =
                 reinterpret_cast<const tsl::tstring*>(self.base());
             int element_idx = 0;
             return BuildNestedBytesList(ptr, self.shape().dims(), 0,
                                         element_idx);
           })
      .def_property_readonly("dtype",
                             [](const xla_serving::HostBuffer& self) {
                               return IfrtDTypeToPyObject(self.dtype());
                             })
      .def_property_readonly("shape",
                             [](const xla_serving::HostBuffer& self) {
                               return IfrtShapeToPyTuple(self.shape());
                             })
      .def_property_readonly("storage_dtype",
                             [](const xla_serving::HostBuffer& self) {
                               return IfrtDTypeToPyObject(self.storage_dtype());
                             })
      .def("__dlpack_device__",
           [](const xla_serving::HostBuffer& self) {
             return py::make_tuple(static_cast<int>(kDLCPU), 0);
           })
      .def(
          "__dlpack__",
          [](xla_serving::HostBuffer& self, py::object stream, py::object max_version,
             py::object dl_device, py::object copy) -> py::object {
            if (!copy.is_none() && py::cast<bool>(copy)) {
              throw std::invalid_argument(
                  "__dlpack__ with copy=True is not supported.");
            }

            bool use_versioned = false;
            if (!max_version.is_none() &&
                py::isinstance<py::tuple>(max_version)) {
              py::tuple max_version_tuple = max_version.cast<py::tuple>();
              if (max_version_tuple.size() >= 2) {
                try {
                  int64_t max_version_major =
                      max_version_tuple[0].cast<int64_t>();
                  if (max_version_major >= 1) {
                    use_versioned = true;
                  }
                } catch (const py::cast_error&) {
                  // Ignores non-integer version mapping error
                }
              }
            }

            if (use_versioned) {
              return DLPackHostBuffer<true>::CreateCapsule(self);
            } else {
              return DLPackHostBuffer<false>::CreateCapsule(self);
            }
          },
          py::arg("stream") = py::none(), py::arg("max_version") = py::none(),
          py::arg("dl_device") = py::none(), py::arg("copy") = py::none());

  // Internal API only. Do not export.
  py::class_<api::Value>(m, "Value")
      .def(py::init<>())
      .def(py::init([](int i) { return api::Value(i); }))
      .def(py::init([](float f) { return api::Value(f); }))
      .def(py::init([](std::string s) { return api::Value(s); }))
      .def(py::init([](const xla_serving::HostBuffer& hb) { return api::Value(hb); }))
      .def(py::init([](const std::vector<xla_serving::HostBuffer>& hb_vec) {
        return api::Value(hb_vec);
      }))
      .def(py::init(
          [](const google::protobuf::Any& any) { return api::Value(any); }))
      .def(py::init([](const std::map<std::string, xla_serving::HostBuffer>& d) {
        return api::Value(d);
      }))
      .def("__repr__", &ValueRepr)
      .def("is_empty", [](const api::Value& self) { return !self.HasValue(); })
      .def("type_info", &api::Value::GetTypeInfo)
      .def(
          "as_encapsulated_type",
          [](api::Value& self) -> absl::StatusOr<py::object> {
            if (!self.HasValue()) {
              return py::none();
            }
            using DictType = std::map<std::string, xla_serving::HostBuffer>;
            using CastFunc = py::object (*)(api::Value&);

            static const auto* const kCastMap =
                new absl::flat_hash_map<api::TypeInfo, CastFunc>{
                    {api::CreateTypeInfo<int>(),
                     [](api::Value& v) { return py::cast(v.Get<int>()); }},
                    {api::CreateTypeInfo<float>(),
                     [](api::Value& v) { return py::cast(v.Get<float>()); }},
                    {api::CreateTypeInfo<std::string>(),
                     [](api::Value& v) {
                       return py::cast(v.Get<std::string>());
                     }},
                    {api::CreateTypeInfo<xla_serving::HostBuffer>(),
                     [](api::Value& v) {
                       return py::cast(v.Get<xla_serving::HostBuffer>());
                     }},
                    {api::CreateTypeInfo<std::vector<xla_serving::HostBuffer>>(),
                     [](api::Value& v) {
                       return py::cast(v.Get<std::vector<xla_serving::HostBuffer>>());
                     }},
                    {api::CreateTypeInfo<DictType>(),
                     [](api::Value& v) { return py::cast(v.Get<DictType>()); }},
                    {api::CreateTypeInfo<google::protobuf::Any>(),
                     [](api::Value& v) {
                       return py::cast(v.Get<google::protobuf::Any>());
                     }},
                };

            auto it = kCastMap->find(self.GetTypeInfo());
            if (it != kCastMap->end()) {
              return it->second(self);
            } else {
              return absl::InvalidArgumentError(absl::StrCat(
                  "Unsupported type: ", self.GetTypeInfo().DebugTypeName()));
            }
          });

  // Internal API only. Do not export.
  py::class_<api::ExecuteMetadata>(m, "ExecuteMetadata")
      .def(py::init<>())
      .def_readwrite("model_name", &api::ExecuteMetadata::model_name)
      .def_readwrite("model_version", &api::ExecuteMetadata::model_version)
      .def_readwrite("trace_id", &api::ExecuteMetadata::trace_id);

  py::class_<api::HostBufferSpec>(m, "HostBufferSpec")
      .def(py::init([](py::object dtype_obj, std::vector<int64_t> shape_vec) {
        auto ifrt_dtype = DtypeToIfRtDTypeWithStrings(dtype_obj);
        if (!ifrt_dtype.ok()) {
          throw py::value_error(std::string(ifrt_dtype.status().message()));
        }
        xla::ifrt::Shape shape(shape_vec);
        return api::HostBufferSpec{*ifrt_dtype, shape};
      }))
      .def_property_readonly("dtype",
                             [](const api::HostBufferSpec& self) {
                               return IfrtDTypeToPyObject(self.dtype);
                             })
      .def_property_readonly("shape", [](const api::HostBufferSpec& self) {
        return IfrtShapeToPyTuple(self.shape);
      });

  py::class_<api::ShapeInferenceValue>(m, "ShapeInferenceValue")
      .def(py::init<>())
      .def(py::init([](const api::HostBufferSpec& spec) {
        return api::ShapeInferenceValue(spec);
      }))
      .def(py::init([](const std::vector<api::HostBufferSpec>& specs) {
        return api::ShapeInferenceValue(specs);
      }))
      .def("is_spec", &api::ShapeInferenceValue::IsSpec)
      .def("spec", &api::ShapeInferenceValue::spec,
           py::return_value_policy::reference_internal)
      .def("is_spec_list", &api::ShapeInferenceValue::IsSpecList)
      .def("spec_list", &api::ShapeInferenceValue::spec_list,
           py::return_value_policy::reference_internal);

  // PyBind needs this to mix it in with PyShapeInferenceContext below.
  py::class_<api::ShapeInferenceContext>(m, "_BaseShapeInferenceContext")
      .def(py::init<>())
      .def_readwrite("metadata", &api::ShapeInferenceContext::metadata)
      .def_property_readonly("inputs",
                             [](const api::ShapeInferenceContext& self) {
                               return std::vector<api::ShapeInferenceValue>(
                                   self.inputs.begin(), self.inputs.end());
                             })
      .def_property(
          "outputs",
          [](const api::ShapeInferenceContext& self) {
            return std::vector<api::ShapeInferenceValue>(self.outputs.begin(),
                                                         self.outputs.end());
          },
          [](api::ShapeInferenceContext& self,
             absl::Span<const api::ShapeInferenceValue> val) {
            self.outputs.assign(val.begin(), val.end());
          });

  py::class_<PyShapeInferenceContext, api::ShapeInferenceContext>(
      m, "ShapeInferenceContext")
      .def(py::init<>())
      .def_property(
          "inputs",
          [](const PyShapeInferenceContext& self) {
            return std::vector<api::ShapeInferenceValue>(self.inputs.begin(),
                                                         self.inputs.end());
          },
          [](PyShapeInferenceContext& self,
             std::vector<api::ShapeInferenceValue> inputs) {
            self.inputs_storage = std::move(inputs);
            self.inputs = absl::MakeConstSpan(self.inputs_storage);
          });

  // Internal API only. Do not export.
  py::class_<api::ExecuteContext>(m, "_ExecuteContext")
      .def(py::init<>())
      .def_readwrite("metadata", &api::ExecuteContext::metadata)
      .def_property_readonly("inputs",
                             [](const api::ExecuteContext& self) {
                               return std::vector<api::Value>(
                                   self.inputs.begin(), self.inputs.end());
                             })
      .def_property(
          "outputs",
          [](const api::ExecuteContext& self) {
            return std::vector<api::Value>(self.outputs.begin(),
                                           self.outputs.end());
          },
          [](api::ExecuteContext& self, absl::Span<const api::Value> val) {
            self.outputs.assign(val.begin(), val.end());
          });

  // Internal API only. Do not export.
  py::class_<PyExecuteContext, api::ExecuteContext>(m, "ExecuteContext")
      .def(py::init<>())
      .def_property(
          "inputs",
          [](const PyExecuteContext& self) {
            return std::vector<api::Value>(self.inputs.begin(),
                                           self.inputs.end());
          },
          [](PyExecuteContext& self, std::vector<api::Value> inputs) {
            self.inputs_storage = std::move(inputs);
            self.inputs = absl::MakeConstSpan(self.inputs_storage);
          });

  // Internal API only. Do not export.
  py::classh<TransformBase, PyTransform>(m, "TransformBase")
      .def(py::init<>())
      .def("__repr__", &TransformBase::DebugString)
      .def("execute", &TransformBase::Execute)
      .def("propagate_signature", &TransformBase::PropagateSignature)
      .def("input_types",
           [](const TransformBase& self) {
             auto span = self.InputTypes();
             return std::vector<api::TypeInfo>(span.begin(), span.end());
           })
      .def("output_types",
           [](const TransformBase& self) {
             auto span = self.OutputTypes();
             return std::vector<api::TypeInfo>(span.begin(), span.end());
           })
      .def("input_constraints",
           [](const TransformBase& self) -> py::list {
             absl::StatusOr<const core::TransformFactoryHolder* absl_nonnull>
                 holder = self.GetTransformFactoryHolder();
             if (!holder.ok()) {
               return py::list();
             }
             py::list py_constraints;
             for (const auto& constraint : (*holder)->InputConstraints()) {
               py::list allowed_dtypes;
               for (auto dtype : constraint.allowed_dtypes) {
                 allowed_dtypes.append(
                     IfrtDTypeToPyObject(xla::ifrt::DType(dtype)));
               }
               py::dict py_constraint;
               py_constraint["allowed_dtypes"] = allowed_dtypes;
               py_constraint["is_list"] = constraint.is_list;
               if (constraint.expected_shape.has_value()) {
                 py_constraint["expected_shape"] =
                     py::cast(*constraint.expected_shape);
               }
               py_constraints.append(py_constraint);
             }
             return py_constraints;
           })
      .def("to_proto",
           [](const TransformBase& self) -> py::object {
             // Manual workaround for ImportNativeProtoCasters not generating
             // the automated conversions for the `TransformConfig` proto.
             absl::StatusOr<TransformConfig> config = self.ToProto();
             if (!config.ok()) {
               throw std::runtime_error(std::string(config.status().message()));
             }
             py::object module;
             try {
               module = py::module::import(
                   "xla_serving_data.processor.proto.transform_config_pb2");
             } catch (py::error_already_set& e) {
               try {
                 module = py::module::import(
                     "data.processor.proto."
                     "transform_config_pb2");
               } catch (const py::error_already_set& e2) {
                 throw std::runtime_error(
                     "Failed to import both xla_serving_data and data. Original "
                     "error: " +
                     std::string(e.what()));
               }
             }
             py::object py_config = module.attr("TransformConfig")();
             py_config.attr("ParseFromString")(
                 py::bytes(config->SerializeAsString()));
             return py_config;
           })
      .def("assets",
           [](const TransformBase& self) -> py::dict {
             py::dict assets;
             for (const auto& resource : self.GetAllResources()) {
               for (const auto& asset : resource->assets()) {
                 assets[asset.name.c_str()] = asset.path.c_str();
               }
             }
             return assets;
           })
      .def_static(
          "from_proto",
          [](const py::object& py_config) -> std::unique_ptr<TransformBase> {
            // For some reason ImportNativeProtoCasters is not generating the
            // automated conversions for the `TransformConfig` proto. We
            // bypass this by manually implementing the python proto -> cc
            // proto conversion here.
            TransformConfig config;
            py::bytes bytes = py_config.attr("SerializeToString")();
            std::string serialized = bytes;
            if (!config.ParseFromString(serialized)) {
              throw std::runtime_error("failed to parse config");
            }
            auto result =
                core::TransformRegistry::Instance().DeserializeTransform(
                    config);
            if (!result.ok())
              throw std::runtime_error(std::string(result.status().message()));
            return *std::move(result);
          });

  // Internal API only. Do not export.
  py::class_<core::GraphEdge>(m, "GraphEdge")
      .def_static(
          "graph_input",
          [](int index) {
            auto edge = core::GraphEdge::GraphInput(index);
            if (!edge.ok()) {
              throw py::value_error(std::string(edge.status().message()));
            }
            return *edge;
          })
      .def_static(
          "node_output",
          [](int node_index, int port_index) {
            auto edge = core::GraphEdge::NodeOutput(node_index, port_index);
            if (!edge.ok()) {
              throw py::value_error(std::string(edge.status().message()));
            }
            return *edge;
          },
          py::arg("node_index"), py::arg("port_index") = 0)
      .def_static("constant", &core::GraphEdge::Constant);

  // Internal API only. Do not export.
  py::class_<core::ExecutionGraph>(m, "ExecutionGraph")
      .def(py::init([]() {
        absl::StatusOr<core::ExecutionGraph> graph =
            core::ExecutionGraph::Create({}).value();
        if (!graph.ok()) {
          throw py::value_error(
              absl::StrCat("Failed to create ExecutionGraph due error: ",
                           graph.status().message()));
        }
        return graph.value();
      }))
      .def("add_transform",
           [](core::ExecutionGraph& self, std::string name,
              std::string transform_name) -> absl::Status {
             LOG(WARNING)
                 << "`add_transform` is deprecated. Please use `add_node` "
                    "and `set_outputs` to create the graph instead.";
             ASSIGN_OR_RETURN(
                 auto transform,
                 core::TransformRegistry::Instance().CreateTransform(
                     std::move(transform_name)));
             return self.push_back(std::move(name), std::move(transform));
           })
      .def("add_transform",
           [](core::ExecutionGraph& self, std::string name,
              std::shared_ptr<TransformBase> transform) -> absl::Status {
             LOG(WARNING)
                 << "`add_transform` is deprecated. Please use `add_node` "
                    "and `set_outputs` to create the graph instead.";
             return self.push_back(std::move(name), std::move(transform));
           })
      .def("add_node",
           [](core::ExecutionGraph& self, std::string name,
              std::string transform_name,
              std::vector<core::GraphEdge> input_edges) -> absl::StatusOr<int> {
             ASSIGN_OR_RETURN(
                 auto transform,
                 core::TransformRegistry::Instance().CreateTransform(
                     std::move(transform_name)));
             return self.AddNode(std::move(name), std::move(transform),
                                 std::move(input_edges));
           })
      .def("add_node",
           [](core::ExecutionGraph& self, std::string name,
              std::shared_ptr<TransformBase> transform,
              std::vector<core::GraphEdge> input_edges) -> int {
             return self.AddNode(std::move(name), std::move(transform),
                                 std::move(input_edges));
           })
      .def(
          "set_outputs",
          [](core::ExecutionGraph& self, std::vector<core::GraphEdge> outputs) {
            self.SetOutputs(std::move(outputs));
          })
      .def_property("io_structure", &core::ExecutionGraph::io_structure,
                    &core::ExecutionGraph::SetIOStructure);

  // Internal API only. Do not export.
  py::class_<core::CompositeProcessor>(m, "CompositeProcessor")
      .def_static(
          "from_proto",
          [](const CompositeProcessorConfig& config)
              -> absl::StatusOr<std::unique_ptr<core::CompositeProcessor>> {
            auto result = core::CompositeProcessor::FromProto(config);
            if (!result.ok()) return result.status();
            return std::make_unique<core::CompositeProcessor>(
                *std::move(result));
          })
      .def("to_proto", &core::CompositeProcessor::ToProto)
      .def_static(
          "create",
          [](core::ExecutionGraph& graph)
              -> absl::StatusOr<std::unique_ptr<core::CompositeProcessor>> {
            ASSIGN_OR_RETURN(auto processor, core::CompositeProcessor::Create(
                                                 std::move(graph)));
            return std::make_unique<core::CompositeProcessor>(
                std::move(processor));
          })
      .def("process",
           [](core::CompositeProcessor& self,
              const std::vector<api::Value>& inputs)
               -> absl::StatusOr<py::list> {
             api::ExecuteContext context;
             context.inputs = absl::MakeConstSpan(inputs);
             RETURN_IF_ERROR(self.Process(context));
             // TODO: consider making an opaque type for
             // InlinedVector to further reduce copies. Right now we directly
             // populate a py::list instead of std::vector to avoid
             // vector->list copies done by pybind.
             py::list outputs(context.outputs.size());
             for (size_t i = 0; i < context.outputs.size(); ++i) {
               outputs[i] = std::move(context.outputs[i]);
             }
             return outputs;
           })
      .def("propagate_signature", &core::CompositeProcessor::PropagateSignature)
      .def_property_readonly("graph", &core::CompositeProcessor::graph);
}

}  // namespace
}  // namespace composite
}  // namespace xla_serving::data

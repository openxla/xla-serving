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

#include "runtime/python/numpy_utils.h"  // from @neptune

#include <Python.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <optional>
#include <string>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "include/nanobind/nanobind.h"  // from @nanobind_bazel
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/tensorflow/utils/ifrt_tensor_utils.h"  // from @neptune
#include "numpy/ndarrayobject.h"
#include "numpy/ndarraytypes.h"
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/nb_numpy.h"  // from @xla
#include "xla/python/types.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/python/lib/core/numpy.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tsl/platform/tstring.h"  // from @tsl

namespace nb = ::nanobind;

namespace neptune {
namespace {

struct PyDecrefDeleter {
  void operator()(PyObject* p) const { Py_DECREF(p); }
};

using Safe_PyObjectPtr = std::unique_ptr<PyObject, PyDecrefDeleter>;

// Throws a `nb::value_error` with the given message.
void ThrowIfError(const absl::Status& status, absl::string_view msg) {
  if (status.ok()) return;
  throw nb::value_error(
      absl::StrCat(msg, ". Error: ", status.message()).c_str());
}

// Converts a `tensorflow::DataType` to an `xla::nb_dtype`.
absl::StatusOr<xla::nb_dtype> ToNbDtype(tensorflow::DataType dtype) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType ifrt_dtype, neptune::ToIfrtDType(dtype));
  return xla::IfrtDtypeToNbDtype(ifrt_dtype);
}

// Converts an `xla::nb_dtype` to a `tensorflow::DataType`.
absl::StatusOr<tensorflow::DataType> ToTfDtype(xla::nb_dtype nb_dtype) {
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType ifrt_dtype,
                      xla::DtypeToIfRtDType(nb_dtype));
  return neptune::ToTensorDataType(ifrt_dtype);
}

// Gets a string view from a Python unicode object.
absl::StatusOr<absl::string_view> GetStringViewFromUnicode(PyObject* obj) {
  Py_ssize_t len;
  auto* str = PyUnicode_AsUTF8AndSize(obj, &len);
  if (str == nullptr) {
    return absl::InternalError("PyUnicode_AsUTF8AndSize failed");
  }
  return absl::string_view(str, len);
}

// Gets a string view from a Python bytes object.
absl::StatusOr<absl::string_view> GetStringViewFromBytes(PyObject* obj) {
  Py_ssize_t len;
  char* ptr;
  if (PyBytes_AsStringAndSize(obj, &ptr, &len) < 0) {
    return absl::InternalError("PyBytes_AsStringAndSize failed");
  }
  return absl::string_view(ptr, len);
}

// Gets a string view from a Python object.
absl::StatusOr<absl::string_view> GetStringViewFromPyObject(PyObject* obj) {
  if (PyUnicode_Check(obj)) {
    return GetStringViewFromUnicode(obj);
  } else if (PyBytes_Check(obj)) {
    return GetStringViewFromBytes(obj);
  } else {
    return absl::InternalError("Unsupported Python object type");
  }
}

// Gets elements from a numpy array.
absl::Status GetElementsFromNumpyArray(PyArrayObject* py_array_obj,
                                       neptune::TTypes<tsl::tstring>::Flat flat) {
  auto iter = Safe_PyObjectPtr(
      PyArray_IterNew(reinterpret_cast<PyObject*>(py_array_obj)));

  int idx = 0;
  while (PyArray_ITER_NOTDONE(iter.get())) {
    void* iter_data = PyArray_ITER_DATA(iter.get());
    PyObject* item =
        PyArray_GETITEM(py_array_obj, static_cast<char*>(iter_data));
    if (!item) {
      return absl::InternalError(
          "Failed to get elements out of the ndarray iter.");
    }
    TF_ASSIGN_OR_RETURN(absl::string_view item_str,
                        GetStringViewFromPyObject(item));
    flat(idx) = tsl::tstring(item_str);

    PyArray_ITER_NEXT(iter.get());
    ++idx;
  }
  return absl::OkStatus();
}

// Converts tsl::string to numpy array.
absl::Status ConvertStringToPyArray(PyArrayObject* py_array_obj,
                                    neptune::TTypes<const tsl::tstring>::Flat flat,
                                    int num_elements) {
  nb::object iter =
      nb::steal(PyArray_IterNew(reinterpret_cast<PyObject*>(py_array_obj)));
  for (int i = 0; i < num_elements; ++i) {
    auto item = flat(i);
    absl::string_view bytes = absl::string_view(item.data(), item.size());
    nb::object py_bytes =
        nb::steal(PyBytes_FromStringAndSize(bytes.data(), bytes.size()));

    if (py_bytes.ptr() == nullptr) {
      return absl::InternalError("PyBytes_FromStringAndSize failed");
    }
    if (PyArray_SETITEM(py_array_obj,
                        static_cast<char*>(PyArray_ITER_DATA(iter.ptr())),
                        py_bytes.ptr()) != 0) {
      return absl::InternalError("PyArray_SETITEM failed");
    }

    PyArray_ITER_NEXT(iter.ptr());
  }
  return absl::OkStatus();
}

// Fills a tensor with string type from a `xla::nb_numpy_ndarray`.
absl::Status FillTensorFromStringNumpyArray(
    const xla::nb_numpy_ndarray& numpy_array, tensorflow::Tensor& tensor) {
  if (numpy_array.size() != tensor.NumElements()) {
    return absl::InternalError(
        "The number of elements in the ndarray and tensor do not match.");
  }

  TF_RETURN_IF_ERROR(GetElementsFromNumpyArray(
      reinterpret_cast<PyArrayObject*>(numpy_array.ptr()),
      tensor.flat<tsl::tstring>()));

  return absl::OkStatus();
}

// Fills a `xla::nb_numpy_ndarray` with string type from a `tensorflow::Tensor`.
absl::Status FillStringNumpyArrayFromTensor(
    const tensorflow::Tensor& tensor, xla::nb_numpy_ndarray& numpy_array) {
  int num_elements = tensor.NumElements();
  if (num_elements != numpy_array.size()) {
    return absl::InternalError(
        "The number of elements in the ndarray and tensor do not match.");
  }

  TF_RETURN_IF_ERROR(ConvertStringToPyArray(
      reinterpret_cast<PyArrayObject*>(numpy_array.ptr()),
      tensor.flat<tsl::tstring>(), num_elements));

  return absl::OkStatus();
}

// Fills a HostBuffer with string type from a `xla::nb_numpy_ndarray`.
absl::Status FillHostBufferFromStringNumpyArray(
    const xla::nb_numpy_ndarray& numpy_array, neptune::HostBuffer& host_buffer) {
  if (numpy_array.size() != host_buffer.shape().num_elements()) {
    return absl::InternalError(
        "The number of elements in the ndarray and HostBuffer do not match.");
  }

  TF_RETURN_IF_ERROR(GetElementsFromNumpyArray(
      reinterpret_cast<PyArrayObject*>(numpy_array.ptr()),
      host_buffer.flat<tsl::tstring>()));

  return absl::OkStatus();
}

// Fills a `xla::nb_numpy_ndarray` with string type from a HostBuffer.
absl::Status FillStringNumpyArrayFromHostBuffer(
    const neptune::HostBuffer& host_buffer, xla::nb_numpy_ndarray& numpy_array) {
  int num_elements = host_buffer.shape().num_elements();
  if (num_elements != numpy_array.size()) {
    return absl::InternalError(
        "The number of elements in the ndarray and HostBuffer do not match.");
  }

  TF_RETURN_IF_ERROR(ConvertStringToPyArray(
      reinterpret_cast<PyArrayObject*>(numpy_array.ptr()),
      host_buffer.flat<tsl::tstring>(), num_elements));

  return absl::OkStatus();
}

// Returns true if the xla::nb_dtype is a string NumPy dtype.
bool IsStringDtype(xla::nb_dtype dtype) {
  // See type character codes at
  // https://numpy.org/doc/stable/reference/arrays.interface.html#python-side
  return dtype.kind() == 'S' || dtype.kind() == 'U' || dtype.kind() == 'O';
}

}  // namespace

tensorflow::Tensor ToTFTensor(xla::nb_numpy_ndarray numpy_array) {
  // Get the shape.
  absl::Span<int64_t const> shape(
      reinterpret_cast<const int64_t*>(numpy_array.shape()),
      numpy_array.ndim());

  absl::StatusOr<tensorflow::DataType> tf_dtype =
      ToTfDtype(numpy_array.dtype());
  if (!tf_dtype.ok()) {
    xla::nb_dtype dtype = numpy_array.dtype();
    if (IsStringDtype(dtype)) {
      LOG(INFO) << absl::StrFormat(
          "Treating NumPy dtype as string: char %c kind %c itemsize %d",
          dtype.char_(), dtype.kind(), dtype.itemsize());
      tf_dtype = tensorflow::DT_STRING;
    } else {
      ThrowIfError(
          tf_dtype.status(),
          "Failed to convert `xla::nb_dtype` to `tensorflow::DataType`");
    }
  }
  tensorflow::Tensor tensor(*tf_dtype, tensorflow::TensorShape(shape));

  if (*tf_dtype == tensorflow::DT_STRING) {
    ThrowIfError(FillTensorFromStringNumpyArray(numpy_array, tensor),
                 "Failed to fill tensor from string numpy array");
  } else {
    // Copy the data from the xla::nb_numpy_ndarray to the TensorFlow tensor.
    // nb_numpy_ndarray may not be in a row-major contiguous layout, so we
    // create a contiguous array first to ensure that it is suitable for
    // copying with memcpy.
    xla::nb_numpy_ndarray contiguous_array =
        xla::nb_numpy_ndarray::from_any(numpy_array, NPY_ARRAY_C_CONTIGUOUS);
    std::memcpy(tensor.data(), contiguous_array.data(),
                contiguous_array.size() * contiguous_array.itemsize());
  }

  return tensor;
}

xla::nb_numpy_ndarray ToNbArray(const tensorflow::Tensor& tensor) {
  // Copy the data from the tensor to xla::nb_numpy_ndarray.
  absl::StatusOr<xla::nb_dtype> dtype = ToNbDtype(tensor.dtype());
  ThrowIfError(dtype.status(),
               "Failed to convert `tensorflow::DataType` to `xla::nb_dtype`");
  xla::nb_numpy_ndarray numpy_array(*dtype, tensor.shape().dim_sizes(),
                                    std::nullopt);

  if (tensor.dtype() == tensorflow::DT_STRING) {
    // Special handling for string type.
    ThrowIfError(FillStringNumpyArrayFromTensor(tensor, numpy_array),
                 "Failed to fill string numpy array from tensor");
  } else {
    // Using memcpy is safe here because the tensorflow::Tensor is
    // guaranteed to be row-major and contiguous.
    std::memcpy(numpy_array.mutable_data(), tensor.data(), tensor.TotalBytes());
  }

  return numpy_array;
}

// TODO: Add the Allocator parameter to the applicable callers.
neptune::HostBuffer ToHostBuffer(xla::nb_numpy_ndarray numpy_array,
                             neptune::Allocator* allocator) {
  // Get the shape.
  absl::Span<int64_t const> shape(
      reinterpret_cast<const int64_t*>(numpy_array.shape()),
      numpy_array.ndim());

  absl::StatusOr<xla::ifrt::DType> ifrt_dtype =
      xla::DtypeToIfRtDType(numpy_array.dtype());
  if (!ifrt_dtype.ok()) {
    xla::nb_dtype dtype = numpy_array.dtype();
    if (IsStringDtype(dtype)) {
      LOG(INFO) << absl::StrFormat(
          "Treating NumPy dtype as string: char %c kind %c itemsize %d",
          dtype.char_(), dtype.kind(), dtype.itemsize());
      ifrt_dtype = xla::ifrt::DType(xla::ifrt::DType::kString);
    } else {
      ThrowIfError(ifrt_dtype.status(),
                   "Failed to convert `xla::nb_dtype` to `xla::ifrt::DType`");
    }
  }
  neptune::HostBuffer host_buffer(*ifrt_dtype, xla::ifrt::Shape(shape), allocator);

  if (ifrt_dtype->kind() == xla::ifrt::DType::kString) {
    ThrowIfError(FillHostBufferFromStringNumpyArray(numpy_array, host_buffer),
                 "Failed to fill host buffer from string numpy array");
  } else {
    // Copy the data from the xla::nb_numpy_ndarray to the HostBuffer.
    // nb_numpy_ndarray may not be in a row-major contiguous layout, so we
    // create a contiguous array first to ensure that it is suitable for
    // copying with memcpy.
    xla::nb_numpy_ndarray contiguous_array =
        xla::nb_numpy_ndarray::from_any(numpy_array, NPY_ARRAY_C_CONTIGUOUS);
    std::memcpy(host_buffer.base(), contiguous_array.data(),
                contiguous_array.size() * contiguous_array.itemsize());
  }

  return host_buffer;
}

xla::nb_numpy_ndarray ToNbArray(const neptune::HostBuffer& host_buffer) {
  // Copy the data from the HostBuffer to xla::nb_numpy_ndarray.
  absl::StatusOr<xla::nb_dtype> dtype =
      xla::IfrtDtypeToNbDtype(host_buffer.dtype());
  ThrowIfError(dtype.status(),
               "Failed to convert `xla::ifrt::DType` to `xla::nb_dtype`");
  if (host_buffer.dtype().kind() == xla::ifrt::DType::kString) {
    xla::nb_numpy_ndarray numpy_array(*dtype, host_buffer.shape().dims(),
                                      std::nullopt);
    ThrowIfError(FillStringNumpyArrayFromHostBuffer(host_buffer, numpy_array),
                 "Failed to fill string numpy array from HostBuffer");
    return numpy_array;
  } else {
    std::optional<absl::Span<int64_t const>> strides = std::nullopt;
    if (host_buffer.byte_strides().has_value()) {
      strides = host_buffer.byte_strides().value();
    }
    xla::nb_numpy_ndarray view(*dtype, host_buffer.shape().dims(), strides,
                                host_buffer.base());
    return xla::nb_numpy_ndarray::from_any(view, NPY_ARRAY_C_CONTIGUOUS);
  }
}

}  // namespace neptune

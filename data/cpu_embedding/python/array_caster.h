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

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_PYTHON_ARRAY_CASTER_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_PYTHON_ARRAY_CASTER_H_

#include <any>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "include/pybind11/numpy.h"  // from @pybind11_bazel
#include "include/pybind11/pybind11.h"  // from @pybind11_bazel

namespace py = pybind11;

namespace neptune::data::cpu_embedding {

// Wraps a py::array object to ensure GIL-safe destruction.
struct NdarrayOwner {
  std::optional<py::array> array;

  NdarrayOwner() = default;
  explicit NdarrayOwner(py::array a) : array(std::move(a)) {}

  NdarrayOwner(NdarrayOwner&& other) noexcept : array(std::move(other.array)) {}
  NdarrayOwner& operator=(NdarrayOwner&& other) noexcept {
    array = std::move(other.array);
    return *this;
  }

  NdarrayOwner(const NdarrayOwner& other) {
    py::gil_scoped_acquire acquire;
    array = other.array;
  }

  NdarrayOwner& operator=(const NdarrayOwner& other) {
    if (this != &other) {
      py::gil_scoped_acquire acquire;
      array = other.array;
    }
    return *this;
  }

  ~NdarrayOwner() {
    if (array.has_value()) {
      py::gil_scoped_acquire acquire;
      array.reset();
    }
  }
};

inline Array ArrayFromNumpy(py::handle src) {
  py::array np_array = py::array::ensure(src, py::array::c_style);
  if (!np_array) {
    throw py::value_error("Incompatible array format for cpu_embedding::Array");
  }

  py::buffer_info buf = np_array.request();
  cpu_embedding::DType dtype;
  if (buf.format == py::format_descriptor<float>::format()) {
    dtype = cpu_embedding::DType::kFloat32;
  } else if (buf.format == py::format_descriptor<int32_t>::format()) {
    dtype = cpu_embedding::DType::kInt32;
  } else if (buf.format == py::format_descriptor<int64_t>::format()) {
    dtype = cpu_embedding::DType::kInt64;
  } else {
    throw py::value_error("Unsupported numpy dtype for Array");
  }
  std::vector<int64_t> dims(buf.shape.begin(), buf.shape.end());
  return cpu_embedding::Array::Wrap(
      buf.ptr, cpu_embedding::Shape(std::move(dims)), dtype,
      std::make_any<NdarrayOwner>(NdarrayOwner(std::move(np_array))));
}

inline py::array ArrayToNumpy(const Array& self) {
  std::string format;
  size_t itemsize;
  switch (self.GetDType()) {
    case cpu_embedding::DType::kFloat32:
      format = py::format_descriptor<float>::format();
      itemsize = sizeof(float);
      break;
    case cpu_embedding::DType::kInt32:
      format = py::format_descriptor<int32_t>::format();
      itemsize = sizeof(int32_t);
      break;
    case cpu_embedding::DType::kInt64:
      format = py::format_descriptor<int64_t>::format();
      itemsize = sizeof(int64_t);
      break;
    default:
      throw py::value_error("Unsupported DType for ArrayToNumpy()");
  }

  std::vector<int64_t> strides(self.GetShape().dims.size());
  if (!strides.empty()) {
    strides.back() = itemsize;
    for (int i = strides.size() - 2; i >= 0; --i) {
      strides[i] = strides[i + 1] * self.GetShape().dims[i + 1];
    }
  }

  std::vector<ssize_t> py_shape(self.GetShape().dims.begin(),
                                self.GetShape().dims.end());
  std::vector<ssize_t> py_strides(strides.begin(), strides.end());

  py::capsule base_handle(new cpu_embedding::Array(self), [](void* ptr) {
    delete static_cast<cpu_embedding::Array*>(ptr);
  });

  return py::array(
      py::buffer_info(const_cast<void*>(self.Data()), itemsize, format,
                      py_shape.size(), py_shape, py_strides),
      base_handle);
}

}  // namespace neptune::data::cpu_embedding

namespace pybind11::detail {

template <>
struct type_caster<::neptune::data::cpu_embedding::Array> {
 public:
  PYBIND11_TYPE_CASTER(
      ::neptune::data::cpu_embedding::Array,
      const_name("Union[numpy.ndarray, jax_cpu_embedding.Array]"));

  bool load(handle src, bool convert) {
    // 1. Try to load from an existing cpu_embedding::Array instance.
    type_caster_base<::neptune::data::cpu_embedding::Array> base_caster;
    if (base_caster.load(src, convert)) {
      value = std::move(
          static_cast<::neptune::data::cpu_embedding::Array&>(base_caster));
      return true;
    }

    // 2. Try to load from NumPy.
    py::array np_array = py::array::ensure(src, py::array::c_style);
    if (np_array) {
      value = ::neptune::data::cpu_embedding::ArrayFromNumpy(np_array);
      return true;
    }

    return false;
  }

  static handle cast(const ::neptune::data::cpu_embedding::Array& src,
                     return_value_policy policy, handle parent) {
    return ::neptune::data::cpu_embedding::ArrayToNumpy(src).release();
  }
};

}  // namespace pybind11::detail

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_PYTHON_ARRAY_CASTER_H_

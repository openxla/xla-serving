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
#include <new>
#include <optional>
#include <stdexcept>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "include/nanobind/nanobind.h"  // from @nanobind_bazel
#include "include/nanobind/nb_defs.h"  // from @nanobind_bazel
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/python/numpy_utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/nb_numpy.h"  // from @xla
#include "xla/tsl/python/lib/core/numpy.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tensorflow/core/platform/tstring.h"  // from @org_tensorflow

namespace nb = ::nanobind;

namespace {

// Adds one to each element in the tensor.
xla::nb_numpy_ndarray TensorAddOne(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = xla_serving::ToTFTensor(array);
  auto flat = tensor.flat<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) += 1.0;
  }
  return xla_serving::ToNbArray(tensor);
}

// Multiplies each element in the host buffer by two.
xla::nb_numpy_ndarray HostBufferTimesTwo(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  xla_serving::HostBuffer host_buffer = xla_serving::ToHostBuffer(array);
  auto flat = host_buffer.flat<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) *= 2.0;
  }
  return xla_serving::ToNbArray(host_buffer);
}

// Converts a numpy array to a tensor and back.
xla::nb_numpy_ndarray TensorRoundTrip(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = xla_serving::ToTFTensor(array);
  return xla_serving::ToNbArray(tensor);
}

// String tensor to int tensor.
xla::nb_numpy_ndarray StringArrayGetLength(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = xla_serving::ToTFTensor(array);
  tensorflow::Tensor int_tensor(tensorflow::DT_INT32, tensor.shape());
  auto flat = int_tensor.flat<int32_t>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) =
        static_cast<int32_t>(tensor.flat<tensorflow::tstring>()(i).size());
  }
  return xla_serving::ToNbArray(int_tensor);
}

// Wraps linear data in a Column-Major HostBuffer and converts it to NbArray.
xla::nb_numpy_ndarray ColMajorHostBufferToNbArray(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  if (array.size() != 6) {
    throw std::invalid_argument("Expected array of size 6");
  }
  void* data = const_cast<void*>(array.data());
  xla::ifrt::DType dtype(xla::ifrt::DType::Kind::kF32);
  xla::ifrt::Shape shape({2, 3});
  // Column-Major strides for (2, 3) float32 array:
  // Dim 0 stride: 4 bytes
  // Dim 1 stride: 8 bytes
  std::optional<absl::InlinedVector<int64_t, 4>> strides =
      absl::InlinedVector<int64_t, 4>{4, 8};
  xla_serving::HostBuffer host_buffer(data, dtype, shape, strides, nullptr);
  return xla_serving::ToNbArray(host_buffer);
}

// Convert numpy array to HostBuffer with tracking allocator, check for
// overwrites, and convert back.
xla::nb_numpy_ndarray HostBufferRoundTripWithSentinelCheck(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);

  struct TrackingAllocator : public xla_serving::Allocator {
    void* ptr = nullptr;
    size_t size = 0;
    bool overwritten = false;

    void* Allocate(size_t sz, size_t alignment) override {
      size = sz;
      ptr = ::operator new[](sz + 8, std::align_val_t(alignment));
      std::memset(static_cast<char*>(ptr) + sz, 0x5a, 8);
      return ptr;
    }

    void Free(void* p, size_t alignment) override {
      if (p && p == ptr) {
        const char* sentinel = static_cast<const char*>(ptr) + size;
        for (int i = 0; i < 8; ++i) {
          if (static_cast<uint8_t>(sentinel[i]) != 0x5a) {
            overwritten = true;
          }
        }
      }
      ::operator delete[](p, std::align_val_t(alignment));
    }
  };

  TrackingAllocator allocator;
  xla::nb_numpy_ndarray result;
  {
    xla_serving::HostBuffer host_buffer = xla_serving::ToHostBuffer(array, &allocator);
    result = xla_serving::ToNbArray(host_buffer);
  }

  if (allocator.overwritten) {
    throw std::runtime_error(
        "Sentinel bytes overwritten! Memory corruption detected.");
  }

  return result;
}

}  // namespace

NB_MODULE(numpy_utils_test_utils, m) {
  m.def("tensor_add_one", &TensorAddOne);
  m.def("host_buffer_times_two", &HostBufferTimesTwo);
  m.def("tensor_round_trip", &TensorRoundTrip);
  m.def("string_array_get_length", &StringArrayGetLength);
  m.def("col_major_host_buffer_to_nb_array", &ColMajorHostBufferToNbArray);
  m.def("host_buffer_round_trip_with_sentinel_check",
        &HostBufferRoundTripWithSentinelCheck);
}

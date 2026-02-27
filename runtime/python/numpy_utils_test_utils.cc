#include <cstdint>

#include "third_party/nanobind/include/nanobind/nanobind.h"
#include "third_party/nanobind/include/nanobind/nb_defs.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/neptune/runtime/python/numpy_utils.h"
#include "third_party/tensorflow/compiler/xla/python/nb_numpy.h"
#include "third_party/tensorflow/compiler/xla/tsl/python/lib/core/numpy.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor.proto.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/types.proto.h"
#include "third_party/tensorflow/core/platform/tstring.h"

namespace nb = ::nanobind;

namespace {

// Adds one to each element in the tensor.
xla::nb_numpy_ndarray TensorAddOne(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = neptune::ToTFTensor(array);
  auto flat = tensor.flat<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) += 1.0;
  }
  return neptune::ToNbArray(tensor);
}

// Multiplies each element in the host buffer by two.
xla::nb_numpy_ndarray HostBufferTimesTwo(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  jsv::HostBuffer host_buffer = neptune::ToHostBuffer(array);
  auto flat = host_buffer.flat<float>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) *= 2.0;
  }
  return neptune::ToNbArray(host_buffer);
}

// Converts a numpy array to a tensor and back.
xla::nb_numpy_ndarray TensorRoundTrip(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = neptune::ToTFTensor(array);
  return neptune::ToNbArray(tensor);
}

// String tensor to int tensor.
xla::nb_numpy_ndarray StringArrayGetLength(nb::handle arg) {
  auto array = xla::nb_numpy_ndarray::ensure(arg);
  tensorflow::Tensor tensor = neptune::ToTFTensor(array);
  tensorflow::Tensor int_tensor(tensorflow::DT_INT32, tensor.shape());
  auto flat = int_tensor.flat<int32_t>();
  for (int i = 0; i < flat.size(); ++i) {
    flat(i) =
        static_cast<int32_t>(tensor.flat<tensorflow::tstring>()(i).size());
  }
  return neptune::ToNbArray(int_tensor);
}

}  // namespace

NB_MODULE(numpy_utils_test_utils, m) {
  m.def("tensor_add_one", &TensorAddOne);
  m.def("host_buffer_times_two", &HostBufferTimesTwo);
  m.def("tensor_round_trip", &TensorRoundTrip);
  m.def("string_array_get_length", &StringArrayGetLength);
}

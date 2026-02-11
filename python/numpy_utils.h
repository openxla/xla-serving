#ifndef THIRD_PARTY_NEPTUNE_PYTHON_NUMPY_UTILS_H_
#define THIRD_PARTY_NEPTUNE_PYTHON_NUMPY_UTILS_H_

#include "learning/infra/mira/experimental/orbax_model/core/allocator.h"
#include "learning/infra/mira/experimental/orbax_model/core/default_allocator.h"
#include "learning/infra/mira/experimental/orbax_model/core/host_buffer.h"
#include "third_party/tensorflow/compiler/xla/python/nb_numpy.h"
#include "third_party/tensorflow/compiler/xla/tsl/python/lib/core/numpy.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorflow/core/framework/tensor.h"
#include "third_party/tensorflow/core/framework/tensor.proto.h"
#include "third_party/tensorflow/core/framework/tensor_shape.h"
#include "third_party/tensorflow/core/framework/types.proto.h"

namespace neptune {

// Converts an `xla::nb_numpy_ndarray` to a `tensorflow::Tensor`.
tensorflow::Tensor ToTFTensor(xla::nb_numpy_ndarray numpy_array);

// Converts a `tensorflow::Tensor` to a `xla::nb_numpy_ndarray`. When `tensor`
// is of string type, the returned numpy array contains bytes and will have
// dtype `np.object_`.
xla::nb_numpy_ndarray ToNbArray(const tensorflow::Tensor& tensor);

// Converts an `xla::nb_numpy_ndarray` to a `jsv::HostBuffer`. The `HostBuffer`
// is allocated using the provided `allocator`. The DefaultAllocator is used if
// `allocator` is not provided.
jsv::HostBuffer ToHostBuffer(
    xla::nb_numpy_ndarray numpy_array,
    jsv::Allocator* allocator = jsv::DefaultAllocator::GetInstance());

// Converts a `jsv::HostBuffer` to a `xla::nb_numpy_ndarray`.
xla::nb_numpy_ndarray ToNbArray(const jsv::HostBuffer& host_buffer);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_PYTHON_NUMPY_UTILS_H_

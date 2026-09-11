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

#ifndef XLA_SERVING_RUNTIME_PYTHON_NUMPY_UTILS_H_
#define XLA_SERVING_RUNTIME_PYTHON_NUMPY_UTILS_H_

#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/nb_numpy.h"  // from @xla
#include "xla/tsl/python/lib/core/numpy.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow

namespace xla_serving {

// Converts an `xla::nb_numpy_ndarray` to a `tensorflow::Tensor`.
tensorflow::Tensor ToTFTensor(xla::nb_numpy_ndarray numpy_array);

// Converts a `tensorflow::Tensor` to a `xla::nb_numpy_ndarray`. When `tensor`
// is of string type, the returned numpy array contains bytes and will have
// dtype `np.object_`.
xla::nb_numpy_ndarray ToNbArray(const tensorflow::Tensor& tensor);

// Converts an `xla::nb_numpy_ndarray` to a `xla_serving::HostBuffer`. The `HostBuffer`
// is allocated using the provided `allocator`. The DefaultAllocator is used if
// `allocator` is not provided.
xla_serving::HostBuffer ToHostBuffer(
    xla::nb_numpy_ndarray numpy_array,
    xla_serving::Allocator* allocator = xla_serving::DefaultAllocator::GetInstance());

// Converts a `xla_serving::HostBuffer` to a `xla::nb_numpy_ndarray`.
xla::nb_numpy_ndarray ToNbArray(const xla_serving::HostBuffer& host_buffer);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_PYTHON_NUMPY_UTILS_H_

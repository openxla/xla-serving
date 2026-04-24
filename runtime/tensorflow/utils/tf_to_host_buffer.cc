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

#include "runtime/tensorflow/utils/tf_to_host_buffer.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "runtime/tensorflow/utils/ifrt_tensor_utils.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorflow/core/framework/allocation_description.pb.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor.h"  // from @org_tensorflow
#include "tensorflow/core/framework/tensor_shape.h"  // from @org_tensorflow
#include "tensorflow/core/framework/types.pb.h"  // from @org_tensorflow
#include "tsl/platform/refcount.h"  // from @tsl

namespace neptune {

namespace {
// A TensorBuffer that manages the lifetime of a host array.
// This is used for HostArray to TFTensor conversion.
// HostArray may or may not own the underlying data buffer; so we simply
// manage the lifetime of the HostArray itself.
//
class HostBufferTensorBuffer : public tensorflow::TensorBuffer {
 public:
  static absl::StatusOr<tsl::core::RefCountPtr<TensorBuffer>> Create(
      HostBuffer host_buffer) {
    if (host_buffer.dtype().kind() != xla::ifrt::DType::Kind::kString) {
      if (!host_buffer.storage_dtype().byte_size().has_value()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Unsupported Storage dtype ", host_buffer.storage_dtype()));
      }
    }

    return tsl::core::RefCountPtr<tensorflow::TensorBuffer>(
        new HostBufferTensorBuffer(std::move(host_buffer)));
  }

  tensorflow::TensorBuffer* root_buffer() override { return this; }

  size_t size() const override { return size_; }
  bool OwnsMemory() const override { return false; }

  void FillAllocationDescription(
      tensorflow::AllocationDescription* proto) const override {
    proto->set_requested_bytes(size_);
  }

 private:
  explicit HostBufferTensorBuffer(HostBuffer host_buffer)
      : tensorflow::TensorBuffer(host_buffer.base()),
        host_buffer_(std::move(host_buffer)) {
    size_ = host_buffer_.total_storage_bytes();
  }

  size_t size_;
  HostBuffer host_buffer_;
};
}  // namespace

absl::StatusOr<HostBuffer> ToHostBuffer(tensorflow::Tensor tensor) {
  VLOG(2) << "ConvertTFTensorToHostBuffer: tensor shape: " << tensor.shape()
          << ", dtype: " << tensor.dtype();

  absl::InlinedVector<int64_t, 4UL> dims_vector = tensor.shape().dim_sizes();
  absl::Span<int64_t const> dims(dims_vector);
  TF_ASSIGN_OR_RETURN(xla::ifrt::DType dtype, ToIfrtDType(tensor.dtype()));

  // Get raw data pointer first to avoid use after move of tensor.
  void* raw_data = tensor.data();
  return HostBuffer(
      raw_data, dtype, xla::ifrt::Shape(dims),
      /*owner=*/std::make_shared<tensorflow::Tensor>(std::move(tensor)));
}

absl::StatusOr<tensorflow::Tensor> ToTFTensor(const HostBuffer& host_buffer) {
  VLOG(2) << "ConvertHostBufferToTFTensor: host buffer shape: "
          << host_buffer.shape() << ", dtype: " << host_buffer.dtype();
  TF_ASSIGN_OR_RETURN(
      tsl::core::RefCountPtr<tensorflow::TensorBuffer> tensor_buffer,
      HostBufferTensorBuffer::Create(host_buffer));
  TF_ASSIGN_OR_RETURN(auto data_type, ToTensorDataType(host_buffer.dtype()));
  return tensorflow::Tensor(data_type,
                            tensorflow::TensorShape(host_buffer.shape().dims()),
                            std::move(tensor_buffer));
}

absl::StatusOr<Value> ToJsvValue(tensorflow::Tensor tensor,
                                 const KernelIODataType data_type) {
  switch (data_type) {
    case KernelIODataType::kHostBuffer: {
      TF_ASSIGN_OR_RETURN(HostBuffer host_buffer,
                          ToHostBuffer(std::move(tensor)));
      return Value(std::move(host_buffer));
    }
    case KernelIODataType::kTensor:
      return Value(std::move(tensor));
    default:
      return absl::UnimplementedError(
          absl::StrCat("Unsupported data type: ", data_type));
  }
}

}  // namespace neptune

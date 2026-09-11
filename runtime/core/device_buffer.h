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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_

#include <memory>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/array_spec.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {

// Physical memory placement kind for on-device and zero-copy host buffers.
enum class BufferMemoryKind { kDevice = 0, kPinnedHost = 1 };

// A metadata-rich container representing an array that already resides in
// device memory. This struct bundles the physical `xla::ifrt::ArrayRef` with
// the execution context (client, device list) and the `xla::ifrt::ShardingRef`
// layout. Downstream operators require this metadata to accurately orchestrate
// Device-to-Host (D2H) transfers (e.g., via MakeHostBufferFromArray).
// (Do not add constructors or destructors to this struct. We want it to remain
// a simple aggregate so we can use designated initializers).
struct DeviceBuffer {
  // Shared context used for underlying IFRT operations (e.g. transfers).
  std::shared_ptr<xla::ifrt::Client> ifrt_client;
  // The physical on-device array payload.
  xla::ifrt::ArrayRef ifrt_array;
  // The IFRT sharding (physical/logical layout) of the array.
  xla::ifrt::ShardingRef sharding;
  // The specific devices across which the array is distributed.
  xla::ifrt::DeviceListRef device_list;
  // Physical memory placement kind.
  BufferMemoryKind memory_kind = BufferMemoryKind::kDevice;
  // Shared ownership handle to prevent premature deallocation of backing host
  // memory.
  std::shared_ptr<void> owner = nullptr;

  absl::StatusOr<DeviceBuffer> Reshape(const xla::ifrt::Shape& shape) &&;
  absl::StatusOr<DeviceBuffer> Bitcast(xla::ifrt::DType dtype,
                                      const xla::ifrt::Shape& shape) &&;
};

inline absl::StatusOr<DeviceBuffer> DeviceBuffer::Reshape(
    const xla::ifrt::Shape& shape) && {
  if (ifrt_client == nullptr || ifrt_array == nullptr) {
    return absl::FailedPreconditionError("DeviceBuffer is uninitialized.");
  }
  return std::move(*this).Bitcast(ifrt_array->dtype(), shape);
}

inline absl::StatusOr<DeviceBuffer> DeviceBuffer::Bitcast(
    xla::ifrt::DType dtype, const xla::ifrt::Shape& shape) && {
  if (ifrt_client == nullptr || ifrt_array == nullptr) {
    return absl::FailedPreconditionError("DeviceBuffer is uninitialized.");
  }
  xla::ifrt::ShardingRef target_sharding =
      sharding != nullptr ? sharding : ifrt_array->shared_ptr_sharding();
  xla::ifrt::ArraySpec new_spec = {
      .dtype = dtype,
      .shape = shape,
      .sharding = target_sharding,
  };
  xla::ifrt::ArrayRef array_ref = std::move(ifrt_array);
  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::ArrayRef> new_arrays,
      ifrt_client->BitcastArrays(
          absl::MakeSpan(&array_ref, 1),
          absl::MakeSpan(&new_spec, 1),
          xla::ifrt::ArrayCopySemantics::kDonateInput));
  xla::ifrt::ShardingRef result_sharding =
      new_arrays[0]->shared_ptr_sharding();
  return DeviceBuffer{
      .ifrt_client = std::move(ifrt_client),
      .ifrt_array = std::move(new_arrays[0]),
      .sharding = std::move(result_sharding),
      .device_list = std::move(device_list),
      .memory_kind = memory_kind,
      .owner = std::move(owner),
  };
}

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_

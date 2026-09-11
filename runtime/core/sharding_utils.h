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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_SHARDING_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_SHARDING_UTILS_H_

#include <cstdint>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/device_buffer.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving::sharding_internal {

// Splits the given `input_buffer` into equal shapes of slices.
//
// `num_partitions_per_axis` specifies the number of partitions along
// each axis (dimension).
absl::StatusOr<std::vector<HostBuffer>> DisassembleHostBuffer(
    const HostBuffer& input_buffer,
    const std::vector<int32_t>& num_partitions_per_axis,
    const tsl::thread::ThreadPool& thread_pool);

// Reassembles split HostBuffers back into one HostBuffer.
//
// `num_concats` specifies the number of split buffers along
// each axis (dimension).
//
// `input_disassembled_buffers` contains a list of buffers flattened into the
// minor-to-major order.
//
absl::StatusOr<HostBuffer> MakeHostBufferFromDisassembledBuffers(
    absl::Span<const HostBuffer> input_disassembled_buffers,
    const std::vector<int>& num_concats, xla::ifrt::DType output_buffer_type,
    const xla::ifrt::Shape& output_buffer_shape,
    const tsl::thread::ThreadPool& thread_pool);

// Creates an ifrt::Array from the given host buffer based on given device ids
// and sharding information.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_host_buffer,
    absl::Span<const int> device_ids, xla::ifrt::ShardingRef sharding,
    const tsl::thread::ThreadPool& thread_pool,
    xla::ifrt::Client::HostBufferSemantics semantics = xla::ifrt::Client::
        HostBufferSemantics::kImmutableUntilTransferCompletes);

// A variant of the above api. The difference is that the user passes in
// device_list directly instead of a list of device_ids.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_host_buffer,
    const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef sharding, const tsl::thread::ThreadPool& thread_pool,
    xla::ifrt::Client::HostBufferSemantics semantics = xla::ifrt::Client::
        HostBufferSemantics::kImmutableUntilTransferCompletes);

// Reshards a disassembled array list back to one single tensor based on given
// sharding spec.
//
// input_array: the input device buffers.
//
// hlo_sharding: sharding spec that describes how the input device buffers are
// sharded.
//
// device_list: list of devices that is aligned with the order of device
// buffers in the `input_array`.
//
tsl::Future<HostBuffer> MakeHostBufferFromArray(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::ifrt::ShardingRef& ifrt_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool);

// Creates an `xla::ifrt::DeviceList` from a list of device ids.
absl::StatusOr<xla::ifrt::DeviceListRef> CreateDeviceList(
    xla::ifrt::Client& ifrt_client, absl::Span<const int> device_ids);

// Converts `hlo_sharding` to `xla::ifrt::Sharding`.
absl::StatusOr<xla::ifrt::ShardingRef> ToIfrtSharding(
    xla::ifrt::Client& ifrt_client, const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list);

// Creates a DeviceBuffer bound to pinned host memory with zero-copy semantics.
absl::StatusOr<DeviceBuffer> MakeDeviceBufferFromPinnedHost(
    const std::shared_ptr<xla::ifrt::Client>& client,
    const HostBuffer& host_buffer, const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef sharding,
    const tsl::thread::ThreadPool& thread_pool);

}  // namespace xla_serving::sharding_internal

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_SHARDING_UTILS_H_

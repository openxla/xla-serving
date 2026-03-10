#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_SHARDING_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_SHARDING_UTILS_H_

#include <cstdint>
#include <optional>
#include <vector>

#include "third_party/absl/container/inlined_vector.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/host_buffer.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device_list.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/dtype.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "third_party/tensorflow/compiler/xla/shape.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/ref_count.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv::sharding_internal {

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
    const tsl::thread::ThreadPool& thread_pool);

// A variant of the above api. The difference is that the user passes in
// device_list directly instead of a list of device_ids.
absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_host_buffer,
    const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef sharding,
    const tsl::thread::ThreadPool& thread_pool);


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
    const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool);

// Creates an `xla::ifrt::DeviceList` from a list of device ids.
absl::StatusOr<xla::ifrt::DeviceListRef> CreateDeviceList(
    xla::ifrt::Client& ifrt_client, absl::Span<const int> device_ids);

}  // namespace jsv::sharding_internal

#endif  //  THIRD_PARTY_NEPTUNE_RUNTIME_SHARDING_UTILS_H_

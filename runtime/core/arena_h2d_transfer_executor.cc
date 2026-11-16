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

#include "runtime/core/arena_h2d_transfer_executor.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/device_event.h"  // from @xla
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_buffer.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_device.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

namespace {

constexpr int64_t AlignUp(int64_t value, size_t alignment) {
  const int64_t align = static_cast<int64_t>(alignment);
  return (value + align - 1) & ~(align - 1);
}

// Per-tensor input to `TransferDeviceArena`, pairing the host source view with
// its resolved on-device shape and byte size.
struct DeviceArenaEntry {
  // Canonical on-device shape with memory-space default layout populated.
  xla::Shape on_device_shape;
  // Size in bytes of `on_device_shape` in device memory (queried via
  // `CommonPjRtClient::GetOnDeviceBytesCount`).
  int64_t on_device_bytes;
  // Unpacked byte size of the tensor on host (`ShapeUtil::ByteSizeOf`).
  int64_t unpacked_host_bytes;
  // Pointer to the start of the tensor's data in `source_buffer`.
  const void* data;
  // Byte strides of `source_buffer` along each dimension.
  absl::InlinedVector<int64_t, 4> byte_strides;
  // Keeps the backing host memory alive during async zero-copy H2D transfers.
  HostBuffer source_buffer;
};

// Checks sharding-agnostic `HostBuffer` and `MemoryKind` properties to
// determine whether `input` can use arena-backed H2D transfer instead of
// falling back to `MakeArrayFromHostBuffer`.
bool IsArenaEligibleInput(const HostBuffer& input,
                          xla::ifrt::MemoryKind memory_kind) {
  // CommonPjRtClient::Linearize and raw buffer H2D DMA only support fixed-size
  // POD element types, not tsl::tstring objects.
  if (input.dtype().kind() == xla::ifrt::DType::kString) {
    return false;
  }

  // Sub-byte types (e.g. int4) are stored unpacked as kU8 in HostBuffer
  // (dtype != storage_dtype), whereas CommonPjRtClient::Linearize expects the
  // input buffer element width to match the on-device primitive type.
  if (input.dtype() != input.storage_dtype()) {
    return false;
  }

  // Zero-element tensors have 0 on-device bytes and require no H2D DMA;
  // AllocateHostArena also requires size > 0.
  if (input.shape().num_elements() == 0) {
    return false;
  }

  // TODO: ArenaH2DTransferExecutor currently allocates from
  // pjrt_device->default_memory_space(), so non-default memory kinds (e.g.,
  // pinned_host) must fall back.
  if (!memory_kind.is_default()) {
    return false;
  }

  return true;
}

absl::Status ValidateInputByteStrides(const HostBuffer& input) {
  if (input.byte_strides().has_value() &&
      input.byte_strides()->size() != input.shape().dims().size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Byte strides rank (", input.byte_strides()->size(),
                     ") does not match input shape rank (",
                     input.shape().dims().size(), ")"));
  }
  return absl::OkStatus();
}

// Normalizes a single-device maximal `HloSharding` into a
// `SingleDeviceSharding` targeting its unique device.
absl::StatusOr<xla::ifrt::ShardingRef> ResolveSingleDeviceHloSharding(
    const xla::HloSharding& hlo_sharding, xla::ifrt::MemoryKind memory_kind,
    xla::ifrt::Client* client) {
  int unique_device_id = hlo_sharding.GetUniqueDevice();
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::Device * target_device,
      client->LookupDevice(xla::ifrt::DeviceId(unique_device_id)));
  if (!target_device) {
    return absl::InternalError(absl::StrCat(
        "LookupDevice returned null for device id ", unique_device_id));
  }
  return xla::ifrt::SingleDeviceSharding::Create(target_device, memory_kind);
}

// Allocates a host staging arena of `size` bytes (`size > 0`) using
// `host_allocator`. Returns `std::shared_ptr<void>` so multiple in-flight async
// H2D transfer callbacks can share ownership of the arena, preserving
// `HostMemoryAllocator::Deleter` in the control block so the arena is freed via
// the backend allocator once the last transfer completes.
absl::StatusOr<std::shared_ptr<void>> AllocateHostArena(
    xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
    xla::HostMemoryAllocator* absl_nonnull host_allocator,
    xla::PjRtDevice* absl_nonnull pjrt_device, size_t size) {
  // TODO: Route host arena allocations through
  // CommonPjRtClient::linearize_throttler()->AllocateStagingDest (as
  // AllocateLinearizeDest does) so the arena path shares the same in-flight
  // staging backpressure instead of bypassing it.
  const xla::HostMemoryAllocator::AllocateOptions alloc_options = {
      .numa_node = common_pjrt_client->raw_client()->GetNumaNode(
          pjrt_device->local_device_id()),
      .local_device_id = pjrt_device->local_device_id(),
  };
  xla::HostMemoryAllocator::OwnedPtr staging_buffer =
      host_allocator->Allocate(size, alloc_options);
  if (staging_buffer == nullptr) {
    // A large contiguous arena can fail if the pinned host pool is exhausted or
    // fragmented; fail fast with a clear error rather than chunking or falling
    // back to per-tensor transfers unless host fragmentation occurs in
    // practice.
    return absl::ResourceExhaustedError(absl::StrCat(
        "Failed to allocate a ", size,
        "-byte pinned host staging arena for host-to-device transfer."));
  }
  return std::shared_ptr<void>(std::move(staging_buffer));
}

// Allocates a single device arena and host staging arena on `pjrt_device`,
// linearizes `entries`, issues async H2D transfers, and returns the resulting
// `PjRtBuffer`s in `entries` order.
absl::StatusOr<std::vector<std::shared_ptr<xla::PjRtBuffer>>>
TransferDeviceArena(
    xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
    xla::HostMemoryAllocator* absl_nonnull host_memory_allocator,
    size_t arena_alignment, xla::PjRtDevice* absl_nonnull pjrt_device,
    std::vector<DeviceArenaEntry> entries) {
  TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                      pjrt_device->default_memory_space());

  // TODO: Coalesce small tensors into a single contiguous H2D
  // transfer + MultiSlice.
  std::vector<xla::PjRtRawBufferInterface::SliceInfo> transfer_slices(
      entries.size());
  int64_t total_arena_size_in_bytes = 0;
  for (size_t idx = 0; idx < entries.size(); ++idx) {
    total_arena_size_in_bytes =
        AlignUp(total_arena_size_in_bytes, arena_alignment);
    transfer_slices[idx] = {
        .offset = total_arena_size_in_bytes,
        .size = entries[idx].on_device_bytes,
    };
    total_arena_size_in_bytes += entries[idx].on_device_bytes;
  }

  TF_ASSIGN_OR_RETURN(xla::PjRtRawBufferRef device_raw_buffer,
                      common_pjrt_client->AllocateRawBuffer(
                          memory_space, total_arena_size_in_bytes,
                          /*retry_on_oom=*/true,
                          /*allocate_after=*/{}));

  // Decide zero-copy vs. host-arena staging in one place. Entries bypass the
  // host arena only when staging is enabled (so
  // `ShouldPerformZeroCopyLinearize` enforces `IsDmaMapped`) and PjRt confirms
  // the host buffer can be DMA'd directly (matching strides, no padding, and
  // host-DMA-aligned).
  const bool can_zero_copy =
      common_pjrt_client->should_stage_host_to_device_transfers();
  absl::InlinedVector<bool, 16> is_zero_copy(entries.size(), false);
  absl::InlinedVector<int, 16> non_zero_copy_indices;
  non_zero_copy_indices.reserve(entries.size());
  for (int idx = 0; idx < entries.size(); ++idx) {
    const DeviceArenaEntry& entry = entries[idx];
    if (can_zero_copy && common_pjrt_client->ShouldPerformZeroCopyLinearize(
                             entry.data, entry.on_device_shape,
                             entry.on_device_shape.element_type(),
                             entry.on_device_shape.dimensions(),
                             entry.byte_strides, memory_space)) {
      is_zero_copy[idx] = true;
    } else {
      non_zero_copy_indices.push_back(idx);
    }
  }

  std::vector<int64_t> host_arena_offsets(entries.size(), 0);
  std::shared_ptr<void> host_arena_shared;
  if (!non_zero_copy_indices.empty()) {
    int64_t total_host_arena_size_in_bytes = 0;
    for (int idx : non_zero_copy_indices) {
      total_host_arena_size_in_bytes =
          AlignUp(total_host_arena_size_in_bytes, arena_alignment);
      host_arena_offsets[idx] = total_host_arena_size_in_bytes;
      total_host_arena_size_in_bytes += entries[idx].on_device_bytes;
    }

    TF_ASSIGN_OR_RETURN(
        host_arena_shared,
        AllocateHostArena(common_pjrt_client, host_memory_allocator,
                          pjrt_device, total_host_arena_size_in_bytes));

    for (int idx : non_zero_copy_indices) {
      const DeviceArenaEntry& entry = entries[idx];
      const xla::PjRtRawBufferInterface::SliceInfo& slice =
          transfer_slices[idx];
      uint8_t* arena_destination =
          static_cast<uint8_t*>(host_arena_shared.get()) +
          host_arena_offsets[idx];
      TF_RETURN_IF_ERROR(common_pjrt_client->Linearize(
          absl::MakeSpan(arena_destination, slice.size), entry.data,
          entry.byte_strides, entry.on_device_shape,
          /*dynamic_sizes=*/{}, memory_space));
    }
  }

  std::vector<std::shared_ptr<xla::PjRtBuffer>> buffers(entries.size());
  for (size_t idx = 0; idx < entries.size(); ++idx) {
    DeviceArenaEntry& entry = entries[idx];
    const xla::PjRtRawBufferInterface::SliceInfo& transfer_slice =
        transfer_slices[idx];

    xla::PjRtDeviceEventRef copy_event;
    if (is_zero_copy[idx]) {
      // On-device layouts can include trailing tile padding (e.g., 1D 32-bit
      // tensors on TPU) that `ShouldPerformZeroCopyLinearize` omits from the
      // DMA size. Transfer at most `unpacked_host_bytes` so the H2D DMA never
      // reads past `entry.data`, while `Slice` below still carves out the full
      // padded `transfer_slice.size` on device.
      const int64_t transfer_size =
          std::min<int64_t>(transfer_slice.size, entry.unpacked_host_bytes);
      TF_ASSIGN_OR_RETURN(
          copy_event, device_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                          entry.data, transfer_slice.offset, transfer_size));
      copy_event.AndThen(
          [source_buffer = std::move(entry.source_buffer)]() {});
    } else {
      const uint8_t* arena_destination =
          static_cast<const uint8_t*>(host_arena_shared.get()) +
          host_arena_offsets[idx];
      TF_ASSIGN_OR_RETURN(
          copy_event,
          device_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
              arena_destination, transfer_slice.offset, transfer_slice.size));
      copy_event.AndThen([host_arena_shared]() {});
    }

    TF_ASSIGN_OR_RETURN(
        xla::PjRtRawBufferRef sliced_buffer,
        device_raw_buffer->Slice(transfer_slice.offset, transfer_slice.size));
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtBuffer> buffer,
        common_pjrt_client->DefineBuffer(
            std::move(entry.on_device_shape), memory_space,
            std::move(sliced_buffer), {std::move(copy_event)}));
    buffers[idx] = std::shared_ptr<xla::PjRtBuffer>(std::move(buffer));
  }
  return buffers;
}

}  // namespace

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
ArenaH2DTransferExecutor::Create(xla::ifrt::Client* client,
                                 tsl::thread::ThreadPool* thread_pool) {
  if (client == nullptr) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a non-null client.");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a non-null thread_pool.");
  }
  if (client->platform_id() == xla::CpuId()) {
    return absl::UnimplementedError(
        "ArenaH2DTransferExecutor is not supported on CPU.");
  }
  xla::ifrt::PjRtCompatibleClient* pjrt_compatible_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  xla::CommonPjRtClient* common_pjrt_client =
      pjrt_compatible_client != nullptr
          ? dynamic_cast<xla::CommonPjRtClient*>(
                pjrt_compatible_client->pjrt_client())
          : nullptr;
  if (common_pjrt_client == nullptr) {
    return absl::UnimplementedError(absl::StrCat(
        "ArenaH2DTransferExecutor requires a CommonPjRtClient (platform: ",
        client->platform_name(), ")."));
  }
  if (common_pjrt_client->raw_client() == nullptr) {
    return absl::InternalError(absl::StrCat(
        "CommonPjRtClient has a null PjRtRawClient on platform ",
        client->platform_name(), "."));
  }
  TF_ASSIGN_OR_RETURN(
      size_t arena_alignment,
      common_pjrt_client->raw_client()->GetDeviceAddressAlignment());
  if (arena_alignment == 0 || (arena_alignment & (arena_alignment - 1)) != 0) {
    return absl::InternalError(absl::StrCat(
        "GetDeviceAddressAlignment returned invalid alignment (must be a "
        "non-zero power of two): ",
        arena_alignment));
  }
  // Checked after GetDeviceAddressAlignment so backends without arena support
  // fail with Unimplemented above rather than InternalError here.
  xla::HostMemoryAllocator* host_memory_allocator =
      common_pjrt_client->GetHostMemoryAllocator();
  if (host_memory_allocator == nullptr) {
    return absl::InternalError(absl::StrCat(
        "CommonPjRtClient has a null HostMemoryAllocator on platform ",
        client->platform_name(), "."));
  }
  return absl::WrapUnique(new ArenaH2DTransferExecutor(
      pjrt_compatible_client, common_pjrt_client, host_memory_allocator,
      arena_alignment, thread_pool));
}

ArenaH2DTransferExecutor::ArenaH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* absl_nonnull client,
    xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
    xla::HostMemoryAllocator* absl_nonnull host_memory_allocator,
    size_t arena_alignment, tsl::thread::ThreadPool* absl_nonnull thread_pool)
    : client_(client),
      common_pjrt_client_(common_pjrt_client),
      host_memory_allocator_(host_memory_allocator),
      arena_alignment_(arena_alignment),
      thread_pool_(thread_pool) {}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
ArenaH2DTransferExecutor::ExecuteTransfer(
    absl::Span<const HostBuffer> inputs,
    absl::Span<const xla::ifrt::ShardingRef> input_shardings) const {
  if (inputs.size() != input_shardings.size()) {
    return absl::InvalidArgumentError("inputs and shardings size mismatch");
  }
  if (inputs.empty()) {
    return std::vector<xla::ifrt::ArrayRef>();
  }

  struct DeviceBatch {
    std::vector<DeviceArenaEntry> entries;
    std::vector<int> input_indices;
  };

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());
  absl::flat_hash_map<xla::PjRtDevice*, DeviceBatch> batches_by_device;
  // Per-input destination PjRtBuffer slots (`shared_ptr` as required by
  // `PjRtArray::Create`); remains null for fallback inputs.
  std::vector<std::shared_ptr<xla::PjRtBuffer>> arena_buffers(inputs.size());
  std::vector<xla::ifrt::ShardingRef> effective_shardings(inputs.size());
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  // Phase 1: Dispatch by sharding type and classify inputs into per-device
  // arena entries or fallback paths before any arena allocations or transfers
  // run.
  for (int i = 0; i < inputs.size(); ++i) {
    const xla::ifrt::ShardingRef& sharding = input_shardings[i];
    if (sharding == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input sharding at index ", i, " is null."));
    }

    std::optional<xla::ifrt::ShardingRef> single_device_sharding;
    if (const xla::ifrt::SingleDeviceSharding* ifrt_single_device_sharding =
            xla::ifrt::dyn_cast<const xla::ifrt::SingleDeviceSharding>(
                sharding.get())) {
      if (ifrt_single_device_sharding->devices()->devices().empty()) {
        return absl::InvalidArgumentError(
            "SingleDeviceSharding has no target devices.");
      }
      single_device_sharding = sharding;
    } else {
      const xla::ifrt::HloSharding* ifrt_hlo_sharding =
          xla::ifrt::dyn_cast<const xla::ifrt::HloSharding>(sharding.get());
      if (ifrt_hlo_sharding == nullptr) {
        return absl::UnimplementedError(
            absl::StrCat("Unsupported sharding for H2D transfer: ", *sharding));
      }
      const xla::HloSharding& hlo_sharding =
          ifrt_hlo_sharding->xla_hlo_sharding();

      if (hlo_sharding.IsSingleDevice()) {
        TF_ASSIGN_OR_RETURN(
            single_device_sharding,
            ResolveSingleDeviceHloSharding(hlo_sharding,
                                           sharding->memory_kind(), client_));
      } else if (hlo_sharding.IsReplicated()) {
        // TODO: Support replicated sharding in
        // ArenaH2DTransferExecutor.
      } else if (hlo_sharding.IsTiled()) {
        // TODO: Support tiled sharding in
        // ArenaH2DTransferExecutor.
      } else {
        return absl::UnimplementedError(absl::StrCat(
            "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
            hlo_sharding.ToString()));
      }
    }

    if (!single_device_sharding.has_value() ||
        !IsArenaEligibleInput(inputs[i], sharding->memory_kind())) {
      fallback_input_indices.push_back(i);
      continue;
    }
    TF_RETURN_IF_ERROR(ValidateInputByteStrides(inputs[i]));

    xla::ifrt::Device* target_device =
        (*single_device_sharding)->devices()->devices()[0];
    xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(
            target_device);
    if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
      return absl::InternalError("Target device is not PJRT compatible.");
    }
    xla::PjRtDevice* pjrt_device = pjrt_compatible_device->pjrt_device();
    TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                        pjrt_device->default_memory_space());

    TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                        xla::ifrt::ToPrimitiveType(inputs[i].dtype()));
    xla::Shape host_shape =
        xla::ShapeUtil::MakeShape(primitive_type, inputs[i].shape().dims());
    absl::InlinedVector<int64_t, 4> byte_strides;
    if (inputs[i].byte_strides().has_value()) {
      byte_strides = *inputs[i].byte_strides();
    } else {
      byte_strides.resize(inputs[i].shape().dims().size());
      TF_RETURN_IF_ERROR(xla::ShapeUtil::UnpackedByteStrides(
          host_shape, absl::MakeSpan(byte_strides)));
    }

    const int64_t unpacked_host_bytes = xla::ShapeUtil::ByteSizeOf(host_shape);
    TF_ASSIGN_OR_RETURN(
        xla::Shape on_device_shape,
        common_pjrt_client_->MakeDefaultShapeForMemorySpace(
            memory_space, host_shape,
            host_shape.has_layout() ? &host_shape.layout() : nullptr));
    TF_ASSIGN_OR_RETURN(int64_t on_device_bytes,
                        common_pjrt_client_->GetOnDeviceBytesCount(
                            memory_space, on_device_shape));

    effective_shardings[i] = *std::move(single_device_sharding);
    DeviceBatch& batch = batches_by_device[pjrt_device];
    batch.entries.push_back(DeviceArenaEntry{
        .on_device_shape = std::move(on_device_shape),
        .on_device_bytes = on_device_bytes,
        .unpacked_host_bytes = unpacked_host_bytes,
        .data = inputs[i].base(),
        .byte_strides = std::move(byte_strides),
        .source_buffer = inputs[i],
    });
    batch.input_indices.push_back(i);
  }

  // Phase 2: Execute per-device arena allocations, linearization, and async
  // H2D transfers.
  for (auto& [pjrt_device, batch] : batches_by_device) {
    TF_ASSIGN_OR_RETURN(
        std::vector<std::shared_ptr<xla::PjRtBuffer>> device_buffers,
        TransferDeviceArena(common_pjrt_client_, host_memory_allocator_,
                            arena_alignment_, pjrt_device,
                            std::move(batch.entries)));
    for (size_t b = 0; b < device_buffers.size(); ++b) {
      arena_buffers[batch.input_indices[b]] = std::move(device_buffers[b]);
    }
  }

  // Phase 3: Assemble IFRT Arrays for all arena-transferred inputs.
  for (int i = 0; i < inputs.size(); ++i) {
    if (arena_buffers[i] == nullptr) {
      continue;
    }
    xla::ifrt::PjRtArray::PjRtBuffers pjrt_buffers = {
        std::move(arena_buffers[i])};
    TF_ASSIGN_OR_RETURN(
        tsl::RCReference<xla::ifrt::PjRtArray> array,
        xla::ifrt::PjRtArray::Create(
            client_, inputs[i].dtype(), inputs[i].shape(),
            effective_shardings[i], std::move(pjrt_buffers),
            /*layout=*/std::shared_ptr<const xla::PjRtLayout>()));
    results[i] = tsl::RCReference<xla::ifrt::Array>(std::move(array));
  }

  // Phase 4: Initiate default transfers for fallback inputs (e.g. strings,
  // sub-byte types, 0-element tensors, non-default memory).
  for (int idx : fallback_input_indices) {
    TF_ASSIGN_OR_RETURN(
        results[idx],
        sharding_internal::MakeArrayFromHostBuffer(
            *client_, inputs[idx], input_shardings[idx]->devices(),
            input_shardings[idx], *thread_pool_));
  }

  return results;
}

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
ArenaH2DTransferExecutorFactory::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* thread_pool) {
  return ArenaH2DTransferExecutor::Create(client, thread_pool);
}

}  // namespace xla_serving

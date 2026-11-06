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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/device_event.h"  // from @xla
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_buffer.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
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

using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

constexpr int64_t AlignUp(int64_t value, size_t alignment) {
  const int64_t align = static_cast<int64_t>(alignment);
  return (value + align - 1) & ~(align - 1);
}

// Represents a single shard's target device, shape, and zero-copy pointer view
// into the original HostBuffer.
struct ShardTarget {
  // Target device assigned to receive this shard.
  xla::ifrt::Device* device;
  // Unpadded logical shape of this shard.
  xla::Shape shape;
  // Pointer to the start of this shard's data in the original HostBuffer.
  const void* data;
};

// Decision-only plan produced for an input eligible for arena H2D transfer.
struct H2DTransferPlan {
  // Ordered to match the target sharding's device list (1 for single-device,
  // N for replicated or partitioned).
  std::vector<ShardTarget> shard_targets;
  // The sharding to attach to the assembled PjRtArray (may be narrowed from a
  // multi-device maximal HloSharding to a SingleDeviceSharding).
  xla::ifrt::ShardingRef effective_sharding;
  // Byte strides of the source HostBuffer along each dimension.
  absl::InlinedVector<int64_t, 4> byte_strides;
};

// Represents an entry in a per-device arena transfer.
struct DeviceArenaEntry {
  // Pointer to the specific slot in `arena_inputs` where this buffer belongs.
  std::shared_ptr<xla::PjRtBuffer>* output_slot;
  // Canonical on-device shape with memory-space default layout populated.
  xla::Shape on_device_shape;
  // Size in bytes of `on_device_shape` in device memory (queried via
  // `CommonPjRtClient::GetOnDeviceBytesCount`).
  int64_t on_device_bytes;
  // Unpacked byte size of the shard on host (`ShapeUtil::ByteSizeOf`).
  int64_t unpacked_host_bytes;
  // Pointer to the start of this shard's data in `source_buffer`.
  const void* data;
  // Byte strides of `source_buffer` along each dimension.
  absl::InlinedVector<int64_t, 4> byte_strides;
  // Keeps the backing host memory alive during async zero-copy H2D transfers.
  HostBuffer source_buffer;
  // Whether this tensor can be transferred directly from `data` without
  // host-arena linearization.
  bool zero_copy;
};

// Inspects `input` and `sharding` to build a zero-copy transfer plan. Returns
// `std::nullopt` when the input must fall back to `MakeArrayFromHostBuffer`.
absl::StatusOr<std::optional<H2DTransferPlan>> GetH2DTransferPlanIfEligible(
    const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
    xla::ifrt::Client* client) {
  // CommonPjRtClient::Linearize and raw buffer H2D DMA only support fixed-size
  // POD element types, not tsl::tstring objects.
  if (input.dtype().kind() == xla::ifrt::DType::kString) {
    return std::nullopt;
  }

  // Sub-byte types (e.g. int4) are stored unpacked as kU8 in HostBuffer
  // (dtype != storage_dtype), whereas CommonPjRtClient::Linearize expects the
  // input buffer element width to match the on-device primitive type.
  if (input.dtype() != input.storage_dtype()) {
    return std::nullopt;
  }

  // Zero-element tensors have 0 on-device bytes and require no H2D DMA;
  // AllocateHostArena also requires size > 0.
  if (input.shape().num_elements() == 0) {
    return std::nullopt;
  }

  // TODO: ArenaH2DTransferExecutor currently allocates from
  // pjrt_device->default_memory_space(), so non-default memory kinds (e.g.,
  // pinned_host) must fall back.
  if (!sharding->memory_kind().is_default()) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(input.dtype()));
  xla::Shape full_xla_shape =
      xla::ShapeUtil::MakeShape(primitive_type, input.shape().dims());

  absl::InlinedVector<int64_t, 4> byte_strides;
  if (input.byte_strides().has_value()) {
    byte_strides = *input.byte_strides();
  } else {
    byte_strides.resize(input.shape().dims().size());
    TF_RETURN_IF_ERROR(xla::ShapeUtil::UnpackedByteStrides(
        full_xla_shape, absl::MakeSpan(byte_strides)));
  }
  if (byte_strides.size() != input.shape().dims().size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Byte strides rank (", byte_strides.size(),
                     ") does not match input shape rank (",
                     input.shape().dims().size(), ")"));
  }

  // Case 1: SingleDeviceSharding (tensor targeted to 1 device)
  if (const auto* single_device_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::SingleDeviceSharding>(
              sharding.get())) {
    absl::Span<xla::ifrt::Device* const> devices =
        single_device_sharding->devices()->devices();
    if (devices.empty()) {
      return absl::InvalidArgumentError(
          "SingleDeviceSharding has no target devices.");
    }
    return H2DTransferPlan{
        .shard_targets = {ShardTarget{
            .device = devices[0],
            .shape = std::move(full_xla_shape),
            .data = input.base(),
        }},
        .effective_sharding = sharding,
        .byte_strides = std::move(byte_strides),
    };
  }

  // Case 2: Tile-Maximal HloSharding (assigns an entire, unpartitioned tensor
  // to 1 specific device within a multi-device mesh)
  if (const auto* hlo_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::HloSharding>(sharding.get())) {
    const xla::HloSharding& xla_hlo_sharding = hlo_sharding->xla_hlo_sharding();
    if (!xla_hlo_sharding.IsTiled() &&
        !xla_hlo_sharding.IsReplicatedOrSingleDevice()) {
      return absl::UnimplementedError(absl::StrCat(
          "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
          xla_hlo_sharding.ToString()));
    }
    if (xla_hlo_sharding.IsSingleDevice()) {
      int unique_device_id = xla_hlo_sharding.GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Device * target_device,
          client->LookupDevice(xla::ifrt::DeviceId(unique_device_id)));
      if (!target_device) {
        return absl::InternalError(absl::StrCat(
            "LookupDevice returned null for device id ", unique_device_id));
      }
      return H2DTransferPlan{
          .shard_targets = {ShardTarget{
              .device = target_device,
              .shape = std::move(full_xla_shape),
              .data = input.base(),
          }},
          .effective_sharding = xla::ifrt::SingleDeviceSharding::Create(
              target_device, sharding->memory_kind()),
          .byte_strides = std::move(byte_strides),
      };
    }
  }

  // TODO: Support replicated and partitioned/tiled shardings in
  // ArenaH2DTransferExecutor.
  return std::nullopt;
}

// Allocates a host staging arena of `size` bytes (`size > 0`). Uses
// `CommonPjRtClient::GetHostMemoryAllocator` when available (TPU, GPU), or
// falls back to `DefaultAllocator` with `alignment` (CPU). Returns
// `std::shared_ptr<void>` so multiple in-flight async H2D transfer callbacks
// can share ownership of the arena and type-erase the backend deleter.
std::shared_ptr<void> AllocateHostArena(
    xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
    xla::PjRtDevice* absl_nonnull pjrt_device, size_t size, size_t alignment) {
  if (xla::HostMemoryAllocator* host_allocator =
          common_pjrt_client->GetHostMemoryAllocator();
      host_allocator != nullptr) {
    xla::HostMemoryAllocator::AllocateOptions alloc_options;
    if (common_pjrt_client->raw_client() != nullptr) {
      alloc_options.numa_node = common_pjrt_client->raw_client()->GetNumaNode(
          pjrt_device->local_device_id());
    }
    alloc_options.local_device_id = pjrt_device->local_device_id();
    return std::shared_ptr<void>(host_allocator->Allocate(size, alloc_options));
  }
  void* ptr = DefaultAllocator::GetInstance()->Allocate(size, alignment);
  return std::shared_ptr<void>(ptr, [alignment](void* p) {
    DefaultAllocator::GetInstance()->Free(p, alignment);
  });
}

absl::StatusOr<size_t> GetArenaAlignment(
    const xla::CommonPjRtClient* absl_nonnull common_pjrt_client) {
  if (common_pjrt_client->raw_client() == nullptr) {
    return absl::UnimplementedError(
        "GetDeviceAddressAlignment requires a non-null PjRtRawClient.");
  }
  return common_pjrt_client->raw_client()->GetDeviceAddressAlignment();
}

}  // namespace

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
ArenaH2DTransferExecutor::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options) {
  xla::ifrt::PjRtCompatibleClient* pjrt_compatible_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (!pjrt_compatible_client) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a PjRtCompatibleClient.");
  }
  auto* common_pjrt_client = dynamic_cast<xla::CommonPjRtClient*>(
      pjrt_compatible_client->pjrt_client());
  if (common_pjrt_client == nullptr) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a CommonPjRtClient.");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a non-null thread_pool.");
  }
  return absl::WrapUnique(new ArenaH2DTransferExecutor(
      pjrt_compatible_client, common_pjrt_client, std::move(device_list),
      thread_pool, std::move(options)));
}

ArenaH2DTransferExecutor::ArenaH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* absl_nonnull client,
    xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options)
    : client_(client),
      common_pjrt_client_(common_pjrt_client),
      device_list_(std::move(device_list)),
      thread_pool_(thread_pool),
      options_(std::move(options)) {}

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

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());

  absl::flat_hash_map<xla::ifrt::Device*, std::vector<DeviceArenaEntry>>
      entries_by_device;
  std::vector<std::optional<PjRtBuffers>> arena_inputs(inputs.size());
  std::vector<xla::ifrt::ShardingRef> effective_shardings(inputs.size());
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  // Phase 1: Classify inputs into per-device arena entries or fallback paths.
  const bool can_zero_copy =
      common_pjrt_client_->GetHostMemoryAllocator() == nullptr ||
      common_pjrt_client_->should_stage_host_to_device_transfers();
  for (int i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        std::optional<H2DTransferPlan> transfer_plan,
        GetH2DTransferPlanIfEligible(inputs[i], input_shardings[i], client_));
    if (!transfer_plan.has_value()) {
      fallback_input_indices.push_back(i);
      continue;
    }

    effective_shardings[i] = std::move(transfer_plan->effective_sharding);
    PjRtBuffers& slots =
        arena_inputs[i].emplace(transfer_plan->shard_targets.size());

    for (int k = 0; k < transfer_plan->shard_targets.size(); ++k) {
      ShardTarget& target = transfer_plan->shard_targets[k];
      xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(
              target.device);
      if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
        return absl::InternalError("Target device is not PJRT compatible.");
      }
      TF_ASSIGN_OR_RETURN(
          xla::PjRtMemorySpace * memory_space,
          pjrt_compatible_device->pjrt_device()->default_memory_space());
      const int64_t unpacked_host_bytes =
          xla::ShapeUtil::ByteSizeOf(target.shape);
      TF_ASSIGN_OR_RETURN(
          xla::Shape on_device_shape,
          common_pjrt_client_->MakeDefaultShapeForMemorySpace(
              memory_space, target.shape,
              target.shape.has_layout() ? &target.shape.layout() : nullptr));
      TF_ASSIGN_OR_RETURN(int64_t on_device_bytes,
                          common_pjrt_client_->GetOnDeviceBytesCount(
                              memory_space, on_device_shape));
      // On backends with a host staging allocator (TPU/GPU), only bypass the
      // host arena when staging is enabled and PjRt confirms the buffer can be
      // DMA'd directly (matching strides, no padding, host-DMA-aligned, and
      // already in DMA-mapped memory). On CPU (no host staging allocator),
      // bypass the intermediate host arena whenever no layout/padding/stride
      // transformation is needed.
      bool zero_copy =
          on_device_bytes > options_.small_tensor_threshold_in_bytes &&
          can_zero_copy &&
          common_pjrt_client_->ShouldPerformZeroCopyLinearize(
              target.data, on_device_shape, on_device_shape.element_type(),
              on_device_shape.dimensions(), transfer_plan->byte_strides,
              memory_space);

      entries_by_device[target.device].push_back(DeviceArenaEntry{
          .output_slot = &slots[k],
          .on_device_shape = std::move(on_device_shape),
          .on_device_bytes = on_device_bytes,
          .unpacked_host_bytes = unpacked_host_bytes,
          .data = target.data,
          .byte_strides = transfer_plan->byte_strides,
          .source_buffer = inputs[i],
          .zero_copy = zero_copy,
      });
    }
  }

  // Phase 2: Execute per-device arena allocations, linearization, and async
  // H2D transfers.
  size_t arena_alignment = 1;
  if (!entries_by_device.empty()) {
    TF_ASSIGN_OR_RETURN(arena_alignment,
                        GetArenaAlignment(common_pjrt_client_));
  }
  for (auto& [device, entries] : entries_by_device) {
    xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(device);
    if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
      return absl::InternalError("Target device is not PJRT compatible.");
    }
    xla::PjRtDevice* pjrt_device = pjrt_compatible_device->pjrt_device();
    TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                        pjrt_device->default_memory_space());

    absl::InlinedVector<int, 16> small_input_indices;
    absl::InlinedVector<int, 16> large_input_indices;
    for (int b = 0; b < entries.size(); ++b) {
      if (entries[b].on_device_bytes <=
          options_.small_tensor_threshold_in_bytes) {
        small_input_indices.push_back(b);
      } else {
        large_input_indices.push_back(b);
      }
    }
    if (small_input_indices.size() == 1) {
      const int idx = small_input_indices.front();
      DeviceArenaEntry& entry = entries[idx];
      entry.zero_copy =
          can_zero_copy && common_pjrt_client_->ShouldPerformZeroCopyLinearize(
                               entry.data, entry.on_device_shape,
                               entry.on_device_shape.element_type(),
                               entry.on_device_shape.dimensions(),
                               entry.byte_strides, memory_space);
      large_input_indices.push_back(idx);
      small_input_indices.clear();
    }

    // Pre-compute offsets and sizes for all tensors on the device in
    // "Large then Small" order so that all small tensors form a single
    // contiguous region at the end of the arena.
    std::vector<xla::PjRtRawBufferInterface::SliceInfo> transfer_slices(
        entries.size());
    int64_t total_arena_size_in_bytes = 0;
    for (int idx : large_input_indices) {
      total_arena_size_in_bytes =
          AlignUp(total_arena_size_in_bytes, arena_alignment);
      transfer_slices[idx] = {
          .offset = total_arena_size_in_bytes,
          .size = entries[idx].on_device_bytes,
      };
      total_arena_size_in_bytes += entries[idx].on_device_bytes;
    }
    const int64_t small_batch_start_offset =
        AlignUp(total_arena_size_in_bytes, arena_alignment);
    total_arena_size_in_bytes = small_batch_start_offset;
    for (int idx : small_input_indices) {
      total_arena_size_in_bytes =
          AlignUp(total_arena_size_in_bytes, arena_alignment);
      transfer_slices[idx] = {
          .offset = total_arena_size_in_bytes,
          .size = entries[idx].on_device_bytes,
      };
      total_arena_size_in_bytes += entries[idx].on_device_bytes;
    }

    // Allocate a single raw buffer on the device as the destination arena.
    TF_ASSIGN_OR_RETURN(xla::PjRtRawBufferRef device_raw_buffer,
                        common_pjrt_client_->AllocateRawBuffer(
                            memory_space, total_arena_size_in_bytes,
                            /*retry_on_oom=*/true,
                            /*allocate_after=*/{}));

    // Collect tensors that require staging and linearization in the host arena.
    absl::InlinedVector<int, 16> non_zero_copy_indices;
    non_zero_copy_indices.reserve(entries.size());
    for (int idx : large_input_indices) {
      if (!entries[idx].zero_copy) {
        non_zero_copy_indices.push_back(idx);
      }
    }
    for (int idx : small_input_indices) {
      non_zero_copy_indices.push_back(idx);
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

      host_arena_shared =
          AllocateHostArena(common_pjrt_client_, pjrt_device,
                            total_host_arena_size_in_bytes, arena_alignment);

      // Linearize all non-zero-copy tensors into their respective slices in
      // `host_arena_shared` in parallel.
      absl::Mutex status_mu;
      absl::Status linearize_status = absl::OkStatus();
      auto linearize_fn = [&](int64_t begin, int64_t end) {
        for (int64_t i = begin; i < end; ++i) {
          const int idx = non_zero_copy_indices[i];
          const DeviceArenaEntry& entry = entries[idx];
          const xla::PjRtRawBufferInterface::SliceInfo& slice =
              transfer_slices[idx];
          uint8_t* arena_destination =
              static_cast<uint8_t*>(host_arena_shared.get()) +
              host_arena_offsets[idx];
          absl::Status st = common_pjrt_client_->Linearize(
              absl::MakeSpan(arena_destination, slice.size), entry.data,
              entry.byte_strides, entry.on_device_shape,
              /*dynamic_sizes=*/{}, memory_space);
          if (!st.ok()) {
            absl::MutexLock lock(status_mu);
            linearize_status.Update(st);
            return;
          }
        }
      };

      if (non_zero_copy_indices.size() == 1) {
        linearize_fn(0, 1);
      } else {
        thread_pool_->ParallelFor(
            non_zero_copy_indices.size(),
            tsl::thread::ThreadPool::SchedulingParams::Adaptive(10000),
            linearize_fn);
      }
      TF_RETURN_IF_ERROR(linearize_status);
    }

    // 1. Transfer and slice large tensors individually.
    for (int idx : large_input_indices) {
      DeviceArenaEntry& entry = entries[idx];
      const xla::PjRtRawBufferInterface::SliceInfo& transfer_slice =
          transfer_slices[idx];

      xla::PjRtDeviceEventRef copy_event;
      if (entry.zero_copy) {
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

      DCHECK_EQ(transfer_slice.offset % arena_alignment, 0);
      TF_ASSIGN_OR_RETURN(
          xla::PjRtRawBufferRef sliced_buffer,
          device_raw_buffer->Slice(transfer_slice.offset, transfer_slice.size));
      TF_ASSIGN_OR_RETURN(
          std::unique_ptr<xla::PjRtBuffer> buffer,
          common_pjrt_client_->DefineBuffer(
              std::move(entry.on_device_shape), memory_space,
              std::move(sliced_buffer), {std::move(copy_event)}));
      *entry.output_slot = std::shared_ptr<xla::PjRtBuffer>(std::move(buffer));
    }

    // 2. Coalesce all small tensors into a single H2D transfer + MultiSlice.
    if (!small_input_indices.empty()) {
      const int64_t small_batch_size =
          total_arena_size_in_bytes - small_batch_start_offset;
      const uint8_t* small_batch_host_buffer =
          static_cast<const uint8_t*>(host_arena_shared.get()) +
          host_arena_offsets[small_input_indices.front()];

      TF_ASSIGN_OR_RETURN(xla::PjRtDeviceEventRef copy_to_device_event,
                          device_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                              small_batch_host_buffer, small_batch_start_offset,
                              small_batch_size));
      copy_to_device_event.AndThen(
          [host_arena_shared = std::move(host_arena_shared)]() {});

      std::vector<xla::PjRtRawBufferInterface::SliceInfo> small_slices;
      small_slices.reserve(small_input_indices.size());
      for (int idx : small_input_indices) {
        DCHECK_EQ(transfer_slices[idx].offset % arena_alignment, 0);
        small_slices.push_back(transfer_slices[idx]);
      }
      TF_ASSIGN_OR_RETURN(std::vector<xla::PjRtRawBufferRef> small_raw_buffers,
                          device_raw_buffer->MultiSlice(small_slices));

      for (size_t i = 0; i < small_input_indices.size(); ++i) {
        DeviceArenaEntry& entry = entries[small_input_indices[i]];
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<xla::PjRtBuffer> buffer,
            common_pjrt_client_->DefineBuffer(
                std::move(entry.on_device_shape), memory_space,
                std::move(small_raw_buffers[i]), {copy_to_device_event}));
        *entry.output_slot =
            std::shared_ptr<xla::PjRtBuffer>(std::move(buffer));
      }
    }
  }

  // Phase 3: Assemble IFRT Arrays for all arena-transferred inputs.
  for (int i = 0; i < inputs.size(); ++i) {
    if (!arena_inputs[i].has_value()) {
      continue;
    }
    PjRtBuffers& buffers = *arena_inputs[i];
    xla::ifrt::PjRtArray::PjRtBuffers pjrt_buffers(buffers.begin(),
                                                   buffers.end());
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
    TF_ASSIGN_OR_RETURN(results[idx],
                        sharding_internal::MakeArrayFromHostBuffer(
                            *client_, inputs[idx], device_list_,
                            input_shardings[idx], *thread_pool_));
  }

  return results;
}

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
ArenaH2DTransferExecutorFactory::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  return ArenaH2DTransferExecutor::Create(client, std::move(device_list),
                                          thread_pool, options_);
}

}  // namespace xla_serving

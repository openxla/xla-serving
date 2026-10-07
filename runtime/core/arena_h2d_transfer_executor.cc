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
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_plan.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/device_event.h"  // from @xla
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_buffer.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_device.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

namespace {

using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

// Alignment (in bytes) for arena slice offsets across TPU (<= 1024), GPU (256),
// and CPU (16).
constexpr int64_t kArenaAlignment = 1024;

constexpr int64_t AlignUp(int64_t value, int64_t alignment) {
  return (value + alignment - 1) & ~(alignment - 1);
}

// Represents an entry in a per-device arena batch.
struct DeviceArenaEntry {
  // Pointer to the specific slot in `batched_inputs` where this buffer belongs.
  std::shared_ptr<xla::PjRtBuffer>* output_slot;
  // Logical shard shape prior to memory-space canonicalization.
  xla::Shape shape;
  // Canonical on-device shape with memory-space default layout populated.
  xla::Shape on_device_shape;
  // Size in bytes of the tensor in device memory.
  int64_t on_device_bytes = 0;
  // Source host buffer (C-contiguous).
  HostBuffer source_buffer;
  // Whether this tensor can be transferred directly from `source_buffer`
  // without host-arena linearization.
  bool zero_copy = false;
};

// Allocates a host staging arena of `size` bytes. Uses `host_allocator` when
// available (TPU, GPU), or falls back to `DefaultAllocator` with `alignment`
// (CPU).
std::shared_ptr<void> AllocateHostArena(
    xla::HostMemoryAllocator* host_allocator, xla::PjRtDevice* pjrt_device,
    size_t size, size_t alignment) {
  if (size == 0) {
    return nullptr;
  }
  if (host_allocator != nullptr) {
    xla::HostMemoryAllocator::AllocateOptions alloc_options;
    alloc_options.local_device_id = pjrt_device->local_device_id();
    return std::shared_ptr<void>(host_allocator->Allocate(size, alloc_options));
  }
  void* ptr = DefaultAllocator::GetInstance()->Allocate(size, alignment);
  return std::shared_ptr<void>(ptr, [alignment](void* p) {
    DefaultAllocator::GetInstance()->Free(p, alignment);
  });
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
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "ArenaH2DTransferExecutor requires a non-null thread_pool.");
  }
  return absl::WrapUnique(new ArenaH2DTransferExecutor(
      pjrt_compatible_client, std::move(device_list), thread_pool, options));
}

ArenaH2DTransferExecutor::ArenaH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options)
    : client_(client),
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

  // If the underlying PjRtClient does not inherit from CommonPjRtClient (e.g.,
  // PjRtCApiClient), fall back to standard per-input IFRT host buffer transfer.
  auto* common_pjrt_client =
      dynamic_cast<xla::CommonPjRtClient*>(client_->pjrt_client());
  if (common_pjrt_client == nullptr) {
    for (int i = 0; i < inputs.size(); ++i) {
      TF_ASSIGN_OR_RETURN(std::optional<HostBuffer> contiguous_input,
                          CompactHostBufferIfStrided(inputs[i], *thread_pool_));
      const HostBuffer& input =
          contiguous_input.has_value() ? *contiguous_input : inputs[i];
      TF_ASSIGN_OR_RETURN(results[i],
                          sharding_internal::MakeArrayFromHostBuffer(
                              *client_, input, device_list_, input_shardings[i],
                              *thread_pool_));
    }
    return results;
  }

  absl::flat_hash_map<xla::ifrt::Device*, std::vector<DeviceArenaEntry>>
      batches_by_device;
  std::vector<std::optional<PjRtBuffers>> batched_inputs(inputs.size());
  std::vector<xla::ifrt::ShardingRef> effective_shardings(inputs.size());
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  // Phase 1: Classify inputs into per-device arena batches or fallback paths.
  for (int i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(std::optional<BatchPlan> batch_plan,
                        GetBatchPlanIfEligible(inputs[i], input_shardings[i],
                                               client_, *thread_pool_));
    if (!batch_plan.has_value()) {
      fallback_input_indices.push_back(i);
      continue;
    }

    effective_shardings[i] = std::move(batch_plan->effective_sharding);
    PjRtBuffers& slots =
        batched_inputs[i].emplace(batch_plan->shard_targets.size());

    for (int k = 0; k < batch_plan->shard_targets.size(); ++k) {
      ShardTarget& target = batch_plan->shard_targets[k];
      batches_by_device[target.device].push_back(DeviceArenaEntry{
          .output_slot = &slots[k],
          .shape = std::move(target.shape),
          .source_buffer = std::move(target.source_buffer),
      });
    }
  }

  // Phase 2: Execute per-device arena allocations, linearization, and async
  // H2D transfers.
  xla::HostMemoryAllocator* host_allocator =
      common_pjrt_client->GetHostMemoryAllocator();

  for (auto& [device, entries] : batches_by_device) {
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
      DeviceArenaEntry& entry = entries[b];
      TF_ASSIGN_OR_RETURN(
          entry.on_device_shape,
          common_pjrt_client->MakeDefaultShapeForMemorySpace(
              memory_space, entry.shape,
              entry.shape.has_layout() ? &entry.shape.layout() : nullptr));
      TF_ASSIGN_OR_RETURN(entry.on_device_bytes,
                          common_pjrt_client->GetOnDeviceBytesCount(
                              memory_space, entry.on_device_shape));
      if (entry.on_device_bytes <= options_.small_tensor_threshold_in_bytes) {
        small_input_indices.push_back(b);
      } else {
        large_input_indices.push_back(b);
      }
    }

    // Pre-compute offsets and sizes for all tensors on the device in
    // "Large then Small" order so that all small tensors form a single
    // contiguous region at the end of the arena.
    std::vector<xla::PjRtRawBufferInterface::SliceInfo> transfer_slices(
        entries.size());
    int64_t total_arena_size_in_bytes = 0;
    for (int idx : large_input_indices) {
      total_arena_size_in_bytes =
          AlignUp(total_arena_size_in_bytes, kArenaAlignment);
      transfer_slices[idx] = {
          .offset = total_arena_size_in_bytes,
          .size = entries[idx].on_device_bytes,
      };
      total_arena_size_in_bytes += entries[idx].on_device_bytes;
    }
    const int64_t small_batch_start_offset =
        AlignUp(total_arena_size_in_bytes, kArenaAlignment);
    total_arena_size_in_bytes = small_batch_start_offset;
    for (int idx : small_input_indices) {
      total_arena_size_in_bytes =
          AlignUp(total_arena_size_in_bytes, kArenaAlignment);
      transfer_slices[idx] = {
          .offset = total_arena_size_in_bytes,
          .size = entries[idx].on_device_bytes,
      };
      total_arena_size_in_bytes += entries[idx].on_device_bytes;
    }

    // Allocate a single raw buffer on the device as the destination arena.
    TF_ASSIGN_OR_RETURN(xla::PjRtRawBufferRef device_raw_buffer,
                        common_pjrt_client->AllocateRawBuffer(
                            memory_space, total_arena_size_in_bytes,
                            /*retry_on_oom=*/true,
                            /*allocate_after=*/{}));

    // Determine which large tensors are eligible for zero-copy transfer vs
    // staging in the host arena.
    absl::InlinedVector<int, 16> non_zero_copy_indices;
    non_zero_copy_indices.reserve(entries.size());
    for (int idx : large_input_indices) {
      DeviceArenaEntry& entry = entries[idx];
      entry.zero_copy =
          device_raw_buffer->GetHostPointer() == nullptr &&
          common_pjrt_client->ShouldPerformZeroCopyLinearize(
              entry.source_buffer.raw_data().data(), entry.on_device_shape,
              entry.on_device_shape.element_type(),
              entry.on_device_shape.dimensions(),
              /*byte_strides=*/std::nullopt, memory_space);
      if (!entry.zero_copy) {
        non_zero_copy_indices.push_back(idx);
      }
    }
    for (int idx : small_input_indices) {
      non_zero_copy_indices.push_back(idx);
    }

    std::shared_ptr<void> host_arena_shared;
    if (!non_zero_copy_indices.empty()) {
      host_arena_shared =
          AllocateHostArena(host_allocator, pjrt_device,
                            total_arena_size_in_bytes, kArenaAlignment);

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
              static_cast<uint8_t*>(host_arena_shared.get()) + slice.offset;
          absl::Status st = common_pjrt_client->Linearize(
              absl::MakeSpan(arena_destination, slice.size),
              entry.source_buffer.raw_data().data(),
              /*byte_strides=*/{}, entry.on_device_shape,
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
        const void* src_ptr = entry.source_buffer.raw_data().data();
        const int64_t transfer_size = std::min<int64_t>(
            transfer_slice.size, entry.source_buffer.raw_data().size());
        TF_ASSIGN_OR_RETURN(
            copy_event, device_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                            src_ptr, transfer_slice.offset, transfer_size));
        copy_event.AndThen(
            [source_buffer = std::move(entry.source_buffer)]() {});
      } else {
        const uint8_t* arena_destination =
            static_cast<const uint8_t*>(host_arena_shared.get()) +
            transfer_slice.offset;
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
      *entry.output_slot = std::shared_ptr<xla::PjRtBuffer>(std::move(buffer));
    }

    // 2. Coalesce all small tensors into a single H2D transfer + MultiSlice.
    if (!small_input_indices.empty()) {
      const int64_t small_batch_size =
          total_arena_size_in_bytes - small_batch_start_offset;
      const uint8_t* small_batch_host_buffer =
          static_cast<const uint8_t*>(host_arena_shared.get()) +
          small_batch_start_offset;

      TF_ASSIGN_OR_RETURN(xla::PjRtDeviceEventRef copy_to_device_event,
                          device_raw_buffer->CopyRawHostToDeviceAndReturnEvent(
                              small_batch_host_buffer, small_batch_start_offset,
                              small_batch_size));
      copy_to_device_event.AndThen(
          [host_arena_shared = std::move(host_arena_shared)]() {});

      std::vector<xla::PjRtRawBufferInterface::SliceInfo> small_slices;
      small_slices.reserve(small_input_indices.size());
      for (int idx : small_input_indices) {
        small_slices.push_back(transfer_slices[idx]);
      }
      TF_ASSIGN_OR_RETURN(std::vector<xla::PjRtRawBufferRef> small_raw_buffers,
                          device_raw_buffer->MultiSlice(small_slices));

      for (size_t i = 0; i < small_input_indices.size(); ++i) {
        DeviceArenaEntry& entry = entries[small_input_indices[i]];
        TF_ASSIGN_OR_RETURN(
            std::unique_ptr<xla::PjRtBuffer> buffer,
            common_pjrt_client->DefineBuffer(
                std::move(entry.on_device_shape), memory_space,
                std::move(small_raw_buffers[i]), {copy_to_device_event}));
        *entry.output_slot =
            std::shared_ptr<xla::PjRtBuffer>(std::move(buffer));
      }
    }
  }

  // Phase 3: Assemble IFRT Arrays for all arena-transferred inputs.
  for (int i = 0; i < inputs.size(); ++i) {
    if (!batched_inputs[i].has_value()) {
      continue;
    }
    PjRtBuffers& buffers = *batched_inputs[i];
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

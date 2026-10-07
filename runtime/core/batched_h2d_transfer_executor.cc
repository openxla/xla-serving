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

#include "runtime/core/batched_h2d_transfer_executor.h"  // from @xla_serving

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
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_plan.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/literal.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
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
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

namespace {

using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

// Represents an entry in a per-device batch, mapping a device buffer slot back
// to the output slot and the source host buffer.
struct DeviceBatchEntry {
  // Pointer to the specific slot in `batched_inputs` where this buffer belongs.
  std::shared_ptr<xla::PjRtBuffer>* output_slot;
  // Source host buffer, needed for DMA transfer.
  HostBuffer source_buffer;
};

// Encapsulates data needed to batch inputs on a target device.
struct DeviceBatchData {
  std::vector<DeviceBatchEntry> entries;
  absl::InlinedVector<xla::Shape, 4> shapes;
};

}  // namespace

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
BatchedH2DTransferExecutor::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options) {
  xla::ifrt::PjRtCompatibleClient* pjrt_compatible_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (!pjrt_compatible_client) {
    return absl::InvalidArgumentError(
        "BatchedH2DTransferExecutor requires a PjRtCompatibleClient.");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "BatchedH2DTransferExecutor requires a non-null thread_pool.");
  }
  return absl::WrapUnique(new BatchedH2DTransferExecutor(
      pjrt_compatible_client, std::move(device_list), thread_pool, options));
}

BatchedH2DTransferExecutor::BatchedH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options)
    : client_(client),
      device_list_(std::move(device_list)),
      thread_pool_(thread_pool),
      options_(std::move(options)) {}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
BatchedH2DTransferExecutor::ExecuteTransfer(
    absl::Span<const HostBuffer> inputs,
    absl::Span<const xla::ifrt::ShardingRef> input_shardings) const {
  if (inputs.size() != input_shardings.size()) {
    return absl::InvalidArgumentError("inputs and shardings size mismatch");
  }

  absl::flat_hash_map<xla::ifrt::Device*, DeviceBatchData> batches_by_device;
  std::vector<std::optional<PjRtBuffers>> batched_inputs(inputs.size());
  std::vector<xla::ifrt::ShardingRef> effective_shardings(inputs.size());
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());

  // Phase 1: Classify inputs into per-device batches or fallback paths.
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
      DeviceBatchData& batch = batches_by_device[target.device];
      batch.entries.push_back({
          .output_slot = &slots[k],
          .source_buffer = std::move(target.source_buffer),
      });
      batch.shapes.push_back(std::move(target.shape));
    }
  }

  // Phase 2: Execute per-device batched allocations and async DMA transfers.
  xla::PjRtClient* pjrt_client = client_->pjrt_client();

  for (auto& [device, batch_data] : batches_by_device) {
    xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(device);
    if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
      return absl::InternalError("Target device is not PJRT compatible.");
    }
    TF_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * memory_space,
        pjrt_compatible_device->pjrt_device()->default_memory_space());

    std::vector<DeviceBatchEntry>& batched_entries = batch_data.entries;
    const absl::InlinedVector<xla::Shape, 4>& batched_shapes =
        batch_data.shapes;

    // Use a shared_ptr so we can keep the manager alive until all
    // transfers for this batch complete.
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>
        transfer_manager;
    TF_ASSIGN_OR_RETURN(transfer_manager,
                        pjrt_client->CreateBuffersForAsyncHostToDevice(
                            batched_shapes, memory_space));

    int num_batched = batched_entries.size();

    // 1. Retrieve Loop: Get all PjRtBuffer handles for this device and store
    // them directly into their target output slots.
    for (int b = 0; b < num_batched; ++b) {
      std::unique_ptr<xla::PjRtBuffer> buffer =
          transfer_manager->RetrieveBuffer(b);
      if (!buffer) return absl::InternalError("Failed to retrieve PJRT buffer");

      DeviceBatchEntry& entry = batched_entries[b];
      *entry.output_slot = std::shared_ptr<xla::PjRtBuffer>(buffer.release());
    }

    // 2. Transfer Loop: Start asynchronous DMA transfers for all buffers on
    // this device.
    for (int b = 0; b < num_batched; ++b) {
      DeviceBatchEntry& entry = batched_entries[b];
      auto literal = std::make_unique<xla::BorrowingLiteral>(
          static_cast<const char*>(entry.source_buffer.raw_data().data()),
          batched_shapes[b]);
      xla::BorrowingLiteral* literal_ptr = literal.get();

      // Capture input (kept alive by copy/refcount), literal (safely moved),
      // and transfer_manager (kept alive by shared_ptr). This prevents the host
      // buffers and backend managers from being destroyed while the transfer is
      // still physically ongoing in the background, and keeps the
      // transfer_manager's internally owned buffer alive until completion.
      auto on_done = [input = std::move(entry.source_buffer),
                      literal = std::move(literal),
                      transfer_manager]() mutable {};

      if (absl::Status st = transfer_manager->TransferLiteralToBuffer(
              b, *literal_ptr, std::move(on_done));
          !st.ok()) {
        transfer_manager->SetBufferError(b, st);
      }
    }
  }

  // Phase 3: Assemble IFRT Arrays for all batched inputs.
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

  // Phase 4: Concurrently initiate default transfers for fallback inputs
  // (e.g. strings, sub-byte types, 0-element tensors, non-default memory).
  for (int idx : fallback_input_indices) {
    TF_ASSIGN_OR_RETURN(
        results[idx],
        sharding_internal::MakeArrayFromHostBuffer(
            *client_, inputs[idx], device_list_, input_shardings[idx],
            *thread_pool_));
  }

  return results;
}

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
BatchedH2DTransferExecutorFactory::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  return BatchedH2DTransferExecutor::Create(
      client, std::move(device_list), thread_pool, options_);
}

}  // namespace xla_serving

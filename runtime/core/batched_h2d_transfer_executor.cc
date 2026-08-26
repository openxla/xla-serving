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

#include "runtime/core/batched_h2d_transfer_executor.h"  // from @neptune

#include <memory>
#include <utility>
#include <optional>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "runtime/core/h2d_transfer_executor.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/sharding_utils.h"  // from @neptune
#include "xla/literal.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_device.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

namespace {

using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

// Encapsulates planning information for an eligible batched input.
struct BatchPlan {
  xla::Shape shape;
  absl::InlinedVector<xla::ifrt::Device*, 4> devices;
};

// Represents an entry in a per-device batch, mapping a device buffer slot back
// to the output slot and the source host buffer.
struct DeviceBatchEntry {
  // Pointer to the specific slot in `batched_inputs` where this buffer belongs.
  std::shared_ptr<xla::PjRtBuffer>* output_slot;
  // Pointer to the source host buffer, needed for DMA transfer.
  const HostBuffer* source_buffer;
};

// Encapsulates data needed to batch inputs on a target device.
struct DeviceBatchData {
  std::vector<DeviceBatchEntry> entries;
  absl::InlinedVector<xla::Shape, 4> shapes;
};

// Inspects the input buffer and sharding to determine if it qualifies for
// batched transfer. If eligible, returns the target shape and device list.
// Returns std::nullopt if the input should route to the fallback path (e.g.
// partitioned shardings, strings, sub-byte types, custom host byte strides).
absl::StatusOr<std::optional<BatchPlan>> GetBatchPlanIfEligible(
    const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
    xla::ifrt::Client* client) {
  // Disqualify strings and packed storage (sub-byte types).
  if (input.dtype().kind() == xla::ifrt::DType::kString ||
      input.dtype() != input.storage_dtype()) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(input.dtype()));

  // Default MakeShape assumes a dense, major-to-minor layout.
  // If the buffer uses custom byte strides (e.g. view or slice),
  // route it to the fallback path which correctly handles host layouts.
  if (input.byte_strides().has_value()) {
    return std::nullopt;
  }

  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(primitive_type, input.shape().dims());

  // Case 1: SingleDeviceSharding (tensor targeted to 1 device)
  if (const auto* single_device_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::SingleDeviceSharding>(
              sharding.get())) {
    absl::Span<xla::ifrt::Device* const> devices =
        single_device_sharding->devices()->devices();
    if (devices.empty()) {
      return std::nullopt;
    }
    return BatchPlan{
        .shape = std::move(xla_shape),
        .devices = {devices.begin(), devices.end()},
    };
  }

  // Case 2: Tile-Maximal HloSharding (assigns an entire, unpartitioned tensor
  // to 1 specific device within a multi-device mesh)
  if (const auto* hlo_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::HloSharding>(
              sharding.get())) {
    if (hlo_sharding->xla_hlo_sharding().IsSingleDevice()) {
      int unique_device_id =
          hlo_sharding->xla_hlo_sharding().GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Device* target_device,
          client->LookupDevice(xla::ifrt::DeviceId(unique_device_id)));
      if (!target_device) {
        return std::nullopt;
      }
      return BatchPlan{
          .shape = std::move(xla_shape),
          .devices = {target_device},
      };
    }
  }

  // Case 3: Fully Replicated Sharding (covers HloSharding::Replicate() and any
  // sharding where the entire tensor is cloned across all target devices)
  if (sharding->IsFullyReplicated()) {
    absl::Span<xla::ifrt::Device* const> devices =
        sharding->devices()->devices();
    if (devices.empty()) {
      return std::nullopt;
    }
    return BatchPlan{
        .shape = std::move(xla_shape),
        .devices = {devices.begin(), devices.end()},
    };
  }

  // Partitioned/tiled or unsupported sharding types route to fallback.
  return std::nullopt;
}

}  // namespace

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
BatchedH2DTransferExecutor::Create(xla::ifrt::Client* client,
                                   xla::ifrt::DeviceListRef device_list,
                                   tsl::thread::ThreadPool* thread_pool,
                                   Options options) {
  xla::ifrt::PjRtCompatibleClient* pjrt_compatible_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (!pjrt_compatible_client) {
    return absl::InvalidArgumentError(
        "BatchedH2DTransferExecutor requires a PjRtCompatibleClient.");
  }
  return absl::WrapUnique(new BatchedH2DTransferExecutor(
      pjrt_compatible_client, std::move(device_list), thread_pool, options));
}

BatchedH2DTransferExecutor::BatchedH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* client,
    xla::ifrt::DeviceListRef device_list, tsl::thread::ThreadPool* thread_pool,
    Options options)
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
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());

  // Phase 1: Classify inputs into per-device batches or fallback paths.
  for (int i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        std::optional<BatchPlan> batch_plan,
        GetBatchPlanIfEligible(inputs[i], input_shardings[i], client_));
    if (!batch_plan.has_value()) {
      fallback_input_indices.push_back(i);
      continue;
    }

    PjRtBuffers& slots =
        batched_inputs[i].emplace(batch_plan->devices.size());

    for (int dev_idx = 0; dev_idx < batch_plan->devices.size(); ++dev_idx) {
      xla::ifrt::Device* device = batch_plan->devices[dev_idx];
      DeviceBatchData& batch = batches_by_device[device];
      batch.entries.push_back(
          {.output_slot = &slots[dev_idx], .source_buffer = &inputs[i]});
      if (dev_idx == batch_plan->devices.size() - 1) {
        batch.shapes.push_back(std::move(batch_plan->shape));
      } else {
        batch.shapes.push_back(batch_plan->shape);
      }
    }
  }

  // Phase 2: Execute per-device batched allocations and async DMA transfers.
  xla::PjRtClient* pjrt_client = client_->pjrt_client();

  for (const auto& [device, batch_data] : batches_by_device) {
    xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(device);
    if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
      return absl::InternalError("Target device is not PJRT compatible.");
    }
    TF_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * memory_space,
        pjrt_compatible_device->pjrt_device()->default_memory_space());

    const std::vector<DeviceBatchEntry>& batched_entries = batch_data.entries;
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

      const DeviceBatchEntry& entry = batched_entries[b];
      *entry.output_slot = std::shared_ptr<xla::PjRtBuffer>(buffer.release());
    }

    // 2. Transfer Loop: Start asynchronous DMA transfers for all buffers on
    // this device.
    for (int b = 0; b < num_batched; ++b) {
      const DeviceBatchEntry& entry = batched_entries[b];
      const HostBuffer& input = *entry.source_buffer;
      auto literal = std::make_unique<xla::BorrowingLiteral>(
          static_cast<const char*>(input.raw_data().data()), batched_shapes[b]);
      xla::BorrowingLiteral* literal_ptr = literal.get();

      // Capture input (kept alive by copy/refcount), literal (safely moved),
      // and transfer_manager (kept alive by shared_ptr). This prevents the host
      // buffers and backend managers from being destroyed while the transfer is
      // still physically ongoing in the background, and keeps the
      // transfer_manager's internally owned buffer alive until completion.
      auto on_done = [input, literal = std::move(literal),
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
            input_shardings[i], std::move(pjrt_buffers),
            /*layout=*/std::shared_ptr<const xla::PjRtLayout>()));
    results[i] = tsl::RCReference<xla::ifrt::Array>(std::move(array));
  }

  // Phase 4: Concurrently initiate default transfers for fallback inputs
  // (e.g. partitioned/tiled shardings, strings, strided buffers).
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
    xla::ifrt::Client* client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* thread_pool) {
  return BatchedH2DTransferExecutor::Create(
      client, std::move(device_list), thread_pool, options_);
}

}  // namespace neptune

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
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
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

// Encapsulates data needed to batch inputs that share the same target device.
// Keeps track of the original input indices and their corresponding shapes.
struct BatchData {
  std::vector<int> indices;
  absl::InlinedVector<xla::Shape, 4> shapes;
};

absl::StatusOr<bool> TryEnqueueForBatching(
    int index, const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
    xla::ifrt::Client* client,
    absl::flat_hash_map<xla::ifrt::Device*, BatchData>& batches) {
  xla::ifrt::Device* target_device = nullptr;
  if (const auto* single_device_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::SingleDeviceSharding>(
              sharding.get())) {
    target_device = single_device_sharding->devices()->devices().front();
  } else if (const auto* hlo_sharding =
                 xla::ifrt::dyn_cast_or_null<xla::ifrt::HloSharding>(
                     sharding.get())) {
    // Handle XLA's tile-maximal sharding, which assigns an entire,
    // unpartitioned tensor to just one specific device within a multi-device
    // scenario.
    if (hlo_sharding->xla_hlo_sharding().IsSingleDevice()) {
      int unique_device_id = hlo_sharding->xla_hlo_sharding().GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(
          target_device,
          client->LookupDevice(xla::ifrt::DeviceId(unique_device_id)));
    }
  }

  if (!target_device) {
    return false;
  }

  // Disqualify strings and packed storage (sub-byte types).
  if (input.dtype().kind() == xla::ifrt::DType::kString ||
      input.dtype() != input.storage_dtype()) {
    return false;
  }

  absl::StatusOr<xla::PrimitiveType> primitive_type_or =
      xla::ifrt::ToPrimitiveType(input.dtype());
  if (!primitive_type_or.ok()) {
    return false;
  }

  // Default MakeShape assumes a dense, major-to-minor layout.
  // If the buffer uses custom byte strides (e.g. view or slice),
  // route it to the fallback path which correctly handles host layouts.
  if (input.byte_strides().has_value()) {
    return false;
  }

  xla::Shape xla_shape =
      xla::ShapeUtil::MakeShape(*primitive_type_or, input.shape().dims());

  BatchData& batch = batches[target_device];
  batch.indices.push_back(index);
  batch.shapes.push_back(std::move(xla_shape));
  return true;
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

  absl::flat_hash_map<xla::ifrt::Device*, BatchData> batches_by_device;
  std::vector<int> fallback_input_indices;

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        bool queued,
        TryEnqueueForBatching(i, inputs[i], input_shardings[i], client_,
                              batches_by_device));
    if (!queued) {
      fallback_input_indices.push_back(i);
    }
  }

  // Handle batched transfers
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

    const std::vector<int>& batched_indices = batch_data.indices;
    const absl::InlinedVector<xla::Shape, 4>& batched_shapes =
        batch_data.shapes;

    // Use a shared_ptr so we can keep the manager alive until all
    // transfers for this batch complete.
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>
        transfer_manager;
    TF_ASSIGN_OR_RETURN(transfer_manager,
                        pjrt_client->CreateBuffersForAsyncHostToDevice(
                            batched_shapes, memory_space));

    int num_batched = batched_indices.size();

    // 1. Retrieve Loop: Get all buffers and wrap them in IFRT arrays.
    for (int b = 0; b < num_batched; ++b) {
      std::unique_ptr<xla::PjRtBuffer> buffer =
          transfer_manager->RetrieveBuffer(b);
      if (!buffer) return absl::InternalError("Failed to retrieve PJRT buffer");

      int original_idx = batched_indices[b];
      std::shared_ptr<xla::PjRtBuffer> buffer_shared(buffer.release());

      absl::StatusOr<tsl::RCReference<xla::ifrt::PjRtCompatibleArray>>
          array_or = client_->CreatePjRtArray(buffer_shared);
      if (!array_or.ok()) return array_or.status();

      results[original_idx] =
          tsl::RCReference<xla::ifrt::Array>(std::move(array_or.value()));
    }

    // 2. Transfer Loop: Start data transfer for all buffers.
    for (int b = 0; b < num_batched; ++b) {
      int original_idx = batched_indices[b];
      const HostBuffer& input = inputs[original_idx];
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

  // Concurrently initiate default transfers
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

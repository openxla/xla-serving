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
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
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
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
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
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

namespace {

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
  // host staging arena allocation also requires size > 0.
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
  if (common_pjrt_client->GetHostMemoryAllocator() == nullptr) {
    return absl::InternalError(absl::StrCat(
        "CommonPjRtClient has a null HostMemoryAllocator on platform ",
        client->platform_name(), "."));
  }
  return absl::WrapUnique(
      new ArenaH2DTransferExecutor(pjrt_compatible_client, thread_pool));
}

ArenaH2DTransferExecutor::ArenaH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* absl_nonnull client,
    tsl::thread::ThreadPool* absl_nonnull thread_pool)
    : client_(client), thread_pool_(thread_pool) {}

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

  // Phase 1: Dispatch by sharding type and check arena eligibility for all
  // inputs up front so unsupported shardings fail fast before any transfers
  // run.
  std::vector<std::optional<xla::ifrt::ShardingRef>> effective_shardings(
      inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
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

    if (single_device_sharding.has_value() &&
        IsArenaEligibleInput(inputs[i], sharding->memory_kind())) {
      TF_RETURN_IF_ERROR(ValidateInputByteStrides(inputs[i]));
      effective_shardings[i] = *std::move(single_device_sharding);
    }
  }

  // TODO: Transfer eligible single-device inputs
  // (`effective_shardings[i]` populated) via per-device host and device arenas
  // instead of MakeArrayFromHostBuffer.
  std::vector<xla::ifrt::ArrayRef> results(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    const xla::ifrt::ShardingRef& target_sharding =
        effective_shardings[i].has_value() ? *effective_shardings[i]
                                           : input_shardings[i];
    TF_ASSIGN_OR_RETURN(results[i],
                        sharding_internal::MakeArrayFromHostBuffer(
                            *client_, inputs[i], target_sharding->devices(),
                            target_sharding, *thread_pool_));
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

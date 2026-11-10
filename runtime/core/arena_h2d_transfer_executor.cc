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
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/raw_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

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
  auto* common_pjrt_client =
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

  // TODO: Add single-device eligibility checks and arena-backed
  // H2D transfers in follow-up CLs.
  std::vector<xla::ifrt::ArrayRef> results(inputs.size());
  for (size_t i = 0; i < inputs.size(); ++i) {
    TF_ASSIGN_OR_RETURN(
        results[i], sharding_internal::MakeArrayFromHostBuffer(
                        *client_, inputs[i], input_shardings[i]->devices(),
                        input_shardings[i], *thread_pool_));
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

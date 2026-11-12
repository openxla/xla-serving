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

#ifndef XLA_SERVING_RUNTIME_CORE_ARENA_H2D_TRANSFER_EXECUTOR_H_
#define XLA_SERVING_RUNTIME_CORE_ARENA_H2D_TRANSFER_EXECUTOR_H_

#include <cstddef>
#include <memory>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Executor that performs explicit arena-based host-to-device transfers using
// OpenXLA's CommonPjRtClient and PjRtRawBuffer APIs across TPU and GPU.
//
// Per target device, allocates a single device arena and a single host staging
// arena, linearizes eligible inputs into the host arena, and transfers them
// into aligned slices of the device arena.
class ArenaH2DTransferExecutor : public H2DTransferExecutor {
 public:
  // Creates an ArenaH2DTransferExecutor, validating backend prerequisites
  // (non-CPU CommonPjRtClient with HostMemoryAllocator and power-of-two
  // GetDeviceAddressAlignment). Returns a non-OK status if `client` or
  // `thread_pool` is invalid or `client` does not support arena transfers.
  static absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, tsl::thread::ThreadPool* thread_pool);

  ~ArenaH2DTransferExecutor() override = default;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings) const override;

 private:
  ArenaH2DTransferExecutor(
      xla::ifrt::PjRtCompatibleClient* absl_nonnull client,
      xla::CommonPjRtClient* absl_nonnull common_pjrt_client,
      xla::HostMemoryAllocator* absl_nonnull host_memory_allocator,
      size_t arena_alignment,
      tsl::thread::ThreadPool* absl_nonnull thread_pool);

  // IFRT client used for device lookup, PjRtArray assembly, and fallback
  // transfers via MakeArrayFromHostBuffer. Not owned.
  xla::ifrt::PjRtCompatibleClient* absl_nonnull client_;
  // Underlying CommonPjRtClient (client_->pjrt_client()), cached to avoid
  // dynamic_cast on every ExecuteTransfer call and used for raw buffer
  // allocation and layout linearization. Not owned.
  xla::CommonPjRtClient* absl_nonnull common_pjrt_client_;
  // Host staging memory allocator
  // (common_pjrt_client_->GetHostMemoryAllocator()), validated non-null in
  // Create. Not owned.
  xla::HostMemoryAllocator* absl_nonnull host_memory_allocator_;
  // Required byte alignment for slice offsets in the device arena (queried once
  // in Create via PjRtRawClient::GetDeviceAddressAlignment).
  const size_t arena_alignment_;
  // Thread pool used to parallelize host-arena tensor linearization
  // (CommonPjRtClient::Linearize) and fallback transfers
  // (MakeArrayFromHostBuffer). Not owned.
  tsl::thread::ThreadPool* absl_nonnull thread_pool_;
};

class ArenaH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  ArenaH2DTransferExecutorFactory() = default;
  ~ArenaH2DTransferExecutorFactory() override = default;

  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) override;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_ARENA_H2D_TRANSFER_EXECUTOR_H_

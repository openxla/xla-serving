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
#include <cstdint>
#include <memory>
#include <utility>
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
// Per target device, allocates a single device arena, coalesces small tensors
// through a host staging arena into a single H2D transfer, and transfers large
// tensors individually into aligned slices of the device arena.
class ArenaH2DTransferExecutor : public H2DTransferExecutor {
 public:
  struct Options {
    // Tensors with on-device size <= this threshold (in bytes) are linearized
    // into a contiguous host arena region and transferred in a single H2D DMA
    // call per device, then sliced into per-tensor buffers. Tensors larger than
    // this threshold are transferred individually into their slices of the
    // device arena. If <= 0, small-tensor coalescing is disabled and all
    // tensors are transferred individually.
    int64_t small_tensor_threshold_in_bytes = -1;
  };

  // Creates an ArenaH2DTransferExecutor if `client` supports arena-based
  // transfers, or returns a non-OK status if `client` or `thread_pool` is
  // invalid or `client` is unsupported (e.g. CPU).
  static absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, tsl::thread::ThreadPool* thread_pool,
      Options options);

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
      tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options);

  // IFRT client used for device lookup, array assembly, and fallback
  // transfers. Not owned.
  xla::ifrt::PjRtCompatibleClient* absl_nonnull client_;
  // Underlying PJRT client, cached to avoid repeated dynamic casts during
  // arena allocation and tensor linearization. Not owned.
  xla::CommonPjRtClient* absl_nonnull common_pjrt_client_;
  // Allocator for pinned host staging arenas. Not owned.
  xla::HostMemoryAllocator* absl_nonnull host_memory_allocator_;
  // Required byte alignment for slice offsets within a device arena.
  const size_t arena_alignment_;
  // Thread pool used for fallback transfers. Not owned.
  tsl::thread::ThreadPool* absl_nonnull thread_pool_;
  Options options_;
};

class ArenaH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  explicit ArenaH2DTransferExecutorFactory(
      ArenaH2DTransferExecutor::Options options = {})
      : options_(std::move(options)) {}

  ~ArenaH2DTransferExecutorFactory() override = default;

  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) override;

 private:
  ArenaH2DTransferExecutor::Options options_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_ARENA_H2D_TRANSFER_EXECUTOR_H_

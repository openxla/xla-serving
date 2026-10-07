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

#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Executor that performs explicit arena-based host-to-device transfers using
// OpenXLA's CommonPjRtClient and PjRtRawBuffer APIs across TPU, GPU, and CPU.
//
// Per target device, allocates a single device arena (AllocateRawBuffer) and
// a single host staging arena (GetHostMemoryAllocator), coalesces small tensors
// into a single H2D transfer + MultiSlice, and transfers large tensors
// individually into their slices of the device arena.
class ArenaH2DTransferExecutor : public H2DTransferExecutor {
 public:
  struct Options {
    // Tensors with on-device size <= this threshold (in bytes) are linearized
    // into a contiguous host arena region and transferred in a single H2D DMA
    // call per device, then sliced via MultiSlice. Tensors larger than this
    // threshold are transferred individually into their slice of the device
    // arena. If <= 0, small-tensor coalescing is disabled and all tensors are
    // transferred individually.
    int64_t small_tensor_threshold_in_bytes = 64 * 1024;
  };

  static absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options);

  ~ArenaH2DTransferExecutor() override = default;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings) const override;

 private:
  ArenaH2DTransferExecutor(xla::ifrt::PjRtCompatibleClient* client,
                           xla::ifrt::DeviceListRef device_list,
                           tsl::thread::ThreadPool* absl_nonnull thread_pool,
                           Options options);

  xla::ifrt::PjRtCompatibleClient* client_;  // Not owned.
  xla::ifrt::DeviceListRef device_list_;
  tsl::thread::ThreadPool* absl_nonnull thread_pool_;  // Not owned.
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
      tsl::thread::ThreadPool* absl_nonnull thread_pool) override;

 private:
  ArenaH2DTransferExecutor::Options options_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_ARENA_H2D_TRANSFER_EXECUTOR_H_

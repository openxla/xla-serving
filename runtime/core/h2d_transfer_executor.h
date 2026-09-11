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

#ifndef XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_
#define XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Interface for Host-to-Device (Arena) Transfers.
// Implementations of this interface handle linear packing of HostBuffers along
// with efficient multi-slice allocation & execution to the underlying devices.
//
// NOTE: Any concrete implementation of this interface MUST be completely
// thread-safe and stateless (or internally synchronized), because
// `ExecuteTransfer` will be invoked concurrently by multiple threads from the
// inference serving thread pool.
class H2DTransferExecutor {
 public:
  virtual ~H2DTransferExecutor() = default;

  // Converts the given `inputs` (HostBuffers) to XLA IFRT Arrays.
  // Returns handles to the resulting IFRT Arrays, which represent asynchronous
  // transfer operations provided by the underlying backend.
  //
  // `inputs`: The user-provided HostBuffers to transfer.
  // `input_shardings`: The exact IFRT Shardings required for these arrays.
  virtual absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings) const = 0;
};

// Factory interface for creating an `H2DTransferExecutor`.
class H2DTransferExecutorFactory {
 public:
  virtual ~H2DTransferExecutorFactory() = default;

  // Creates the `H2DTransferExecutor`.
  //
  // `client`: The IFRT client used for allocation/transfer.
  // `device_list`: The devices targeted for this execution.
  // `thread_pool`: A thread pool to use for concurrent packing/conversion.
  virtual absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) = 0;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_

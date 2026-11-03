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

#ifndef XLA_SERVING_RUNTIME_CORE_BATCHED_H2D_TRANSFER_EXECUTOR_H_
#define XLA_SERVING_RUNTIME_CORE_BATCHED_H2D_TRANSFER_EXECUTOR_H_

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

// Executor that groups and issues batched host-to-device transfers.
// This implementation groups HostBuffers targeted for the same devices and
// delegates them as a batched multi-slice allocation to the IFRT client.
//
// Inputs bound to a static shape (`HostBuffer::static_shape()`) that is larger
// than their data shape are transferred with dynamic DMA: the device buffer is
// allocated for the static shape, only the host data is linearized (in the
// device layout) and copied into the front of that buffer, and the resulting
// array has the static shape. This requires a single-device sharding and a
// `xla::CommonPjRtClient` backend.
class BatchedH2DTransferExecutor : public H2DTransferExecutor {
 public:
  struct Options {};

  static absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options);

  ~BatchedH2DTransferExecutor() override = default;

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings) const override;

 private:
  BatchedH2DTransferExecutor(xla::ifrt::PjRtCompatibleClient* client,
                             xla::ifrt::DeviceListRef device_list,
                             tsl::thread::ThreadPool* absl_nonnull thread_pool,
                             Options options);

  // Transfers `input`, whose static shape is larger than its data shape, with
  // dynamic DMA (see the class comment).
  absl::StatusOr<xla::ifrt::ArrayRef> ExecuteDynamicDmaTransfer(
      const HostBuffer& input, const xla::ifrt::ShardingRef& sharding) const;

  xla::ifrt::PjRtCompatibleClient* client_;  // Not owned.
  xla::ifrt::DeviceListRef device_list_;
  tsl::thread::ThreadPool* absl_nonnull thread_pool_;  // Not owned.
  Options options_;
};

class BatchedH2DTransferExecutorFactory : public H2DTransferExecutorFactory {
 public:
  explicit BatchedH2DTransferExecutorFactory(
      BatchedH2DTransferExecutor::Options options = {})
      : options_(std::move(options)) {}

  ~BatchedH2DTransferExecutorFactory() override = default;

  absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      xla::ifrt::Client* client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* absl_nonnull thread_pool) override;

 private:
  BatchedH2DTransferExecutor::Options options_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_BATCHED_H2D_TRANSFER_EXECUTOR_H_

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
#include <optional>
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
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Executor that groups and issues batched host-to-device transfers.
// This implementation groups HostBuffers targeted for the same devices and
// delegates them as a batched multi-slice allocation to the IFRT client.
//
// An input with fewer rows than its static shape (`static_shapes` in
// `ExecuteTransfer`) is transferred with dynamic DMA, outside the batched
// transfer. The device buffer is allocated for the static shape and only the
// host data is copied, into the front of it. The rest of the buffer is left
// uninitialized. This needs a single-device sharding and a PjRt client derived
// from `xla::CommonPjRtClient`.
//
// It also needs dimension 0 to be the major-most dimension of the layout that
// the device uses for the static shape by default. Only then are the rows of
// the input the front of the buffer. TPU chooses that layout to minimize
// padding, and for an array of rank 2 or more it often has another dimension
// there (f32[8,2] gets `{0,1:T(2,128)}`). On TPU such an input has to be
// flattened to rank 1 before it gets here.
//
// On TPU the element type must also be at most 32 bits wide. TPU stores a
// wider element as several 32-bit words, in one plane per word: the buffer of
// an s64 array holds the low words of all elements and then the high words.
// The high words of an input with fewer rows would not be where the program
// reads them.
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
      absl::Span<const xla::ifrt::ShardingRef> input_shardings,
      absl::Span<const std::optional<xla::ifrt::Shape>> static_shapes)
      const override;

 private:
  BatchedH2DTransferExecutor(xla::ifrt::PjRtCompatibleClient* client,
                             xla::ifrt::DeviceListRef device_list,
                             tsl::thread::ThreadPool* absl_nonnull thread_pool,
                             Options options);

  // Transfers `input`, which has fewer rows than `static_shape`, with dynamic
  // DMA (see the class comment). Returns an array of `static_shape`.
  absl::StatusOr<xla::ifrt::ArrayRef> ExecuteDynamicDmaTransfer(
      const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
      const xla::ifrt::Shape& static_shape) const;

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

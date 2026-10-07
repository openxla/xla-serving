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

#ifndef XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_PLAN_H_
#define XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_PLAN_H_

#include <optional>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Represents a target shard destination on a device.
struct ShardTarget {
  xla::ifrt::Device* device;
  xla::Shape shape;
  HostBuffer source_buffer;
};

// Encapsulates planning information for an eligible batched/arena input.
struct BatchPlan {
  std::vector<ShardTarget> shard_targets;
  // Sharding passed to `PjRtArray::Create` (`SingleDeviceSharding` for
  // single-device `HloSharding` so device count matches `pjrt_buffers.size()`).
  xla::ifrt::ShardingRef effective_sharding;
};

// Inspects the input buffer and sharding to determine if it qualifies for
// batched/arena transfer. If eligible, returns the batch plan with shard
// targets (with any strided/transposed input materialized to C-contiguous, and
// partitioned/tiled inputs disassembled into per-shard host buffers).
// Returns std::nullopt if the input should route to the fallback path (e.g.
// strings, sub-byte types, 0-element tensors, non-default memory).
absl::StatusOr<std::optional<BatchPlan>> GetBatchPlanIfEligible(
    const HostBuffer& original_input, const xla::ifrt::ShardingRef& sharding,
    xla::ifrt::Client* client, tsl::thread::ThreadPool& thread_pool);

// Materializes a C-contiguous copy of `input` if it has non-default byte
// strides; returns `std::nullopt` if `input` is already contiguous or does not
// use byte strides (e.g. strings, sub-byte types, <= 1 element).
absl::StatusOr<std::optional<HostBuffer>> CompactHostBufferIfStrided(
    const HostBuffer& input, tsl::thread::ThreadPool& thread_pool);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_PLAN_H_

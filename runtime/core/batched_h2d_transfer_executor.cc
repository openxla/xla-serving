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

#include "runtime/core/batched_h2d_transfer_executor.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <memory>
#include <numeric>
#include <optional>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/layout_util.h"  // from @xla
#include "xla/literal.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/device_event.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_buffer.h"  // from @xla
#include "xla/pjrt/transpose.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/index.h"  // from @xla
#include "xla/python/ifrt/index_domain.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_device.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/status_macros.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl
#include "tsl/profiler/lib/traceme_encode.h"  // from @tsl

namespace xla_serving {

namespace {

using PjRtBuffers = std::vector<std::shared_ptr<xla::PjRtBuffer>>;

auto* dynamic_dma_bytes_counter = ::tsl::monitoring::Counter<1>::New(
    "/xla_serving/input_transfer/dynamic_dma_bytes",
    "Bytes handled by dynamic DMA transfers: `allocated` is the on-device "
    "size of the static-shape buffers, `transferred` the bytes of actual data "
    "copied into them.",
    "kind");

// Returns true if `input` is to be transferred with dynamic DMA: it is bound to
// a static shape that has more elements than its data shape. An input bound to
// its own shape takes the regular path.
bool IsDynamicDmaInput(const HostBuffer& input,
                       const std::optional<xla::ifrt::Shape>& static_shape) {
  return static_shape.has_value() &&
         static_shape->num_elements() > input.shape().num_elements();
}

// The device a whole, unpartitioned array lives on, together with the sharding
// to create the array with.
struct SingleDeviceTarget {
  xla::ifrt::Device* device;
  xla::ifrt::ShardingRef effective_sharding;
};

// Resolves `sharding` to a single device if it places the entire array on one
// device: a `SingleDeviceSharding`, a tile-maximal `HloSharding` that assigns
// the whole array to one device of a multi-device mesh, or any sharding over a
// single-device list (replicating or tiling over one device still puts the
// whole array on that device). Returns `std::nullopt` for any other sharding.
absl::StatusOr<std::optional<SingleDeviceTarget>> GetSingleDeviceTarget(
    const xla::ifrt::ShardingRef& sharding, xla::ifrt::Client* client) {
  if (const auto* single_device_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::SingleDeviceSharding>(
              sharding.get())) {
    absl::Span<xla::ifrt::Device* const> devices =
        single_device_sharding->devices()->devices();
    if (devices.empty()) {
      return std::nullopt;
    }
    return SingleDeviceTarget{.device = devices[0],
                              .effective_sharding = sharding};
  }
  if (const auto* hlo_sharding =
          xla::ifrt::dyn_cast_or_null<xla::ifrt::HloSharding>(sharding.get())) {
    if (hlo_sharding->xla_hlo_sharding().IsSingleDevice()) {
      int unique_device_id = hlo_sharding->xla_hlo_sharding().GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Device * target_device,
          client->LookupDevice(xla::ifrt::DeviceId(unique_device_id)));
      if (!target_device) {
        return std::nullopt;
      }
      return SingleDeviceTarget{
          .device = target_device,
          .effective_sharding = xla::ifrt::SingleDeviceSharding::Create(
              target_device, sharding->memory_kind())};
    }
  }
  if (sharding->devices()->size() == 1) {
    return SingleDeviceTarget{.device = sharding->devices()->devices()[0],
                              .effective_sharding = sharding};
  }
  return std::nullopt;
}

// Represents a target shard destination on a device.
struct ShardTarget {
  xla::ifrt::Device* device;
  xla::Shape shape;
  HostBuffer source_buffer;
};

// Encapsulates planning information for an eligible batched input.
struct BatchPlan {
  std::vector<ShardTarget> shard_targets;
  // Sharding passed to `PjRtArray::Create` (`SingleDeviceSharding` for
  // single-device `HloSharding` so device count matches `pjrt_buffers.size()`).
  xla::ifrt::ShardingRef effective_sharding;
};

// Represents an entry in a per-device batch, mapping a device buffer slot back
// to the output slot and the source host buffer.
struct DeviceBatchEntry {
  // Pointer to the specific slot in `batched_inputs` where this buffer belongs.
  std::shared_ptr<xla::PjRtBuffer>* output_slot;
  // Source host buffer, needed for DMA transfer.
  HostBuffer source_buffer;
};

// Encapsulates data needed to batch inputs on a target device.
struct DeviceBatchData {
  std::vector<DeviceBatchEntry> entries;
  absl::InlinedVector<xla::Shape, 4> shapes;
};

// Inspects partitioned/tiled sharding to determine if it qualifies for batched
// transfer, disassembles the host buffer, and builds the batch plan.
absl::StatusOr<BatchPlan> GetShardedBatchPlan(
    const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
    xla::PrimitiveType primitive_type, tsl::thread::ThreadPool& thread_pool) {
  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::IndexDomain> index_domains,
      sharding->IndexDomains(
          input.shape(), xla::ifrt::SingleDeviceShardSemantics::kAllShards));
  if (index_domains.size() <= 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expected multiple index domains for partitioned/tiled sharding, but "
        "got ",
        index_domains.size()));
  }

  // Verify equal rank and shape across all index domains.
  const xla::ifrt::Shape& first_shard_shape = index_domains[0].shape();
  if (first_shard_shape.dims().size() != input.shape().dims().size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Rank mismatch between host buffer (", input.shape().dims().size(),
        ") and shard index domain (", first_shard_shape.dims().size(), ")"));
  }
  for (size_t k = 1; k < index_domains.size(); ++k) {
    if (index_domains[k].shape() != first_shard_shape) {
      return absl::UnimplementedError(absl::StrCat(
          "Unequal shard shapes across index domains are not supported: ",
          first_shard_shape, " vs ", index_domains[k].shape()));
    }
  }

  // Calculate partitions along each axis and verify even divisibility.
  std::vector<int32_t> num_partitions_per_axis;
  int total_num_partitions = 1;
  num_partitions_per_axis.reserve(input.shape().dims().size());
  for (int dim = 0; dim < input.shape().dims().size(); ++dim) {
    int64_t target_size = first_shard_shape.dims()[dim];
    int64_t dim_size = input.shape().dims()[dim];
    if (target_size <= 0 || dim_size % target_size != 0) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Only support even sharding, but input buffer dimension ", dim,
          " of size ", dim_size, " is not evenly divisible by shard size ",
          target_size));
    }
    int num_partitions = dim_size / target_size;
    total_num_partitions *= num_partitions;
    num_partitions_per_axis.push_back(num_partitions);
  }

  absl::Span<xla::ifrt::Device* const> sharding_devices =
      sharding->devices()->devices();
  if (total_num_partitions <= 0 ||
      sharding_devices.size() < total_num_partitions ||
      sharding_devices.size() % total_num_partitions != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Number of devices ", sharding_devices.size(),
        " not a multiple of number of partitions ", total_num_partitions));
  }
  int num_replicas = sharding_devices.size() / total_num_partitions;

  // Pairs each IndexDomain with its corresponding device and tracks its
  // original shard index so we can sort them lexicographically by origin (from
  // major to minor dimension) to match the disassembled slice order, while
  // retaining the ability to map shards back to their logical positions.
  struct IndexDomainDevice {
    // Index domain defining the spatial sub-region and origin for this shard.
    const xla::ifrt::IndexDomain* index_domain;
    // Target device assigned to receive this shard.
    xla::ifrt::Device* device;
    // The index of this `device`/`index_domain` in the
    // sharding.devices/index_domains.
    int original_shard_index;
  };
  std::vector<IndexDomainDevice> index_domain_devices;
  index_domain_devices.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    index_domain_devices.push_back({&index_domains[i], sharding_devices[i], i});
  }
  std::sort(index_domain_devices.begin(), index_domain_devices.end(),
            [](const IndexDomainDevice& a, const IndexDomainDevice& b) {
              return std::lexicographical_compare(
                  a.index_domain->origin().elements().begin(),
                  a.index_domain->origin().elements().end(),
                  b.index_domain->origin().elements().begin(),
                  b.index_domain->origin().elements().end());
            });

  // Verify each slice has identical origins across its replicas.
  if (num_replicas > 1) {
    for (int s = 0; s < total_num_partitions; ++s) {
      const xla::ifrt::Index& expected_origin =
          index_domain_devices[s * num_replicas].index_domain->origin();
      for (int r = 1; r < num_replicas; ++r) {
        if (index_domain_devices[s * num_replicas + r].index_domain->origin() !=
            expected_origin) {
          return absl::FailedPreconditionError(
              "Replicas of slice do not have identical index domain origins");
        }
      }
    }
  }

  // Disassemble input buffer into slices.
  TF_ASSIGN_OR_RETURN(std::vector<HostBuffer> split_buffers,
                      sharding_internal::DisassembleHostBuffer(
                          input, num_partitions_per_axis, thread_pool));
  if (split_buffers.size() != total_num_partitions) {
    return absl::InternalError(
        absl::StrCat("Disassembled slices count mismatch: expected ",
                     total_num_partitions, " but got ", split_buffers.size()));
  }

  xla::Shape shard_xla_shape =
      xla::ShapeUtil::MakeShape(primitive_type, first_shard_shape.dims());

  std::vector<ShardTarget> shard_targets(sharding_devices.size());

  for (int slice_idx = 0; slice_idx < total_num_partitions; ++slice_idx) {
    for (int r = 0; r < num_replicas; ++r) {
      int sorted_idx = slice_idx * num_replicas + r;
      const IndexDomainDevice& item = index_domain_devices[sorted_idx];
      const bool is_last_replica = (r == num_replicas - 1);
      const bool is_last_shard = (sorted_idx == sharding_devices.size() - 1);
      shard_targets[item.original_shard_index] = ShardTarget{
          .device = item.device,
          .shape = is_last_shard ? std::move(shard_xla_shape) : shard_xla_shape,
          .source_buffer = is_last_replica ? std::move(split_buffers[slice_idx])
                                           : split_buffers[slice_idx],
      };
    }
  }

  return BatchPlan{
      .shard_targets = std::move(shard_targets),
      .effective_sharding = sharding,
  };
}

// Returns true if the buffer has standard dense major-to-minor byte strides.
bool HasDefaultByteStrides(const HostBuffer& buffer,
                           xla::PrimitiveType primitive_type) {
  // Any tensor with 0 or 1 element has no gaps between elements, so its
  // representation at buffer.base() is already contiguous.
  if (buffer.shape().num_elements() <= 1) {
    return true;
  }
  if (!buffer.byte_strides().has_value()) {
    return true;
  }
  const auto& strides = *buffer.byte_strides();
  const auto& dims = buffer.shape().dims();
  if (strides.size() != dims.size()) {
    return false;
  }
  return xla::LayoutUtil::ByteStridesIsMajorToMinor(strides, dims,
                                                    primitive_type);
}

// Linearizes a strided or transposed HostBuffer into a new C-contiguous
// HostBuffer.
absl::StatusOr<HostBuffer> MakeContiguousHostBuffer(
    const HostBuffer& input, xla::PrimitiveType primitive_type,
    tsl::thread::ThreadPool& thread_pool) {
  if (input.shape().num_elements() <= 1) {
    return input;
  }

  HostBuffer contiguous_buffer(input.dtype(), input.shape());

  std::vector<int64_t> permutation(input.shape().dims().size());
  std::iota(permutation.begin(), permutation.end(), 0);
  // Cap max threads per transpose (matching ConcatCPUImpl in host_buffer.h) to
  // avoid saturating the shared pool under high concurrency; TransposePlan also
  // enforces a 1MB/64MB per-thread minimum and runs smaller buffers inline.
  constexpr int kMaxTransposeThreads = 4;
  std::optional<xla::TransposePlan::Striding> input_striding;
  if (input.byte_strides().has_value()) {
    input_striding.emplace(*input.byte_strides());
  }
  xla::TransposePlan::Options options{
      .elem_size_in_bytes = static_cast<size_t>(
          xla::ShapeUtil::ByteSizeOfPrimitiveType(primitive_type)),
      .dims = input.shape().dims(),
      .permutation = permutation,
      .input_striding = input_striding,
      .num_threads = std::min(kMaxTransposeThreads, thread_pool.NumThreads()),
  };

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::TransposePlan> plan,
                      xla::TransposePlan::Create(options));
  plan->Execute(input.base(), contiguous_buffer.base(),
                [&thread_pool](std::function<void()> fn) {
                  thread_pool.Schedule(std::move(fn));
                });
  return contiguous_buffer;
}

// Inspects the input buffer and sharding to determine if it qualifies for
// batched transfer. If eligible, returns the batch plan with shard targets.
// Returns std::nullopt if the input should route to the fallback path (e.g.
// non-uniform shardings, strings, sub-byte types, 0-element tensors). Returns
// an error for an input bound to a static shape whose sharding spans several
// devices, since neither the batched nor the fallback path can honour the
// static shape for it.
absl::StatusOr<std::optional<BatchPlan>> GetBatchPlanIfEligible(
    const HostBuffer& original_input, const xla::ifrt::ShardingRef& sharding,
    const std::optional<xla::ifrt::Shape>& static_shape,
    xla::ifrt::Client* client, tsl::thread::ThreadPool& thread_pool) {
  // Disqualify strings, packed storage (sub-byte types), and 0-element tensors.
  if (original_input.dtype().kind() == xla::ifrt::DType::kString ||
      original_input.dtype() != original_input.storage_dtype() ||
      original_input.shape().num_elements() == 0) {
    return std::nullopt;
  }

  // TODO: Support batching non-default memory_kind (e.g.,
  // pinned_host) by resolving PjRtMemorySpace* from sharding->memory_kind() and
  // grouping batches by PjRtMemorySpace* instead of defaulting to
  // default_memory_space().
  if (!sharding->memory_kind().is_default()) {
    return std::nullopt;
  }

  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(original_input.dtype()));

  std::optional<HostBuffer> contiguous_input;
  if (!HasDefaultByteStrides(original_input, primitive_type)) {
    TF_ASSIGN_OR_RETURN(
        contiguous_input,
        MakeContiguousHostBuffer(original_input, primitive_type, thread_pool));
  }
  const HostBuffer& input =
      contiguous_input.has_value() ? *contiguous_input : original_input;

  // Case 1: The whole array lives on one device: a `SingleDeviceSharding`, a
  // tile-maximal `HloSharding` that assigns the unpartitioned array to one
  // device of a multi-device mesh, or any sharding over a single-device list.
  TF_ASSIGN_OR_RETURN(std::optional<SingleDeviceTarget> single_device_target,
                      GetSingleDeviceTarget(sharding, client));
  if (single_device_target.has_value()) {
    xla::Shape xla_shape =
        xla::ShapeUtil::MakeShape(primitive_type, input.shape().dims());
    return BatchPlan{
        .shard_targets = {ShardTarget{
            .device = single_device_target->device,
            .shape = std::move(xla_shape),
            .source_buffer = contiguous_input.has_value()
                                 ? std::move(*contiguous_input)
                                 : input,
        }},
        .effective_sharding =
            std::move(single_device_target->effective_sharding),
    };
  }

  // Beyond this point the array is spread over several devices, which is not
  // supported for an input bound to a static shape (see
  // `ExecuteDynamicDmaTransfer`).
  if (static_shape.has_value()) {
    return absl::UnimplementedError(absl::StrCat(
        "Input of shape ", original_input.shape(), " bound to static shape ",
        *static_shape, " has a sharding that spans more than one device (",
        *sharding,
        "). Static shapes are only supported for single-device inputs."));
  }

  // Case 2: Fully replicated sharding (covers `HloSharding::Replicate()` and
  // any sharding where the entire array is cloned onto every target device).
  if (sharding->IsFullyReplicated()) {
    absl::Span<xla::ifrt::Device* const> devices =
        sharding->devices()->devices();
    if (devices.empty()) {
      return std::nullopt;
    }
    xla::Shape xla_shape =
        xla::ShapeUtil::MakeShape(primitive_type, input.shape().dims());
    std::vector<ShardTarget> shard_targets;
    shard_targets.reserve(devices.size());
    for (int dev_idx = 0; dev_idx < devices.size(); ++dev_idx) {
      const bool is_last = (dev_idx == devices.size() - 1);
      shard_targets.push_back(ShardTarget{
          .device = devices[dev_idx],
          .shape = is_last ? std::move(xla_shape) : xla_shape,
          .source_buffer = (is_last && contiguous_input.has_value())
                               ? std::move(*contiguous_input)
                               : input,
      });
    }
    return BatchPlan{
        .shard_targets = std::move(shard_targets),
        .effective_sharding = sharding,
    };
  }

  // Case 3: Partitioned / tiled sharding (shards sliced and distributed across
  // devices).
  TF_ASSIGN_OR_RETURN(
      BatchPlan plan,
      GetShardedBatchPlan(input, sharding, primitive_type, thread_pool));
  return plan;
}

}  // namespace

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
BatchedH2DTransferExecutor::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options) {
  xla::ifrt::PjRtCompatibleClient* pjrt_compatible_client =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleClient>(client);
  if (!pjrt_compatible_client) {
    return absl::InvalidArgumentError(
        "BatchedH2DTransferExecutor requires a PjRtCompatibleClient.");
  }
  if (thread_pool == nullptr) {
    return absl::InvalidArgumentError(
        "BatchedH2DTransferExecutor requires a non-null thread_pool.");
  }
  return absl::WrapUnique(new BatchedH2DTransferExecutor(
      pjrt_compatible_client, std::move(device_list), thread_pool, options));
}

BatchedH2DTransferExecutor::BatchedH2DTransferExecutor(
    xla::ifrt::PjRtCompatibleClient* client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool, Options options)
    : client_(client),
      device_list_(std::move(device_list)),
      thread_pool_(thread_pool),
      options_(std::move(options)) {}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
BatchedH2DTransferExecutor::ExecuteTransfer(
    absl::Span<const HostBuffer> inputs,
    absl::Span<const xla::ifrt::ShardingRef> input_shardings,
    absl::Span<const std::optional<xla::ifrt::Shape>> static_shapes) const {
  if (inputs.size() != input_shardings.size()) {
    return absl::InvalidArgumentError("inputs and shardings size mismatch");
  }
  if (!static_shapes.empty() && static_shapes.size() != inputs.size()) {
    return absl::InvalidArgumentError("inputs and static_shapes size mismatch");
  }
  static const std::optional<xla::ifrt::Shape> kNoStaticShape;
  const auto static_shape_of =
      [&](int i) -> const std::optional<xla::ifrt::Shape>& {
    return static_shapes.empty() ? kNoStaticShape : static_shapes[i];
  };

  absl::flat_hash_map<xla::ifrt::Device*, DeviceBatchData> batches_by_device;
  std::vector<std::optional<PjRtBuffers>> batched_inputs(inputs.size());
  std::vector<xla::ifrt::ShardingRef> effective_shardings(inputs.size());
  std::vector<int> fallback_input_indices;
  fallback_input_indices.reserve(inputs.size());

  std::vector<xla::ifrt::ArrayRef> results(inputs.size());

  // Phase 1: Classify inputs into per-device batches or fallback paths.
  std::vector<int> dynamic_dma_input_indices;
  for (int i = 0; i < inputs.size(); ++i) {
    const std::optional<xla::ifrt::Shape>& static_shape = static_shape_of(i);
    if (IsDynamicDmaInput(inputs[i], static_shape)) {
      dynamic_dma_input_indices.push_back(i);
      continue;
    }
    TF_ASSIGN_OR_RETURN(
        std::optional<BatchPlan> batch_plan,
        GetBatchPlanIfEligible(inputs[i], input_shardings[i], static_shape,
                               client_, *thread_pool_));
    if (!batch_plan.has_value()) {
      if (static_shape.has_value()) {
        return absl::UnimplementedError(absl::StrCat(
            "Input of shape ", inputs[i].shape(), " and dtype ",
            inputs[i].dtype(), " bound to static shape ", *static_shape,
            " is not eligible for a batched transfer (string or packed dtype, "
            "no elements, or non-default memory kind), and the default "
            "transfer path does not support static shapes."));
      }
      fallback_input_indices.push_back(i);
      continue;
    }

    effective_shardings[i] = std::move(batch_plan->effective_sharding);
    PjRtBuffers& slots =
        batched_inputs[i].emplace(batch_plan->shard_targets.size());

    for (int k = 0; k < batch_plan->shard_targets.size(); ++k) {
      ShardTarget& target = batch_plan->shard_targets[k];
      DeviceBatchData& batch = batches_by_device[target.device];
      batch.entries.push_back({
          .output_slot = &slots[k],
          .source_buffer = std::move(target.source_buffer),
      });
      batch.shapes.push_back(std::move(target.shape));
    }
  }

  // Phase 2: Execute per-device batched allocations and async DMA transfers.
  xla::PjRtClient* pjrt_client = client_->pjrt_client();

  for (auto& [device, batch_data] : batches_by_device) {
    xla::ifrt::PjRtCompatibleDevice* pjrt_compatible_device =
        xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(device);
    if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
      return absl::InternalError("Target device is not PJRT compatible.");
    }
    TF_ASSIGN_OR_RETURN(
        xla::PjRtMemorySpace * memory_space,
        pjrt_compatible_device->pjrt_device()->default_memory_space());

    std::vector<DeviceBatchEntry>& batched_entries = batch_data.entries;
    const absl::InlinedVector<xla::Shape, 4>& batched_shapes =
        batch_data.shapes;

    // Use a shared_ptr so we can keep the manager alive until all
    // transfers for this batch complete.
    std::shared_ptr<xla::PjRtClient::AsyncHostToDeviceTransferManager>
        transfer_manager;
    TF_ASSIGN_OR_RETURN(transfer_manager,
                        pjrt_client->CreateBuffersForAsyncHostToDevice(
                            batched_shapes, memory_space));

    int num_batched = batched_entries.size();

    // 1. Retrieve Loop: Get all PjRtBuffer handles for this device and store
    // them directly into their target output slots.
    for (int b = 0; b < num_batched; ++b) {
      std::unique_ptr<xla::PjRtBuffer> buffer =
          transfer_manager->RetrieveBuffer(b);
      if (!buffer) return absl::InternalError("Failed to retrieve PJRT buffer");

      DeviceBatchEntry& entry = batched_entries[b];
      *entry.output_slot = std::shared_ptr<xla::PjRtBuffer>(buffer.release());
    }

    // 2. Transfer Loop: Start asynchronous DMA transfers for all buffers on
    // this device.
    for (int b = 0; b < num_batched; ++b) {
      DeviceBatchEntry& entry = batched_entries[b];
      auto literal = std::make_unique<xla::BorrowingLiteral>(
          static_cast<const char*>(entry.source_buffer.raw_data().data()),
          batched_shapes[b]);
      xla::BorrowingLiteral* literal_ptr = literal.get();

      // Capture input (kept alive by copy/refcount), literal (safely moved),
      // and transfer_manager (kept alive by shared_ptr). This prevents the host
      // buffers and backend managers from being destroyed while the transfer is
      // still physically ongoing in the background, and keeps the
      // transfer_manager's internally owned buffer alive until completion.
      auto on_done = [input = std::move(entry.source_buffer),
                      literal = std::move(literal),
                      transfer_manager]() mutable {};

      if (absl::Status st = transfer_manager->TransferLiteralToBuffer(
              b, *literal_ptr, std::move(on_done));
          !st.ok()) {
        transfer_manager->SetBufferError(b, st);
      }
    }
  }

  // Phase 3: Assemble IFRT Arrays for all batched inputs.
  for (int i = 0; i < inputs.size(); ++i) {
    if (!batched_inputs[i].has_value()) {
      continue;
    }
    PjRtBuffers& buffers = *batched_inputs[i];
    xla::ifrt::PjRtArray::PjRtBuffers pjrt_buffers(buffers.begin(),
                                                   buffers.end());
    TF_ASSIGN_OR_RETURN(
        tsl::RCReference<xla::ifrt::PjRtArray> array,
        xla::ifrt::PjRtArray::Create(
            client_, inputs[i].dtype(), inputs[i].shape(),
            effective_shardings[i], std::move(pjrt_buffers),
            /*layout=*/std::shared_ptr<const xla::PjRtLayout>()));
    results[i] = tsl::RCReference<xla::ifrt::Array>(std::move(array));
  }

  // Phase 4: Concurrently initiate default transfers for fallback inputs
  // (e.g. strings, sub-byte types, 0-element tensors, non-default memory).
  for (int idx : fallback_input_indices) {
    TF_ASSIGN_OR_RETURN(
        results[idx],
        sharding_internal::MakeArrayFromHostBuffer(
            *client_, inputs[idx], device_list_, input_shardings[idx],
            *thread_pool_));
  }

  // Phase 5: Dynamic DMA transfers for inputs bound to a larger static shape.
  for (int idx : dynamic_dma_input_indices) {
    TF_ASSIGN_OR_RETURN(results[idx], ExecuteDynamicDmaTransfer(
                                          inputs[idx], input_shardings[idx],
                                          *static_shapes[idx]));
  }

  return results;
}

absl::StatusOr<xla::ifrt::ArrayRef>
BatchedH2DTransferExecutor::ExecuteDynamicDmaTransfer(
    const HostBuffer& input, const xla::ifrt::ShardingRef& sharding,
    const xla::ifrt::Shape& static_shape) const {
  // `IsDynamicDmaInput()` only holds for arrays with a dimension 0 to shrink.
  TF_RET_CHECK(!static_shape.dims().empty() && !input.shape().dims().empty())
      << "Dynamic DMA input must have rank >= 1, got static shape "
      << static_shape << " and data shape " << input.shape();
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "BatchedH2DTransferExecutor::ExecuteDynamicDmaTransfer",
        {{"shape", absl::StrCat(input.shape())},
         {"static_shape", absl::StrCat(static_shape)}});
  });

  // Dynamic DMA linearizes the host data into the device layout of the static
  // shape and copies it to the front of a device buffer allocated for the
  // static shape. That only works for plain numeric data (no strings, no
  // packed sub-byte storage) in the device's default memory.
  if (input.dtype().kind() == xla::ifrt::DType::kString ||
      input.dtype() != input.storage_dtype()) {
    return absl::UnimplementedError(
        absl::StrCat("Dynamic DMA does not support inputs of dtype ",
                     input.dtype(), " (input of shape ", input.shape(),
                     " bound to static shape ", static_shape, ")."));
  }
  if (!sharding->memory_kind().is_default()) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA only supports the default memory kind, but the input of "
        "shape ",
        input.shape(), " bound to static shape ", static_shape,
        " has sharding ", *sharding, "."));
  }
  TF_ASSIGN_OR_RETURN(std::optional<SingleDeviceTarget> target,
                      GetSingleDeviceTarget(sharding, client_));
  if (!target.has_value()) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA only supports single-device inputs, but the input of "
        "shape ",
        input.shape(), " bound to static shape ", static_shape,
        " has sharding ", *sharding, "."));
  }
  auto* common_client =
      dynamic_cast<xla::CommonPjRtClient*>(client_->pjrt_client());
  if (common_client == nullptr) {
    return absl::UnimplementedError(
        "Dynamic DMA requires a PjRt client derived from "
        "xla::CommonPjRtClient.");
  }
  auto* pjrt_compatible_device =
      xla::ifrt::dyn_cast_or_null<xla::ifrt::PjRtCompatibleDevice>(
          target->device);
  if (!pjrt_compatible_device || !pjrt_compatible_device->pjrt_device()) {
    return absl::InternalError("Target device is not PJRT compatible.");
  }
  TF_ASSIGN_OR_RETURN(
      xla::PjRtMemorySpace * memory_space,
      pjrt_compatible_device->pjrt_device()->default_memory_space());
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(input.dtype()));

  // The device buffer: allocated for the static shape in the device's default
  // layout for it, exactly as the regular batched path (and
  // `CreateBuffersForAsyncHostToDevice`) would allocate a buffer of that shape.
  TF_ASSIGN_OR_RETURN(
      xla::Shape device_shape,
      common_client->MakeDefaultShapeForMemorySpace(
          memory_space,
          xla::ShapeUtil::MakeShape(primitive_type, static_shape.dims()),
          /*layout=*/nullptr));
  // The host data is linearized as if it were an array of its own (smaller)
  // shape in the layout of the device buffer, so it lands as a prefix of the
  // static-shape buffer. `HostBuffer::SetStaticShape()` guarantees the data
  // shape only differs along dimension 0, and dimension 0 must be the
  // major-most dimension of the layout for the prefix property to hold.
  if (device_shape.layout().minor_to_major().back() != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA requires dimension 0 to be the major-most dimension of "
        "the device layout, but the device shape for static shape ",
        static_shape, " is ", device_shape.ToString(/*print_layout=*/true),
        "."));
  }
  xla::Shape host_device_shape = device_shape;
  host_device_shape.set_dimensions(0, input.shape().dims()[0]);

  TF_ASSIGN_OR_RETURN(
      int64_t on_device_bytes,
      common_client->GetOnDeviceBytesCount(memory_space, device_shape));
  TF_ASSIGN_OR_RETURN(int64_t transferred_bytes,
                      common_client->GetDmaByteCount(host_device_shape));
  TF_ASSIGN_OR_RETURN(
      xla::PjRtRawBufferRef raw_buffer,
      common_client->AllocateRawBuffer(memory_space, on_device_bytes,
                                       /*retry_on_oom=*/true,
                                       /*allocate_after=*/{}));

  xla::PjRtDeviceEventRef definition_event;
  if (input.shape().num_elements() == 0) {
    TF_ASSIGN_OR_RETURN(definition_event,
                        raw_buffer->MakeAllocationReadyEvent());
  } else {
    std::optional<absl::Span<const int64_t>> byte_strides;
    if (input.byte_strides().has_value()) {
      byte_strides = absl::MakeConstSpan(*input.byte_strides());
    }
    // Keeps the host data alive until the client is done reading it.
    HostBuffer keep_alive = input;
    TF_ASSIGN_OR_RETURN(
        definition_event,
        common_client->LinearizeIntoImpl(
            input.base(), primitive_type, input.shape().dims(), byte_strides,
            xla::PjRtClient::HostBufferSemantics::
                kImmutableUntilTransferCompletes,
            /*on_done_with_host_buffer=*/
            [keep_alive = std::move(keep_alive)]() {}, host_device_shape,
            /*dynamic_sizes=*/{}, raw_buffer));
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> buffer,
      common_client->DefineBuffer(std::move(device_shape), memory_space,
                                  std::move(raw_buffer),
                                  {std::move(definition_event)}));

  dynamic_dma_bytes_counter->GetCell("allocated")->IncrementBy(on_device_bytes);
  dynamic_dma_bytes_counter->GetCell("transferred")
      ->IncrementBy(transferred_bytes);

  xla::ifrt::PjRtArray::PjRtBuffers pjrt_buffers;
  pjrt_buffers.push_back(std::shared_ptr<xla::PjRtBuffer>(buffer.release()));
  TF_ASSIGN_OR_RETURN(
      tsl::RCReference<xla::ifrt::PjRtArray> array,
      xla::ifrt::PjRtArray::Create(
          client_, input.dtype(), static_shape,
          std::move(target->effective_sharding), std::move(pjrt_buffers),
          /*layout=*/std::shared_ptr<const xla::PjRtLayout>()));
  return tsl::RCReference<xla::ifrt::Array>(std::move(array));
}

absl::StatusOr<std::unique_ptr<H2DTransferExecutor>>
BatchedH2DTransferExecutorFactory::Create(
    xla::ifrt::Client* client, xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  return BatchedH2DTransferExecutor::Create(
      client, std::move(device_list), thread_pool, options_);
}

}  // namespace xla_serving

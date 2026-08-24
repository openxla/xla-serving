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

// Enable definition of Eigen::ThreadPoolDevice instead of just declaration.

#define EIGEN_USE_THREADS

#include "runtime/core/sharding_utils.h"  // from @neptune

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "absl/container/btree_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"
#include "runtime/core/device_buffer.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/host_buffer_type_utils.h"  // from @neptune
#include "runtime/core/sharding_utils_core.h"  // from @neptune
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/index.h"  // from @xla
#include "xla/python/ifrt/index_domain.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/status_macros.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune::sharding_internal {
namespace {

// Shard the given `input_buffer` into equal shapes of slices.
//
// `num_partitions_per_axis` specifies the number of partitions along
// each axis (dimension).
//
// `num_replicas` specifies the number of replication for each partitioned
// sliced buffer.
//
// `devices` contains a list of devices flattend into the following
// order: [slice0][replicate0], [slice0][replicate1], ..., [slice1][replicate0],
// [slice1][replicate1], ...
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
SplitAndCreateArraysFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_buffer,
    const std::vector<int32_t>& num_partitions_per_axis, int num_replicas,
    const std::vector<xla::ifrt::Device*>& devices,
    const tsl::thread::ThreadPool& thread_pool) {
  TF_ASSIGN_OR_RETURN(auto split_buffers,
                      DisassembleHostBuffer(
                          input_buffer, num_partitions_per_axis, thread_pool));

  if (split_buffers.size() * num_replicas != devices.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expect ", devices.size(), " but got ",
                     split_buffers.size(), " x ", num_replicas));
  }

  std::vector<xla::ifrt::ArrayRef> arrays;
  arrays.reserve(devices.size());
  auto device_iter = devices.begin();
  for (int slice_idx = 0; slice_idx < split_buffers.size(); ++slice_idx) {
    auto& host_buffer = split_buffers[slice_idx];

    for (int i = 0; i < num_replicas; ++i) {
      VLOG(2) << "Make array for buffer slice " << slice_idx << " at "
              << host_buffer.base();
      if (device_iter == devices.end()) {
        return absl::InternalError(
            absl::StrCat("Missing Device ", i, " for slice ", slice_idx));
      }
      auto single_device_sharding = xla::ifrt::SingleDeviceSharding::Create(
          *device_iter, xla::ifrt::MemoryKind());

      TF_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef array,
                          ifrt_client.MakeArrayFromHostBuffer(
                              host_buffer.base(), host_buffer.dtype(),
                              std::move(host_buffer.shape()), std::nullopt,
                              std::move(single_device_sharding),
                              /*layout=*/nullptr,
                              xla::ifrt::Client::HostBufferSemantics::
                                  kImmutableUntilTransferCompletes,
                              [host_buffer, slice_idx]() {
                                // Keep host_buffer alive
                                VLOG(2) << "Done with host buffer for slice "
                                        << slice_idx << " at "
                                        << host_buffer.base();
                              }));
      arrays.push_back(std::move(array));
      device_iter++;
    }
  }
  return arrays;
}

absl::StatusOr<int> VerifyIndexDomainsAndGetReplicas(
    absl::Span<xla::ifrt::IndexDomain> index_domains,
    const xla::ifrt::Shape& host_buffer_shape) {
  if (index_domains.size() <= 1) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expect multiple index domains but got ", index_domains.size()));
  }

  for (auto index_domain = index_domains.begin();
       index_domain < index_domains.end(); ++index_domain) {
    VLOG(4) << "index_domain: " << *index_domain;
    if (index_domain->shape().dims().size() !=
        host_buffer_shape.dims().size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expect equal rank of ", host_buffer_shape.dims().size(),
                       " but got ", index_domain->shape().dims().size()));
    }
  }

  // Only support equal shape for all index domains
  auto first_index_domain = index_domains.begin();
  for (auto index_domain = index_domains.begin() + 1;
       index_domain < index_domains.end(); ++index_domain) {
    if (first_index_domain->shape() != index_domain->shape()) {
      return absl::UnimplementedError(
          absl::StrCat("Expect equal shape of ", first_index_domain->shape(),
                       " but got ", index_domain->shape()));
    }
  }

  // Verify that each `IndexDomain` appear the same `num_replica` times. Since
  // shapes are the same for all `IndexDomain`, this also implies each `origin`
  // appear `num_replica` times.
  struct IndexDomainLexicographicalComparator {
    bool operator()(const xla::ifrt::IndexDomain& a,
                    const xla::ifrt::IndexDomain& b) const {
      return std::lexicographical_compare(
          a.origin().elements().begin(), a.origin().elements().end(),
          b.origin().elements().begin(), b.origin().elements().end());
    }
  };
  absl::btree_map<xla::ifrt::IndexDomain, int,
                  IndexDomainLexicographicalComparator>
      index_domain_counts;
  for (const auto& index_domain : index_domains) {
    index_domain_counts[index_domain]++;
  }

  std::vector<xla::ifrt::IndexDomain> unique_index_domains;
  unique_index_domains.reserve(index_domain_counts.size());
  int num_replicas = index_domain_counts.begin()->second;
  for (const auto& [index_domain, count] : index_domain_counts) {
    if (count != num_replicas) {
      return absl::FailedPreconditionError(
          absl::StrCat("Expected ", num_replicas, " replicas for ",
                       index_domain, " but got ", count, " replicas"));
    }
    unique_index_domains.push_back(index_domain);
  }

  // Verify that distances of between origins of neighbouring `IndexDomain`
  // bounded by shape. Note that unique_indexx_domains are already in sorted
  // order.
  auto prev_iter = unique_index_domains.begin();
  auto next_iter = unique_index_domains.begin() + 1;
  const auto& bounded_box = first_index_domain->shape();
  while (prev_iter != unique_index_domains.end() &&
         next_iter != unique_index_domains.end()) {
    xla::ifrt::Index offset = next_iter->origin() - prev_iter->origin();
    for (int dim = 0; dim < bounded_box.dims().size(); ++dim) {
      if (std::abs(offset.elements()[dim]) != bounded_box.dims()[dim] &&
          offset.elements()[dim] != 0) {
        return absl::FailedPreconditionError(absl::StrCat(
            "IndexDomains should not have gap or overlap, but got ", *prev_iter,
            " and ", *next_iter, " that have offset of ", offset));
      }
    }
    prev_iter = next_iter;
    next_iter++;
  }

  // Verify the last `IndexDomain`'s upper end of the bound matches with the
  // HostBuffer shape. Together with the above check, this provides an
  // approximation to the following two assumptions:
  // 1. the union of all IndexDomain covers the entire global shape array with
  // no gaps.
  // 2. no two index_domain have any overlap.
  std::vector<int64_t> bounded_shape;
  const auto& last_index_domain = unique_index_domains.back();
  bounded_shape.reserve(last_index_domain.shape().dims().size());
  for (int d = 0; d < last_index_domain.shape().dims().size(); ++d) {
    bounded_shape.push_back(last_index_domain.origin().elements()[d] +
                            last_index_domain.shape().dims()[d]);
  }

  if (xla::ifrt::Shape(bounded_shape) !=
      xla::ifrt::Shape(host_buffer_shape.dims())) {
    return absl::FailedPreconditionError(absl::StrCat(
        "IndexDomain ", last_index_domain,
        " does not overlap with HostBuffer shape ", host_buffer_shape));
  }

  return num_replicas;
}

// A simple wrapper function to create ifrt array for one single device.
absl::StatusOr<xla::ifrt::ArrayRef> CreateArrayFromHostBufferForSingleDevice(
    xla::ifrt::Client& ifrt_client, const HostBuffer& host_buffer,
    xla::ifrt::Device* device, xla::ifrt::Client::HostBufferSemantics semantics,
    xla::ifrt::MemoryKind memory_kind) {
  VLOG(2) << "Make single device array for buffer slice at "
          << host_buffer.base() << " with semantics "
          << static_cast<int>(semantics);
  auto single_device_sharding =
      xla::ifrt::SingleDeviceSharding::Create(device, memory_kind);

  return ifrt_client.MakeArrayFromHostBuffer(
      host_buffer.base(), host_buffer.dtype(), host_buffer.shape(),
      std::nullopt, std::move(single_device_sharding),
      /*layout=*/nullptr, semantics, [host_buffer]() {
        // Keep host_buffer alive
        VLOG(2) << "Done with single device host buffer for slice " << " at "
                << host_buffer.base();
      });
}

absl::StatusOr<xla::ifrt::ArrayRef> MakeAssembledArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_buffer,
    xla::ifrt::ShardingRef sharding,
    const xla::ifrt::DeviceListRef& device_list,
    const tsl::thread::ThreadPool& thread_pool) {
  // TODO: use xla::HloSharding to identifying sharding axis.

  VLOG(2) << "Assembling arrays by sharding " << sharding;

  TF_ASSIGN_OR_RETURN(auto index_domains,
                      sharding->IndexDomains(
                          input_buffer.shape(),
                          xla::ifrt::SingleDeviceShardSemantics::kAllShards));

  TF_ASSIGN_OR_RETURN(int index_domain_replicas,
                      VerifyIndexDomainsAndGetReplicas(
                          absl::MakeSpan(index_domains), input_buffer.shape()));

  const auto& first_index_domain = index_domains.begin();
  std::vector<int32_t> num_partitions_per_axis;
  int total_num_partitions = 1;
  num_partitions_per_axis.reserve(input_buffer.shape().dims().size());
  for (int dim = 0; dim < input_buffer.shape().dims().size(); ++dim) {
    int target_size = first_index_domain->shape().dims()[dim];
    int dim_size = input_buffer.shape().dims()[dim];
    if (dim_size % target_size != 0) {
      return absl::FailedPreconditionError(
          absl::StrCat("Only support even sharding, but input buffer shape ",
                       input_buffer.shape(), " not even splittable to ",
                       first_index_domain->shape()));
    }
    int num_partitions = dim_size / target_size;
    total_num_partitions *= num_partitions;
    num_partitions_per_axis.push_back(num_partitions);
  }

  if (total_num_partitions > sharding->devices()->size() ||
      sharding->devices()->size() % total_num_partitions != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Number of devices ", sharding->devices()->size(),
        " not a multiple of number of partitions", total_num_partitions));
  }

  // Assume index domains are non-overlapping and each index domain appears
  // exactly num_replicates times. This allows us to rely on
  // lexicographical sorting to replicate slices in the correct order.
  int num_replicas = sharding->devices()->size() / total_num_partitions;
  if (index_domain_replicas != num_replicas) {
    return absl::FailedPreconditionError(
        absl::StrCat("IndexDomain indicates ", index_domain_replicas,
                     " replicas, but got ", num_replicas, " replicas"));
  }

  // Sorted the IndexDomain and devices from major to minor dimenson. For
  // example, a two dimension IndexDomain will be ordered by [0, 0], [0, 1], [1,
  // 0], [1, 1].
  // This is O(n*log(n)) vs looking for devices individually which is O(n^2).
  struct IndexDomainDevice {
    xla::ifrt::IndexDomain index_domain;
    xla::ifrt::Device* device;
    // The index of this `device`/`index_domain` in the
    // sharding.devices/index_domains.
    int original_shard_index;
  };
  std::vector<IndexDomainDevice> index_domain_devices;
  index_domain_devices.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    index_domain_devices.push_back(
        {index_domains[i], sharding->devices()->devices()[i], i});
  }
  std::sort(index_domain_devices.begin(), index_domain_devices.end(),
            [](const IndexDomainDevice& a, const IndexDomainDevice& b) {
              return std::lexicographical_compare(
                  a.index_domain.origin().elements().begin(),
                  a.index_domain.origin().elements().end(),
                  b.index_domain.origin().elements().begin(),
                  b.index_domain.origin().elements().end());
            });
  // Now the devices is in order.
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(index_domain_devices.size());
  std::vector<int> original_device_indices;
  original_device_indices.reserve(index_domain_devices.size());
  for (auto& [index_domain, device, original_device_index] :
       index_domain_devices) {
    devices.push_back(device);
    original_device_indices.push_back(original_device_index);
    VLOG(3) << "Device " << device->ToString();
  }

  TF_ASSIGN_OR_RETURN(auto arrays,
                      SplitAndCreateArraysFromHostBuffer(
                          ifrt_client, input_buffer, num_partitions_per_axis,
                          num_replicas, devices, thread_pool));

  // Re-arranged arrays back to original device order
  std::vector<xla::ifrt::ArrayRef> rearranged_arrays;
  rearranged_arrays.resize(arrays.size());
  for (int i = 0; i < arrays.size(); ++i) {
    rearranged_arrays[original_device_indices[i]] = std::move(arrays[i]);
  }

  return ifrt_client.AssembleArrayFromSingleDeviceArrays(
      input_buffer.dtype(), input_buffer.shape(), std::move(sharding),
      absl::MakeSpan(rearranged_arrays),
      xla::ifrt::ArrayCopySemantics::kDonateInput,
      xla::ifrt::SingleDeviceShardSemantics::kAddressableShards);
}

absl::StatusOr<tsl::Future<HostBuffer>> MakeHostBufferFromArrayHelper(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::ifrt::ShardingRef& ifrt_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool) {
  VLOG(2) << "MakeHostBufferFromArrayHelper" << input_array.dtype();
  VLOG(2) << "Create HostBuffer from array based on sharding: "
          << *ifrt_sharding;
  xla::ifrt::DType input_dtype = input_array.dtype();
  const xla::ifrt::Shape& input_shape = input_array.shape();

  bool is_replicated = false;
  bool is_single_device = false;
  if (const auto* hlo_sharding_ptr =
          xla::ifrt::dyn_cast<const xla::ifrt::HloSharding>(
              ifrt_sharding.get())) {
    is_replicated = hlo_sharding_ptr->xla_hlo_sharding().IsReplicated();
    is_single_device = hlo_sharding_ptr->xla_hlo_sharding().IsSingleDevice();
  } else if (xla::ifrt::isa<const xla::ifrt::SingleDeviceSharding>(
                 ifrt_sharding.get())) {
    is_single_device = true;
  }

  if (is_replicated) {
    VLOG(1) << "Fast path for replication";
    // fast path for replication.
    TF_ASSIGN_OR_RETURN(auto fully_replicated_array,
                        input_array.FullyReplicatedShard(
                            xla::ifrt::ArrayCopySemantics::kDonateInput));

    if (fully_replicated_array->shape() != input_shape) {
      return absl::InternalError(
          absl::StrCat("Not fully replicated output. Expected ", input_shape,
                       " but got ", fully_replicated_array->shape()));
    }

    HostBuffer output_buffer(input_dtype, input_shape);
    return fully_replicated_array
        ->CopyToHostBuffer(output_buffer.base(), std::nullopt,
                           xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
        .MapTo(std::move(output_buffer));
  } else if (is_single_device) {
    VLOG(1) << "Fast path for single device";
    TF_ASSIGN_OR_RETURN(
        std::vector<xla::ifrt::ArrayRef> disassembled_array,
        input_array.DisassembleIntoSingleDeviceArrays(
            xla::ifrt::ArrayCopySemantics::kDonateInput,
            xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

    VLOG(1) << "Fast path for single device, disassembled size: "
            << disassembled_array.size();

    int target_device_index = -1;
    // Fast path: If the physical array only has one slice, it's index 0.
    if (disassembled_array.size() == 1) {
      target_device_index = 0;
    } else if (const auto* hlo_sharding =
                   xla::ifrt::dyn_cast<const xla::ifrt::HloSharding>(
                       ifrt_sharding.get())) {
      // Edge case: Multi-slice array. Safely locate the slice matching the
      // target device ID.
      int target_device_id = hlo_sharding->xla_hlo_sharding().GetUniqueDevice();
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::Device * target_device,
          ifrt_client.LookupDevice(xla::ifrt::DeviceId(target_device_id)));

      for (int i = 0; i < disassembled_array.size(); ++i) {
        const auto& shard_devices =
            disassembled_array[i]->sharding().devices()->devices();
        if (!shard_devices.empty() && shard_devices[0] == target_device) {
          target_device_index = i;
          break;
        }
      }
    } else {
      // Catch-all for unexpected sharding types with size > 1
      target_device_index = 0;
    }

    if (target_device_index < 0 ||
        target_device_index >= disassembled_array.size()) {
      return absl::InternalError(
          absl::StrCat("Target device index ", target_device_index,
                       " out of range for disassembled array of size ",
                       disassembled_array.size()));
    }

    HostBuffer output_buffer(input_dtype, input_shape);
    return disassembled_array[target_device_index]
        ->CopyToHostBuffer(output_buffer.base(), std::nullopt,
                           xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
        .MapTo(std::move(output_buffer));
  }

  TF_ASSIGN_OR_RETURN(
      auto index_domains,
      ifrt_sharding->IndexDomains(
          input_shape, xla::ifrt::SingleDeviceShardSemantics::kAllShards));

  TF_ASSIGN_OR_RETURN(int index_domain_replicas,
                      VerifyIndexDomainsAndGetReplicas(
                          absl::MakeSpan(index_domains), input_shape));
  if (ifrt_sharding->devices()->size() % index_domain_replicas != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Number of devices ", ifrt_sharding->devices()->size(),
        " not a multiple of number of partitions", index_domain_replicas));
  }
  TF_ASSIGN_OR_RETURN(
      std::vector<xla::ifrt::ArrayRef> disassembled_array,
      input_array.DisassembleIntoSingleDeviceArrays(
          xla::ifrt::ArrayCopySemantics::kDonateInput,
          xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  if (index_domains.size() != disassembled_array.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Number of index domains ", index_domains.size(),
        " not equal to number of arrays ", disassembled_array.size()));
  }

  for (int i = 0; i < index_domains.size(); ++i) {
    if (index_domains[i].shape() != disassembled_array[i]->shape()) {
      return absl::FailedPreconditionError(absl::StrCat(
          "Index domain ", index_domains[i].shape(),
          " not equal to array shape ", disassembled_array[i]->shape()));
    }
  }

  // `VerifyIndexDomainsAndGetReplicas` already verify all index domains are of
  // equal shape.
  std::vector<int> num_concats;
  num_concats.reserve(input_shape.dims().size());
  const xla::ifrt::Shape& per_split_shape = index_domains[0].shape();
  int num_unique_slices = 1;
  for (int i = 0; i < per_split_shape.dims().size(); ++i) {
    VLOG(4) << "per_split_shape.dims()[i]: " << per_split_shape.dims()[i];
    VLOG(4) << "input_shape.dim_size(i): " << input_shape.dims()[i];
    int dim_num_concats = input_shape.dims()[i] / per_split_shape.dims()[i];
    num_unique_slices *= dim_num_concats;
    num_concats.push_back(dim_num_concats);
  }
  if (num_unique_slices * index_domain_replicas != index_domains.size()) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expect number of slices is ", index_domains.size(), " but got ",
        index_domain_replicas, " replicas of ", num_unique_slices, " slices"));
  }

  VLOG(2) << "Index domains: ";
  for (const auto& index_domain : index_domains) {
    VLOG(2) << index_domain;
  }

  // disassembled array is in device order.
  struct IndexDomainDeviceArray {
    xla::ifrt::IndexDomain index_domain;
    xla::ifrt::ArrayRef array;
  };
  std::vector<IndexDomainDeviceArray> index_domain_device_arrays;
  index_domain_device_arrays.reserve(index_domains.size());
  for (int i = 0; i < index_domains.size(); ++i) {
    index_domain_device_arrays.push_back(
        {index_domains[i], disassembled_array[i]});
  }

  std::sort(
      index_domain_device_arrays.begin(), index_domain_device_arrays.end(),
      [](const IndexDomainDeviceArray& a, const IndexDomainDeviceArray& b) {
        return std::lexicographical_compare(
            a.index_domain.origin().elements().begin(),
            a.index_domain.origin().elements().end(),
            b.index_domain.origin().elements().begin(),
            b.index_domain.origin().elements().end());
      });

  // Now filter out the duplicates.
  auto last_iter = std::unique(
      index_domain_device_arrays.begin(), index_domain_device_arrays.end(),
      [](const IndexDomainDeviceArray& a, const IndexDomainDeviceArray& b) {
        return a.index_domain.origin() == b.index_domain.origin();
      });
  index_domain_device_arrays.erase(last_iter, index_domain_device_arrays.end());

  if (index_domain_device_arrays.size() != num_unique_slices) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Expect number of unique device arrays is ", num_unique_slices,
        " but got ", index_domain_device_arrays.size()));
  }

  std::vector<tsl::Future<>> arrays_copy_status;
  std::vector<HostBuffer> input_buffers;
  input_buffers.reserve(index_domain_device_arrays.size());
  arrays_copy_status.reserve(index_domain_device_arrays.size());
  for (const auto& [index_domain, array] : index_domain_device_arrays) {
    xla::ifrt::Shape shape = index_domain.shape();
    xla::ifrt::DType dtype = array->dtype();
    HostBuffer buffer(dtype, shape);
    input_buffers.push_back(buffer);
    tsl::Future<> copy_status =
        array->CopyToHostBuffer(buffer.base(), std::nullopt,
                                xla::ifrt::ArrayCopySemantics::kAlwaysCopy);
    copy_status.OnReady([buffer](absl::Status status) {
      VLOG(1) << "Copy of buffer " << buffer.base() << " done with status "
              << status;
    });
    arrays_copy_status.push_back(std::move(copy_status));
  }

  return tsl::JoinFutures(arrays_copy_status)
      .Map(*thread_pool.AsExecutor(),
           [&thread_pool, num_concats, input_dtype, input_shape,
            input_buffers = std::move(input_buffers)] {
             return MakeHostBufferFromDisassembledBuffers(
                 input_buffers, num_concats, input_dtype, input_shape,
                 thread_pool);
           });
}

}  // namespace

tsl::Future<HostBuffer> MakeHostBufferFromArray(
    xla::ifrt::Client& ifrt_client, xla::ifrt::Array& input_array,
    const xla::ifrt::ShardingRef& ifrt_sharding,
    const xla::ifrt::DeviceListRef& device_list,
    tsl::thread::ThreadPool& thread_pool) {
  tsl::profiler::TraceMe trace_me("MakeHostBufferFromArray", /*level=*/3);
  absl::StatusOr<tsl::Future<HostBuffer>> output_future =
      MakeHostBufferFromArrayHelper(ifrt_client, input_array, ifrt_sharding,
                                    device_list, thread_pool);
  if (!output_future.ok()) {
    return tsl::Future<HostBuffer>(std::move(output_future).status());
  }
  return output_future.value();
}

absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_host_buffer,
    const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef ifrt_sharding,
    const tsl::thread::ThreadPool& thread_pool,
    xla::ifrt::Client::HostBufferSemantics semantics) {
  tsl::profiler::TraceMe trace_me("MakeArrayFromHostBuffer_DeviceList",
                                  /*level=*/3);
  if (xla::ifrt::isa<const xla::ifrt::SingleDeviceSharding>(
          ifrt_sharding.get())) {
    return CreateArrayFromHostBufferForSingleDevice(
        ifrt_client, input_host_buffer, ifrt_sharding->devices()->devices()[0],
        semantics, ifrt_sharding->memory_kind());
  }
  const xla::ifrt::HloSharding* ifrt_hlo_sharding =
      xla::ifrt::dyn_cast<const xla::ifrt::HloSharding>(ifrt_sharding.get());
  if (ifrt_hlo_sharding == nullptr) {
    return absl::InternalError(
        "Sharding must be an instance of xla::ifrt::HloSharding");
  }
  const xla::HloSharding& xla_hlo_sharding =
      ifrt_hlo_sharding->xla_hlo_sharding();
  VLOG(3) << "HloSharding properties: IsTiled=" << xla_hlo_sharding.IsTiled()
          << ", IsReplicated=" << xla_hlo_sharding.IsReplicated()
          << ", IsReplicatedOrSingleDevice="
          << xla_hlo_sharding.IsReplicatedOrSingleDevice();
  if (!xla_hlo_sharding.IsTiled() &&
      !xla_hlo_sharding.IsReplicatedOrSingleDevice()) {
    return absl::UnimplementedError(absl::StrCat(
        "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
        xla_hlo_sharding.ToString()));
  }

  VLOG(1) << "Hlo sharding: " << xla_hlo_sharding.ToString();
  VLOG(1) << "Device list size: " << device_list->size();

  if (xla_hlo_sharding.IsSingleDevice()) {
    VLOG(1) << "Single device fast path for Maximal tiled buffer";
    xla::ifrt::Device* device;
    int unique_device_id = xla_hlo_sharding.GetUniqueDevice();
    TF_ASSIGN_OR_RETURN(device, ifrt_client.LookupDevice(
                                    xla::ifrt::DeviceId(unique_device_id)));
    return CreateArrayFromHostBufferForSingleDevice(
        ifrt_client, input_host_buffer, device, semantics,
        ifrt_sharding->memory_kind());
  }

  // Fast path for replicated sharding.
  if (xla_hlo_sharding.IsReplicated()) {
    VLOG(1) << "Fast path for replicated sharding";

    xla::ifrt::DType ifrt_dtype = input_host_buffer.dtype();
    return ifrt_client.MakeArrayFromHostBuffer(
        input_host_buffer.base(), ifrt_dtype, input_host_buffer.shape(),
        std::nullopt, std::move(ifrt_sharding),
        /*layout=*/nullptr, semantics,
        [input_host_buffer]() {  // keep host_buffer alive
        });
  }

  return MakeAssembledArrayFromHostBuffer(ifrt_client, input_host_buffer,
                                          std::move(ifrt_sharding), device_list,
                                          thread_pool);
}

absl::StatusOr<xla::ifrt::ArrayRef> MakeArrayFromHostBuffer(
    xla::ifrt::Client& ifrt_client, const HostBuffer& input_host_buffer,
    absl::Span<const int> device_ids, xla::ifrt::ShardingRef ifrt_sharding,
    const tsl::thread::ThreadPool& thread_pool,
    xla::ifrt::Client::HostBufferSemantics semantics) {
  tsl::profiler::TraceMe trace_me("MakeArrayFromHostBuffer_DeviceIds",
                                  /*level=*/3);
  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      CreateDeviceList(ifrt_client, device_ids));
  return MakeArrayFromHostBuffer(ifrt_client, input_host_buffer,
                                 std::move(device_list), ifrt_sharding,
                                 thread_pool, semantics);
}

absl::StatusOr<xla::ifrt::DeviceListRef> CreateDeviceList(
    xla::ifrt::Client& ifrt_client, absl::Span<const int> device_ids) {
  if (device_ids.empty()) {
    return absl::InvalidArgumentError("device_ids cannot be empty");
  }
  std::vector<xla::ifrt::Device*> devices;
  devices.reserve(device_ids.size());
  for (auto device_id : device_ids) {
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        ifrt_client.LookupDevice(xla::ifrt::DeviceId(device_id)));
    devices.push_back(device);
  }

  return ifrt_client.MakeDeviceList(devices);
}

absl::StatusOr<std::vector<HostBuffer>> DisassembleHostBuffer(
    const HostBuffer& input_buffer,
    const std::vector<int32_t>& num_partitions_per_axis,
    const tsl::thread::ThreadPool& thread_pool) {
  Eigen::ThreadPoolDevice thread_pool_device(thread_pool.AsEigenThreadPool(),
                                             thread_pool.NumThreads());

  int64_t num_slices = 1;
  for (auto k : num_partitions_per_axis) {
    num_slices *= k;
  }

  xla::ifrt::DType array_dtype = input_buffer.dtype();
  std::vector<int32_t> paddings(num_partitions_per_axis.size(), 0);
  std::vector<HostBuffer> split_buffers;
  split_buffers.resize(num_slices);

  auto allocate_output_fn = [&](int i,
                                const xla::ifrt::Shape& output_slice_shape,
                                HostBuffer** host_buffer) {
    if (i < 0 || i >= split_buffers.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Index ", i, " out of range [0, ", split_buffers.size(), ")"));
    }
    split_buffers[i] = HostBuffer(array_dtype, output_slice_shape);
    *host_buffer = &split_buffers[i];
    return absl::OkStatus();
  };

  // Fast path for output in the simple no split case.
  auto assign_or_copy_value_fn = [&](const HostBuffer& input) -> absl::Status {
    split_buffers[0] = input;
    return absl::OkStatus();
  };

  // XlaNDSplitter only support rank (0, 8] as there is no concept of split for
  // rank 0 buffer.
  if (input_buffer.shape().dims().empty()) {
    if (split_buffers.size() != 1) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank 0 buffer only expects 1 slice but got ", split_buffers.size()));
    }
    split_buffers[0] = input_buffer;
  } else {
    switch (input_buffer.dtype().kind()) {
#define CASE_TYPE(T, DTYPE_KIND)                                               \
  case DTYPE_KIND: {                                                           \
    TF_ASSIGN_OR_RETURN(auto splitter,                                         \
                        (XlaNDSplitter<Eigen::ThreadPoolDevice, T>::Create(    \
                            num_partitions_per_axis, num_slices, paddings,     \
                            /*has_paddings=*/false)));                         \
    TF_RETURN_IF_ERROR(                                                        \
        splitter.Split(&input_buffer, "input buffer", assign_or_copy_value_fn, \
                       allocate_output_fn, thread_pool_device));               \
  } break;
      APPLY_POD_CASE(CASE_TYPE);
#undef CASE_TYPE
      default:
        return absl::InvalidArgumentError(
            absl::StrCat("Unsupported data type of ", input_buffer.dtype()));
    }
  }
  return split_buffers;
}

absl::StatusOr<HostBuffer> MakeHostBufferFromDisassembledBuffers(
    absl::Span<const HostBuffer> input_disassembled_buffers,
    const std::vector<int>& num_concats, xla::ifrt::DType output_buffer_type,
    const xla::ifrt::Shape& output_buffer_shape,
    const tsl::thread::ThreadPool& thread_pool) {
  Eigen::ThreadPoolDevice thread_pool_device(thread_pool.AsEigenThreadPool(),
                                             thread_pool.NumThreads());

  int num_slices = 1;
  for (int i = 0; i < num_concats.size(); ++i) {
    num_slices *= num_concats[i];
  }

  if (num_slices != input_disassembled_buffers.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Expect number of slices is ", input_disassembled_buffers.size(),
        " but got ", num_slices, ""));
  }
  std::vector<int32_t> paddings(num_concats.size(), 0);

  VLOG(2) << "output_buffer_type: " << output_buffer_type;
  VLOG(2) << "output_buffer_shape: " << output_buffer_shape;
  HostBuffer output_buffer(output_buffer_type, output_buffer_shape);

  auto assign_or_copy_value_fn = [&](const HostBuffer& buffer) {
    output_buffer = buffer;
    return absl::OkStatus();
  };
  auto get_output_fn = [&]() -> absl::StatusOr<HostBuffer*> {
    return &output_buffer;
  };

  std::vector<HostBuffer> disassembled_buffers;
  disassembled_buffers.resize(input_disassembled_buffers.size());
  for (int i = 0; i < input_disassembled_buffers.size(); ++i) {
    disassembled_buffers[i] = input_disassembled_buffers[i];
  }
  switch (output_buffer_type.kind()) {
#define CASE_TYPE(T, DTYPE_KIND)                                          \
  case DTYPE_KIND: {                                                      \
    TF_ASSIGN_OR_RETURN(                                                  \
        auto concatenator,                                                \
        (XlaNDConcatenator<Eigen::ThreadPoolDevice, T>::Create(           \
            num_concats, num_slices, paddings, /*has_paddings=*/false))); \
    TF_RETURN_IF_ERROR(concatenator.ComputeInternal(                      \
        disassembled_buffers, assign_or_copy_value_fn, get_output_fn,     \
        thread_pool_device));                                             \
  } break;
    APPLY_POD_CASE(CASE_TYPE)
#undef CASE_TYPE
    default:
      return absl::InvalidArgumentError(
          absl::StrCat("Unsupported data type of ", output_buffer_type));
  }
  return output_buffer;
}

absl::StatusOr<xla::ifrt::ShardingRef> ToIfrtSharding(
    xla::ifrt::Client& ifrt_client, const xla::HloSharding& hlo_sharding,
    const xla::ifrt::DeviceListRef& device_list) {
  if (!hlo_sharding.IsTiled() && !hlo_sharding.IsReplicatedOrSingleDevice()) {
    return absl::UnimplementedError(absl::StrCat(
        "Only support MAXIMAL, OTHER or REPLICATED, but got sharding : ",
        hlo_sharding.ToString()));
  }

  if (device_list->size() == 1) {
    return xla::ifrt::SingleDeviceSharding::Create(
        device_list->devices().front(), xla::ifrt::MemoryKind());
  }

  if (hlo_sharding.IsSingleDevice()) {
    VLOG(1) << "Single device fast path for Maximal tiled tensor";
    xla::ifrt::Device* device;
    int unique_device_id = hlo_sharding.GetUniqueDevice();
    TF_ASSIGN_OR_RETURN(device, ifrt_client.LookupDevice(
                                    xla::ifrt::DeviceId(unique_device_id)));
    return xla::ifrt::SingleDeviceSharding::Create(device,
                                                   xla::ifrt::MemoryKind());
  }

  return xla::ifrt::HloSharding::Create(device_list, xla::ifrt::MemoryKind(),
                                        hlo_sharding);
}

absl::StatusOr<DeviceBuffer> MakeDeviceBufferFromPinnedHost(
    const std::shared_ptr<xla::ifrt::Client>& client,
    const HostBuffer& host_buffer, const xla::ifrt::DeviceListRef& device_list,
    xla::ifrt::ShardingRef sharding,
    const tsl::thread::ThreadPool& thread_pool) {
  TF_RET_CHECK(client != nullptr) << "Client must not be null.";
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::ifrt::Sharding> pinned_sharding,
      sharding->WithDeviceAssignment(
          /*devices=*/std::nullopt,
          /*memory_kind=*/xla::ifrt::MemoryKind("pinned_host")));
  xla::ifrt::ShardingRef pinned_sharding_ref(std::move(pinned_sharding));
  TF_ASSIGN_OR_RETURN(
      xla::ifrt::ArrayRef array,
      MakeArrayFromHostBuffer(
          *client, host_buffer, device_list, pinned_sharding_ref, thread_pool,
          xla::ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));
  return DeviceBuffer{
      .ifrt_client = client,
      .ifrt_array = std::move(array),
      .sharding = std::move(pinned_sharding_ref),
      .device_list = device_list,
      .memory_kind = BufferMemoryKind::kPinnedHost,
      .owner = host_buffer.owner(),
  };
}

}  // namespace neptune::sharding_internal

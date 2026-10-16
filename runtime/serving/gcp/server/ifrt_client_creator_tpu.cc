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

#include "runtime/serving/gcp/server/ifrt_client_creator.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <string>
#include <utility>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/numeric/bits.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"  // from @xla
#include "xla/pjrt/pjrt_common.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/pjrt/plugin/plugin_names.h"  // from @xla
#include "xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace serving {

namespace {

constexpr absl::string_view kPremappedBufferSize = "premapped_buffer_size";
constexpr absl::string_view kMaximumPremappedBufferSizeForTransfersInBytes =
    "maximum_premapped_buffer_size_for_transfers_in_bytes";
constexpr absl::string_view kNumPremappedPartitions =
    "num_premapped_partitions";

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreatePjRtTpuClient(
    const IfrtClientOptions::TpuOptions& options) {
  absl::flat_hash_map<std::string, xla::PjRtValueType> create_options;
  if (options.use_premapped_buffer) {
    if (options.num_premapped_partitions_per_device <= 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "TPU num_premapped_partitions_per_device must be positive, got ",
          options.num_premapped_partitions_per_device));
    }
    TF_ASSIGN_OR_RETURN(
        std::unique_ptr<xla::PjRtTopologyDescription> tpu_topology,
        xla::GetCApiTopology(kTpuPjrtName, /*topology_name=*/""));
    TF_ASSIGN_OR_RETURN(int cores_per_host,
                        tpu_topology->CoreCountOfDefaultTypePerProcess());
    if (cores_per_host <= 0) {
      return absl::InternalError(absl::StrCat(
          "TPU cores per host must be positive, got ", cores_per_host));
    }
    int64_t premapped_buffer_size =
        static_cast<int64_t>(options.premapped_buffer_size);
    int64_t num_premapped_partitions = static_cast<int64_t>(absl::bit_ceil(
        static_cast<uint64_t>(cores_per_host) *
        static_cast<uint64_t>(options.num_premapped_partitions_per_device)));
    create_options[kPremappedBufferSize] = premapped_buffer_size;
    create_options[kMaximumPremappedBufferSizeForTransfersInBytes] =
        premapped_buffer_size;
    create_options[kNumPremappedPartitions] = num_premapped_partitions;
  }
  TF_ASSIGN_OR_RETURN(auto pjrt_client,
                      xla::GetXlaPjrtTpuClient(create_options));
  return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
}

}  // namespace

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetIfrtClient(
    absl::string_view pjrt_platform, const IfrtClientOptions& options) {
  if (pjrt_platform == "tpu") {
    return CreatePjRtTpuClient(options.tpu);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported or uncompiled PJRT platform: ", pjrt_platform));
}

}  // namespace serving
}  // namespace xla_serving

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

#include "runtime/data_loader/ocp_loader/checkpoint_loader.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @xla_serving
#include "runtime/data_loader/ocp_loader/tensorstore_loader.h"  // from @xla_serving
#include "runtime/data_loader/ocp_loader/tensorstore_utils.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/index.h"  // from @xla
#include "xla/python/ifrt/index_domain.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

// TODO: add test for this file.
namespace xla_serving {

const char _OCDBT_MANIFEST_FILE[] = "manifest.ocdbt";

nlohmann::json GetOcdbtTsLoadSpec(absl::string_view root_path,
                                  absl::string_view weight_name) {
  return GetJsonTspec(std::string(root_path),
                      /*use_ocdbt=*/true, /*use_zarr3=*/false,
                      std::string(weight_name))
      .value();
}

nlohmann::json GetStandardTsLoadSpec(absl::string_view root_path,
                                     absl::string_view weight_name) {
  return GetJsonTspec(std::string(root_path),
                      /*use_ocdbt=*/false, /*use_zarr3=*/false,
                      std::string(weight_name))
      .value();
}

absl::StatusOr<nlohmann::json> GetJsonTsLoadSpec(
    absl::string_view root_path, absl::string_view weight_name,
    CheckpointType checkpoint_type) {
  if (checkpoint_type == CheckpointType::kStandard) {
    return GetStandardTsLoadSpec(root_path, weight_name);
  } else if (checkpoint_type == CheckpointType::kOcdbt) {
    return GetOcdbtTsLoadSpec(root_path, weight_name);
  } else {
    return absl::UnimplementedError(
        absl::StrCat("Unsupported checkpoint type: ", checkpoint_type));
  }
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadCheckpoint(
    absl::string_view root_path, absl::Span<const std::string> weight_names,
    absl::Span<const ArraySpec> weight_specs,
    const xla::ifrt::DeviceListRef& device_list, xla::ifrt::Client& client,
    CheckpointType checkpoint_type) {
  absl::flat_hash_map<std::string, size_t> weight_name_counts;
  for (const auto& name : weight_names) {
    weight_name_counts[name]++;
  }
  std::vector<std::string> duplicate_names_with_counts;
  for (const auto& [name, count] : weight_name_counts) {
    if (count > 1) {
      duplicate_names_with_counts.push_back(
          absl::StrCat(name, " (count: ", count, ")"));
    }
  }
  if (!duplicate_names_with_counts.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Duplicate weight names found: ",
                     absl::StrJoin(duplicate_names_with_counts, ", ")));
  }
  // TODO: change hardcoded values to be configurable.
  if (weight_names.size() != weight_specs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "weight_names and weight_specsmust have the same size, but got ",
        weight_names.size(), " and ", weight_specs.size()));
  }

  std::vector<TensorStoreLoadSpec> ts_load_specs;
  ts_load_specs.reserve(weight_specs.size());

  for (int i = 0; i < weight_specs.size(); ++i) {
    const ArraySpec& weight_spec = weight_specs[i];
    TensorStoreLoadSpec ts_load_spec;
    // TODO: add support for other dtypes.
    xla::PrimitiveType primitive_type;
    TF_ASSIGN_OR_RETURN(primitive_type,
                        xla::ifrt::ToPrimitiveType(weight_spec.dtype));
    ts_load_spec.dtype = static_cast<int64_t>(primitive_type);

    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(weight_spec.sharding));
    auto ifrt_sharding = xla::ifrt::HloSharding::Create(
        device_list, weight_spec.memory_kind.value_or(xla::ifrt::MemoryKind()),
        hlo_sharding);

    VLOG(4) << "ifrt_sharding: " << *ifrt_sharding;

    TF_ASSIGN_OR_RETURN(auto index_domains,
                        ifrt_sharding->IndexDomains(
                            weight_spec.shape,
                            xla::ifrt::SingleDeviceShardSemantics::kAllShards));

    for (int j = 0; j < index_domains.size(); ++j) {
      auto& index_domain = index_domains[j];
      VLOG(4) << "index_domain: " << index_domain;
      auto& shard = ts_load_spec.shards.emplace_back();
      shard.device_id = device_list->devices()[j]->Id().value();
      for (int64_t index : index_domain.origin().elements()) {
        shard.index_offset.push_back(index);
      }
    }
    // TODO: assume equal split. Add error checking in case that is
    // not the case.
    auto json_shape_array = nlohmann::json::array();

    for (auto dim : weight_spec.shape.dims()) {
      ts_load_spec.global_shape.push_back(dim);
      json_shape_array.push_back(dim);
    }
    for (auto dim : index_domains[0].shape().dims()) {
      ts_load_spec.per_shard_shape.push_back(dim);
    }

    TF_ASSIGN_OR_RETURN(
        nlohmann::json ts_load_spec_json,
        GetJsonTsLoadSpec(root_path, weight_names[i], checkpoint_type));

    ts_load_spec.json_ts_spec_string = ts_load_spec_json.dump();

    if (weight_spec.layout.has_value()) {
      ts_load_spec.layout = weight_spec.layout.value().ToProto();
    }
    ts_load_spec.memory_kind = weight_spec.memory_kind;

    VLOG(3) << "ts_load_spec: " << ts_load_spec;
    ts_load_specs.push_back(std::move(ts_load_spec));
  }

  LoadFromTensorstoreExtraOptions opts;
  // Assume each weight has a full path
  return LoadFromTensorstore(&client, absl::MakeSpan(ts_load_specs),
                             /*base_dir=*/root_path, opts, checkpoint_type);
}

bool IsOcdbtCheckpoint(const std::string& path) {
  // Determines whether a checkpoint uses OCDBT format.
  return tsl::Env::Default()->FileExists(
             tsl::io::JoinPath(path, _OCDBT_MANIFEST_FILE)) == absl::OkStatus();
}

absl::StatusOr<std::vector<std::pair<std::string, xla::ifrt::ArrayRef>>>
LoadCheckpointFromOrbax(absl::string_view checkpoint_path,
                        absl::Span<const std::string> weight_names,
                        absl::Span<const ArraySpec> weight_specs,
                        const xla::ifrt::DeviceListRef& device_list,
                        xla::ifrt::Client& client) {
  CheckpointType checkpoint_type =
      IsOcdbtCheckpoint(std::string(checkpoint_path))
          ? CheckpointType::kOcdbt
          : CheckpointType::kStandard;
  TF_ASSIGN_OR_RETURN(
      auto weights, LoadCheckpoint(checkpoint_path, weight_names, weight_specs,
                                   device_list, client, checkpoint_type));
  std::vector<std::pair<std::string, xla::ifrt::ArrayRef>> named_weights;
  named_weights.reserve(weights.size());
  for (size_t i = 0; i < weights.size(); ++i) {
    named_weights.emplace_back(weight_names[i], std::move(weights[i]));
  }
  return named_weights;
}

}  // namespace xla_serving

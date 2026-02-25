#include "third_party/neptune/runtime/data_loader/ocp_loader/checkpoint_loader.h"

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "file/base/path.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/str_join.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/australis/google/ifrt/ifrt_australis.proto.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "third_party/neptune/runtime/data_loader/ocp_loader/common_types.h"
#include "third_party/neptune/runtime/data_loader/ocp_loader/tensorstore_loader.h"
#include "third_party/neptune/runtime/data_loader/ocp_loader/tensorstore_utils.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "third_party/tensorflow/compiler/xla/hlo/ir/hlo_sharding.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device_list.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/index.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/index_domain.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/memory.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/shape.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/sharding.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_dtype.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/xla_sharding.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"

// TODO(qidichen): add test for this file.
namespace neptune {

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

  std::vector<aux::TensorStoreLoadSpec> ts_load_specs;
  ts_load_specs.reserve(weight_specs.size());

  for (int i = 0; i < weight_specs.size(); ++i) {
    const ArraySpec& weight_spec = weight_specs[i];
    aux::TensorStoreLoadSpec ts_load_spec;
    // TODO: add support for other dtypes.
    xla::PrimitiveType primitive_type;
    TF_ASSIGN_OR_RETURN(primitive_type,
                        xla::ifrt::ToPrimitiveType(weight_spec.dtype));
    ts_load_spec.set_dtype(static_cast<int64_t>(primitive_type));

    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(weight_spec.sharding));
    auto ifrt_sharding = xla::ifrt::HloSharding::Create(
        device_list, xla::ifrt::MemoryKind(), hlo_sharding);

    VLOG(4) << "ifrt_sharding: " << ifrt_sharding->DebugString();

    TF_ASSIGN_OR_RETURN(auto index_domains,
                        ifrt_sharding->IndexDomains(
                            weight_spec.shape,
                            xla::ifrt::SingleDeviceShardSemantics::kAllShards));

    for (int j = 0; j < index_domains.size(); ++j) {
      auto& index_domain = index_domains[j];
      VLOG(4) << "index_domain: " << index_domain.DebugString();
      auto* shard = ts_load_spec.add_shards();
      shard->set_device_id(device_list->devices()[j]->Id().value());
      for (int64_t index : index_domain.origin().elements()) {
        shard->add_index_offset(index);
      }
    }
    // TODO: assume equal split. Add error checking in case that is
    // not the case.
    auto json_shape_array = nlohmann::json::array();

    for (auto dim : weight_spec.shape.dims()) {
      ts_load_spec.add_global_shape(dim);
      json_shape_array.push_back(dim);
    }
    for (auto dim : index_domains[0].shape().dims()) {
      ts_load_spec.add_per_shard_shape(dim);
    }

    TF_ASSIGN_OR_RETURN(
        nlohmann::json ts_load_spec_json,
        GetJsonTsLoadSpec(root_path, weight_names[i], checkpoint_type));

    ts_load_spec.set_json_ts_spec_string(ts_load_spec_json.dump());

    if (weight_spec.layout.has_value()) {
      ts_load_spec.mutable_layout()->CopyFrom(
          weight_spec.layout.value().ToProto());
    }

    VLOG(3) << "ts_load_spec: " << ts_load_spec;
    ts_load_specs.push_back(ts_load_spec);
  }

  auto load_specs = proto2::RepeatedPtrField<aux::TensorStoreLoadSpec>(
      ts_load_specs.begin(), ts_load_specs.end());

  // Assume each weight has a full path
  return LoadFromTensorstore(&client, load_specs, /*base_dir=*/root_path,
                             /*opts=*/{}, checkpoint_type);
}

bool IsOcdbtCheckpoint(const std::string& path) {
  // Determines whether a checkpoint uses OCDBT format.
  return tsl::Env::Default()->FileExists(
             file::JoinPath(path, _OCDBT_MANIFEST_FILE)) == absl::OkStatus();
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

}  // namespace neptune

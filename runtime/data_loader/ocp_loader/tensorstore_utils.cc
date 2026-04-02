#include "runtime/data_loader/ocp_loader/tensorstore_utils.h"  // from @neptune

#include <cstdint>
#include <filesystem>  // NOLINT(build/c++17)
#include <optional>
#include <regex>  // NOLINT(build/c++11)
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune {

namespace {

const char kDefaultDriver[] = "file";
// BEGIN GOOGLE-INTERNAL
const char kDefaultGfileDriver[] = "gfile";
// END GOOGLE-INTERNAL
const char kProcessSubdirPrefix[] = "ocdbt.process_";
const char kOcdbtProcessIdRe[] = "[A-Za-z0-9]+";
const char kGcsPathRe[] = "^gs://([^/]*)/(.*)$";

constexpr const char* ZARR_VER2 = "zarr";
constexpr const char* ZARR_VER3 = "zarr3";

}  // namespace

absl::StatusOr<nlohmann::json> GetJsonTspec(
    const std::string& parent_dir, bool use_ocdbt, bool use_zarr3,
    const std::optional<std::string>& name,
    const std::optional<std::variant<int, std::string>>& process_index,
    const std::optional<std::string>& metadata_key) {
  if (parent_dir.empty()) {
    return absl::InvalidArgumentError("Parent directory is not specified.");
  }
  // TODO: Support of Python epath.Path `parent_dir.as_posix()`.
  std::string directory = parent_dir;

  absl::StatusOr<nlohmann::json> kvstore_tspec =
      BuildKvstoreTspec(directory, name, use_ocdbt, process_index);
  if (!kvstore_tspec.ok()) {
    return kvstore_tspec.status();
  }

  nlohmann::json tspec;
  tspec["driver"] = use_zarr3 ? ZARR_VER3 : ZARR_VER2;
  tspec["kvstore"] = kvstore_tspec.value();
  tspec["recheck_cached_data"] = false;
  tspec["recheck_cached_metadata"] = false;

  if (metadata_key.has_value()) {
    tspec["metadata_key"] = metadata_key.value();
  }
  return tspec;
}

absl::StatusOr<nlohmann::json> GetKvstoreForGcs(const std::string& ckpt_path) {
  std::regex gcs_path_regex(kGcsPathRe);
  std::smatch match;
  if (!std::regex_match(ckpt_path, match, gcs_path_regex)) {
    return absl::InvalidArgumentError(
        "The ckpt_path should contain the bucket name and the "
        "file path inside the bucket. Got: " +
        ckpt_path);
  }
  std::string gcs_bucket = match[1];
  std::string path_without_bucket = match[2];
  nlohmann::json json_spec = {
      {"driver", "gcs"}, {"bucket", gcs_bucket}, {"path", path_without_bucket}};
  return json_spec;
}

absl::StatusOr<nlohmann::json> BuildKvstoreTspec(
    const std::string& directory, const std::optional<std::string>& name,
    bool use_ocdbt,
    const std::optional<std::variant<int, std::string>>& process_id) {
  std::string default_driver = std::string(kDefaultDriver);
  // BEGIN GOOGLE-INTERNAL
  default_driver = kDefaultGfileDriver;
  // END GOOGLE-INTERNAL

  // TODO: Consider using tensorflow/tsl/platform/file_system.h to
  // replace `std::filesystem`.
  std::string normalized_directory =
      std::filesystem::path(directory).lexically_normal().string();
  normalized_directory =
      std::regex_replace(normalized_directory, std::regex("gs:/"), "gs://");

  bool is_gcs_path = absl::StartsWith(normalized_directory, "gs://");

  nlohmann::json kv_spec;

  if (use_ocdbt) {
    if (!is_gcs_path &&
        !std::filesystem::path(normalized_directory).is_absolute()) {
      return absl::InvalidArgumentError(
          "Checkpoint path should be absolute. Got " + normalized_directory);
    }

    if (process_id.has_value()) {
      std::string process_id_str;
      if (std::holds_alternative<int>(*process_id)) {
        process_id_str = std::to_string(std::get<int>(*process_id));
      } else {
        process_id_str = std::get<std::string>(*process_id);
      }
      std::regex process_id_regex(kOcdbtProcessIdRe);
      if (!std::regex_match(process_id_str, process_id_regex)) {
        return absl::InvalidArgumentError("process_id must conform to " +
                                          std::string(kOcdbtProcessIdRe) +
                                          " pattern, got " + process_id_str);
      }
      normalized_directory =
          (std::filesystem::path(normalized_directory) /
           (std::string(kProcessSubdirPrefix) + process_id_str))
              .string();
    }

    nlohmann::json base_driver_spec;
    if (is_gcs_path) {
      base_driver_spec = normalized_directory;
    } else {
      base_driver_spec = nlohmann::json{{"driver", default_driver},
                                        {"path", normalized_directory}};
    }

    kv_spec["driver"] = "ocdbt";
    kv_spec["base"] = base_driver_spec;

    if (name.has_value()) {
      kv_spec["path"] = *name;
    }

    kv_spec["experimental_read_coalescing_threshold_bytes"] = 1000000;
    kv_spec["experimental_read_coalescing_merged_bytes"] = 500000000000;
    kv_spec["experimental_read_coalescing_interval"] = "1ms";
    kv_spec["cache_pool"] = "cache_pool#ocdbt";
  } else {
    std::string path =
        name.has_value()
            ? (std::filesystem::path(normalized_directory) / *name).string()
            : normalized_directory;

    if (is_gcs_path) {
      absl::StatusOr<nlohmann::json> gcs_kvstore = GetKvstoreForGcs(path);
      if (!gcs_kvstore.ok()) {
        return gcs_kvstore.status();
      }
      kv_spec = std::move(gcs_kvstore).value();
    } else {
      kv_spec["driver"] = default_driver;
      kv_spec["path"] = path;
    }
  }

  return kv_spec;
}

absl::StatusOr<xla::ifrt::ArrayRef> AssembleOpaqueIfrtArray(
    xla::ifrt::Client* ifrt_client,
    const absl::Span<xla::ifrt::ArrayRef>& ifrt_arrays,
    const std::vector<int64_t>& global_shape) {
  if (ifrt_arrays.empty()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot assemble 0 arrays into an array"));
  }
  absl::InlinedVector<xla::ifrt::Device*, 1> devices;
  devices.reserve(ifrt_arrays.size());
  std::vector<xla::ifrt::Shape> shapes;
  shapes.reserve(ifrt_arrays.size());
  std::optional<xla::ifrt::MemoryKind> memory_kind;
  for (auto& ifrt_array : ifrt_arrays) {
    devices.push_back(ifrt_array->sharding().devices()->devices().front());
    shapes.push_back(ifrt_array->shape());
    xla::ifrt::MemoryKind this_kind = ifrt_array->sharding().memory_kind();
    if (memory_kind == std::nullopt) {
      memory_kind = this_kind;
    } else if (*memory_kind != this_kind) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "assemble arrays of different memory kind, [%v] vs [%v]",
          *memory_kind, this_kind));
    }
  }

  TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                      ifrt_client->MakeDeviceList(devices));
  return ifrt_client->AssembleArrayFromSingleDeviceArrays(
      ifrt_arrays.at(0)->dtype(), xla::ifrt::Shape(global_shape),
      xla::ifrt::ConcreteSharding::Create(
          std::move(device_list), memory_kind.value_or(xla::ifrt::MemoryKind()),
          /*shape=*/xla::ifrt::Shape(global_shape),
          /*shard_shapes=*/std::move(shapes)),
      ifrt_arrays, xla::ifrt::ArrayCopySemantics::kReuseInput,
      xla::ifrt::SingleDeviceShardSemantics::kAddressableShards);
}

}  // namespace neptune

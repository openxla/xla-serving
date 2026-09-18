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

#include "runtime/data_loader/ocp_loader/tensorstore_utils.h"  // from @xla_serving

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
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tensorstore/data_type.h"  // from @tensorstore

namespace xla_serving {

namespace {

const char kDefaultDriver[] = "file";
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

absl::StatusOr<tensorstore::DataType> ToTensorStoreDType(
    xla::ifrt::DType dtype) {
  static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__,
                "Big endian support not implemented");

  switch (dtype.kind()) {
    case xla::ifrt::DType::kPred:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::bool_t>);
    case xla::ifrt::DType::kS2:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int2_t>);
    case xla::ifrt::DType::kS4:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int4_t>);
    case xla::ifrt::DType::kS8:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int8_t>);
    case xla::ifrt::DType::kS16:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int16_t>);
    case xla::ifrt::DType::kS32:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int32_t>);
    case xla::ifrt::DType::kS64:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::int64_t>);
    case xla::ifrt::DType::kU8:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::uint8_t>);
    case xla::ifrt::DType::kU16:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::uint16_t>);
    case xla::ifrt::DType::kU32:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::uint32_t>);
    case xla::ifrt::DType::kU64:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::uint64_t>);
    case xla::ifrt::DType::kBF16:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::bfloat16_t>);
    case xla::ifrt::DType::kF16:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float16_t>);
    case xla::ifrt::DType::kF32:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float32_t>);
    case xla::ifrt::DType::kF64:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float64_t>);
    case xla::ifrt::DType::kC64:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::complex64_t>);
    case xla::ifrt::DType::kC128:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::complex128_t>);
    case xla::ifrt::DType::kF8E4M3FN:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3fn_t>);
    case xla::ifrt::DType::kF8E4M3B11FNUZ:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3b11fnuz_t>);
    case xla::ifrt::DType::kF8E4M3FNUZ:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float8_e4m3fnuz_t>);
    case xla::ifrt::DType::kF8E5M2:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float8_e5m2_t>);
    case xla::ifrt::DType::kF8E5M2FNUZ:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float8_e5m2fnuz_t>);
    case xla::ifrt::DType::kF4E2M1FN:
      return tensorstore::DataType(
          tensorstore::dtype_v<tensorstore::dtypes::float4_e2m1fn_t>);
    default:
      return absl::UnimplementedError(absl::StrCat(
          "Unable to convert IFRT dtype ", dtype, " to TensorStore dtype"));
  }
}

}  // namespace xla_serving

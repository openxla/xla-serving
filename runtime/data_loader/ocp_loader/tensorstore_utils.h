#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_UTILS_H_

#include <cstdint>
#include <optional>
#include <string>
#include <variant>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "data_type.h"  // from @tensorstore

namespace neptune {

/*
 * Constructs a JSON specification for the serialization process based on
 * provided parameters.
 *
 * Parameters:
 *
 * - parent_dir: A path providing location where all files under the same
 * checkpoint should be saved under. All params provided to a given TypeHandler
 * should have the same `parent_dir`. The parent_dir is assumed to be a
 * directory.
 *
 * - name:  Name of the parameter.
 *
 * - use_zarr3: If True, use Zarr ver3 otherwise ver2.
 *
 * - process_index: An optional parameter, which can be an integer or string,
 * representing an index or identifier for the process. If provided, it is used
 * when constructing the key-value store specification.
 *
 * - metadata_key: An optional string representing the key for accessing
 * metadata in the storage system. If provided, it is included in the resulting
 * JSON specification.
 *
 * Returns:
 * - A `StatusOr<nlohmann::json>` object. On success, it contains the
 * constructed JSON specification. On failure, it returns an appropriate error
 * status (e.g., if required fields like `path` or `parent_dir` are missing).
 */
absl::StatusOr<nlohmann::json> GetJsonTspec(
    const std::string& parent_dir, bool use_ocdbt, bool use_zarr3 = false,
    const std::optional<std::string>& name = std::nullopt,
    const std::optional<std::variant<int, std::string>>& process_index =
        std::nullopt,
    const std::optional<std::string>& metadata_key = std::nullopt);

/*
 * Constructs a specification for a Tensorstore KvStore.
 *
 * Parameters:
 * - directory: The base path (key prefix) of the KvStore, used by the
 * underlying file driver.
 * - name: The name (filename) of the parameter (optional). When no
 * `name` is specified, it allows building a tspec that refers to the whole
 * kvstore rather than one key within the kvstore.
 * - use_ocdbt: A flag indicating whether to use the OCDBT driver (default:
 * true).
 * - process_id: Used only with the OCDBT driver. If provided, the path
 *   `{directory}/ocdbt.process_{process_id}` is used as the base path. If a
 * string, it must conform to the pattern `[A-Za-z0-9]+` (optional).
 *
 * Returns:
 * A Tensorstore "KvStore" specification in dictionary (JSON) form.
 */
absl::StatusOr<nlohmann::json>  BuildKvstoreTspec(
    const std::string& directory,
    const std::optional<std::string>& name = std::nullopt,
    bool use_ocdbt = true,
    const std::optional<std::variant<int, std::string>>& process_id =
        std::nullopt);

// Constructs a single ifrt array from multiple ifrt_arrays with
// the partially opaque sharding (ConcreteSharding). The global_shape is the
// shape of the assembled array across all devices.
absl::StatusOr<xla::ifrt::ArrayRef> AssembleOpaqueIfrtArray(
    xla::ifrt::Client* ifrt_client,
    const absl::Span<xla::ifrt::ArrayRef>& ifrt_arrays,
    const std::vector<int64_t>& global_shape);

// Converts an IFRT data type to a TensorStore data type.
absl::StatusOr<tensorstore::DataType> ToTensorStoreDType(
    xla::ifrt::DType dtype);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_UTILS_H_

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace neptune {

// Loads an Array from tensorstore.
struct TensorStoreLoadSpec {
  struct Shard {
    int64_t device_id;
    std::vector<int64_t> index_offset;

    template <typename Sink>
    friend void AbslStringify(Sink& sink, const Shard& shard) {
      absl::Format(&sink, "Shard(device_id=%d, index_offset=[%s])",
                   shard.device_id, absl::StrJoin(shard.index_offset, ", "));
    }
  };
  std::string json_ts_spec_string;
  std::optional<xla::OpSharding> op_sharding;
  std::vector<int64_t> per_shard_shape;
  std::vector<int64_t> global_shape;
  int64_t dtype;  // TODO: xla.PrimitiveType
  std::vector<Shard> shards;

  // Name of the array that this spec is for. This field is intended for
  // debugging purposes, e.g. it's optional and doesn't guarantee uniqueness.
  std::optional<std::string> array_name;

  // The desired layout for the loaded array on devices.
  std::optional<xla::LayoutProto> layout;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TensorStoreLoadSpec& spec) {
    absl::Format(
        &sink,
        "TensorStoreLoadSpec(json_ts_spec_string=%s, op_sharding=%s, "
        "per_shard_shape=[%s], global_shape=[%s], dtype=%d, shards=[%s], "
        "array_name=%s, layout=%s)",
        spec.json_ts_spec_string,
        spec.op_sharding.has_value() ? spec.op_sharding->DebugString() : "null",
        absl::StrJoin(spec.per_shard_shape, ", "),
        absl::StrJoin(spec.global_shape, ", "), spec.dtype,
        absl::StrJoin(spec.shards, ", "),
        spec.array_name.has_value() ? *spec.array_name : "null",
        spec.layout.has_value() ? spec.layout->DebugString() : "null");
  }
};

// TensorStoreLoader(ifrt_client).Load(), but with fastpaths for different
// backends.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client, absl::Span<const TensorStoreLoadSpec> specs,
    absl::string_view base_dir, aux::LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

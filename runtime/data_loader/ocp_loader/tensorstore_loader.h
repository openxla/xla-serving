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

#ifndef XLA_SERVING_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_
#define XLA_SERVING_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

#include <cstdint>
#include <optional>
#include <string>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tensorstore/context.h"  // from @tensorstore

namespace xla_serving {

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

  // The desired memory kind for the loaded array on devices (e.g. pinned_host).
  std::optional<xla::ifrt::MemoryKind> memory_kind;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const TensorStoreLoadSpec& spec) {
    absl::Format(
        &sink,
        "TensorStoreLoadSpec(json_ts_spec_string=%s, op_sharding=%s, "
        "per_shard_shape=[%s], global_shape=[%s], dtype=%d, shards=[%s], "
        "array_name=%s, layout=%s, memory_kind=%s)",
        spec.json_ts_spec_string,
        spec.op_sharding.has_value() ? spec.op_sharding->DebugString() : "null",
        absl::StrJoin(spec.per_shard_shape, ", "),
        absl::StrJoin(spec.global_shape, ", "), spec.dtype,
        absl::StrJoin(spec.shards, ", "),
        spec.array_name.has_value() ? *spec.array_name : "null",
        spec.layout.has_value() ? spec.layout->DebugString() : "null",
        spec.memory_kind.has_value() ? absl::StrCat(*spec.memory_kind)
                                     : "null");
  }
};

struct LoadFromTensorstoreExtraOptions {
  absl::Duration timeout = absl::InfiniteDuration();
  std::optional<xla::ifrt::MemoryKind> memory_kind = std::nullopt;
  std::optional<nlohmann::json> tensorstore_context_spec = std::nullopt;
};

// Shared context for loading multiple ifrt::Arrays from tensorstore.
class TensorStoreLoader {
 public:
  // ifrt_client must outlive
  explicit TensorStoreLoader(
      xla::ifrt::Client* ifrt_client,
      std::optional<tensorstore::Context> context = std::nullopt,
      xla::ifrt::MemoryKind memory_kind = xla::ifrt::MemoryKind())
      : ifrt_client_(ifrt_client),
        context_(context.value_or(tensorstore::Context())),
        memory_kind_(std::move(memory_kind)) {}

  // Loads a single array from tensorstore.
  absl::StatusOr<xla::ifrt::ArrayRef> LoadSingleArray(
      const TensorStoreLoadSpec& proto_spec, absl::string_view base_dir = "",
      int64_t* total_transfer = nullptr);

  absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> Load(
      absl::Span<const TensorStoreLoadSpec> specs,
      absl::string_view base_dir = "", int64_t* total_transfer = nullptr);

 private:
  xla::ifrt::Client* ifrt_client_;
  tensorstore::Context context_;
  xla::ifrt::MemoryKind memory_kind_;
};

// Uses TensorStoreLoader to load arrays from tensorstore.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client, absl::Span<const TensorStoreLoadSpec> specs,
    absl::string_view base_dir, LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

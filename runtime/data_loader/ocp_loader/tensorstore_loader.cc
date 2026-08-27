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

#include "runtime/data_loader/ocp_loader/tensorstore_loader.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/cleanup/cleanup.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/blocking_counter.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "nlohmann/json.hpp"  // from @nlohmann_json
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "runtime/data_loader/ocp_loader/shared.h"  // from @neptune
#include "runtime/data_loader/ocp_loader/tensorstore_utils.h"  // from @neptune
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/layout.h"  // from @xla
#include "xla/overflow_util.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/with_user_context.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_layout.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl
#include "tensorstore/array.h"  // from @tensorstore
#include "tensorstore/context.h"  // from @tensorstore
#include "tensorstore/contiguous_layout.h"  // from @tensorstore
#include "tensorstore/data_type.h"  // from @tensorstore
#include "tensorstore/index.h"  // from @tensorstore
#include "tensorstore/index_space/dim_expression.h"  // from @tensorstore
#include "tensorstore/index_space/index_domain.h"  // from @tensorstore
#include "tensorstore/open.h"  // from @tensorstore
#include "tensorstore/open_mode.h"  // from @tensorstore
#include "tensorstore/tensorstore.h"  // from @tensorstore
#include "tensorstore/util/element_pointer.h"  // from @tensorstore
#include "tensorstore/util/result.h"  // from @tensorstore

namespace neptune {
namespace {

bool IsSubdirectory(absl::string_view parent, absl::string_view child) {
  if (parent.empty()) {
    return false;
  }
  std::string clean_parent = tsl::io::CleanPath(parent);
  std::string clean_child = tsl::io::CleanPath(child);

  if (clean_parent == clean_child) {
    return true;
  }
  if (absl::StartsWith(clean_child, clean_parent)) {
    if (clean_parent.back() == '/' || clean_child[clean_parent.size()] == '/') {
      return true;
    }
  }
  return false;
}

constexpr int kTensorStoreLoaderThreadPoolSize = 32;

tsl::thread::ThreadPool* GetLoaderThreadPool() {
  static absl::NoDestructor<tsl::thread::ThreadPool> thread_pool(
      tsl::Env::Default(), "TensorStoreLoaderPool",
      kTensorStoreLoaderThreadPoolSize);
  return thread_pool.get();
}

// Rewrites the given TensorStore spec to use an absolute path with `base_dir`
// as the base directory.
absl::StatusOr<std::string> ResolveRelativePathAndApplyDefaults(
    absl::string_view ts_spec, absl::string_view base_dir) {
  auto json_ts_spec = ::nlohmann::json::parse(ts_spec, nullptr, false);
  if (json_ts_spec.is_discarded()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Failed to parse TensorStore spec JSON: ", ts_spec));
  }

  if (!json_ts_spec.contains("kvstore")) {
    return absl::InvalidArgumentError("TensorStore spec must have a kvstore.");
  }
  auto& kvstore = json_ts_spec.at("kvstore");

  std::string old_path;
  // Join the base_dir with the path in the kvstore if the path in the kvstore
  // is relative. Otherwise, use the path in the kvstore.
  ::nlohmann::json& path_container =
      kvstore.contains("base") ? kvstore["base"] : kvstore;
  old_path = path_container.value("path", "");

  if (tsl::io::IsAbsolutePath(old_path)) {
    if (!IsSubdirectory(base_dir, old_path)) {
      return absl::InvalidArgumentError(
          "Absolute paths are not allowed in TensorStore specs for security "
          "reasons: " +
          old_path);
    }
  } else if (!old_path.empty()) {
    path_container["path"] = tsl::io::JoinPath(base_dir, old_path);
  }
  if (old_path.empty()) {
    return std::string(ts_spec);
  }
  if (base_dir.empty() && !tsl::io::IsAbsolutePath(old_path)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot resolve relative checkpoint path with an empty "
                     "base_dir since the old path is not an absolute path: ",
                     old_path));
  }

  if (kvstore.value("driver", "") == "ocdbt") {
    kvstore.update({
        // References the cache pool name defined in `tensorstore::Context`
        // initialized by `LoadFromTensorstore`.
        {"cache_pool", "cache_pool#ocdbt"},

        // Enable read coalescing. Uses the same default as Orbax Checkpoint.
        {"experimental_read_coalescing_threshold_bytes", 1000000},
        {"experimental_read_coalescing_merged_bytes", 500000000000},
        {"experimental_read_coalescing_interval", "1ms"},
    });
  }

  // Always assume that data/metadata is fresh and use cached ones if possible
  // because serving checkpoints are immutable.
  json_ts_spec.update({
      {"recheck_cached_data", false},
      {"recheck_cached_metadata", false},
  });

  return json_ts_spec.dump();
}

absl::StatusOr<tensorstore::DataType> GetTensorStoreDataTypeForPrimitiveType(
    xla::PrimitiveType type) {
  TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(type));
  return ToTensorStoreDType(ifrt_dtype);
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorStoreDefaultImpl(
    xla::ifrt::Client* ifrt_client, absl::Span<const TensorStoreLoadSpec> specs,
    absl::string_view base_dir, LoadFromTensorstoreExtraOptions opts,
    std::optional<tensorstore::Context> context) {
  if (!context.has_value()) {
    if (!opts.tensorstore_context_spec.has_value()) {
      opts.tensorstore_context_spec = nlohmann::json::object();
    }
    if (!opts.tensorstore_context_spec->contains("cache_pool#ocdbt")) {
      // Default cache pool for OCDBT.
      (*opts.tensorstore_context_spec)["cache_pool#ocdbt"] = {
          {"total_bytes_limit", 100LL << 20},  // 100MiB
      };
    }
    auto custom_context =
        tensorstore::Context::FromJson(*opts.tensorstore_context_spec);
    if (!custom_context.ok()) {
      return custom_context.status();
    }
    context = *std::move(custom_context);
  }

  TensorStoreLoader loader(ifrt_client, std::move(context),
                           opts.memory_kind.value_or(xla::ifrt::MemoryKind()));
  return loader.Load(specs, base_dir);
}

}  // namespace

absl::StatusOr<xla::ifrt::ArrayRef> TensorStoreLoader::LoadSingleArray(
    const TensorStoreLoadSpec& spec, absl::string_view base_dir,
    int64_t* total_transfer) {
  std::vector<xla::ifrt::ArrayRef> ifrt_arrays;
  std::vector<int64_t> global_shape(spec.global_shape.begin(),
                                    spec.global_shape.end());

  // Verify that shape dimensions and transfer sizes do not cause overflows.
  auto dtype = static_cast<xla::PrimitiveType>(spec.dtype);
  int64_t total_shard_size = xla::ShapeUtil::ByteSizeOfPrimitiveType(dtype);
  for (int64_t shard_dim_size : spec.per_shard_shape) {
    auto [next_shard_size, overflow] =
        xla::OverflowSafeMultiply(total_shard_size, shard_dim_size);
    if (overflow) {
      return absl::InvalidArgumentError(
          "Integer overflow calculating total_shard_size");
    }
    total_shard_size = next_shard_size;
  }

  TF_ASSIGN_OR_RETURN(auto ts_spec, ResolveRelativePathAndApplyDefaults(
                                        spec.json_ts_spec_string, base_dir));
  auto json_ts_spec = ::nlohmann::json::parse(ts_spec,
                                              /*cb*/ nullptr,
                                              /*allow_allow_exceptions=*/false);

  auto tmp =
      tensorstore::Open(json_ts_spec, context_, tensorstore::OpenMode::open,
                        tensorstore::ReadWriteMode::read);
  TF_RETURN_IF_ERROR(tmp.result().status());

  int64_t xfer_size = 0;
  for (const auto& shard : spec.shards) {
    TF_ASSIGN_OR_RETURN(
        auto* dest_device,
        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(shard.device_id)));
    if (!dest_device->IsAddressable()) {
      continue;
    }

    std::optional<int64_t> safe_xfer_size =
        xla::OverflowSafeAdd(xfer_size, total_shard_size);
    if (!safe_xfer_size.has_value()) {
      return absl::InvalidArgumentError(
          "Integer overflow calculating xfer_size");
    }
    xfer_size = *safe_xfer_size;

    std::vector<tensorstore::Index> per_shard_shape(
        spec.per_shard_shape.begin(), spec.per_shard_shape.end());
    std::vector<tensorstore::Index> index_offset(shard.index_offset.begin(),
                                                 shard.index_offset.end());
    TF_ASSIGN_OR_RETURN(auto ifrt_dtype, xla::ifrt::ToDType(dtype));
    std::unique_ptr<uint8_t[]> data(new uint8_t[total_shard_size]);

    auto read_domain = tensorstore::IndexDomain(per_shard_shape) |
                       tensorstore::AllDims().TranslateTo(index_offset);
    auto target_domain = tensorstore::IndexDomain(per_shard_shape);

    TF_ASSIGN_OR_RETURN(auto data_type,
                        GetTensorStoreDataTypeForPrimitiveType(dtype));
    auto target_array = tensorstore::UnownedToShared(tensorstore::Array<void>(
        tensorstore::ElementPointer<void>(reinterpret_cast<void*>(&data[0]),
                                          data_type),
        per_shard_shape, tensorstore::row_major_order));
    tensorstore::TensorStore<> input_store = tmp.result().value();

    TF_RETURN_IF_ERROR(tensorstore::Read(input_store | *read_domain,
                                         target_array | target_domain)
                           .result()
                           .status());

    std::shared_ptr<xla::ifrt::PjRtLayout> layout_ref;
    if (spec.layout.has_value()) {
      TF_ASSIGN_OR_RETURN(xla::Layout layout,
                          xla::Layout::FromProto(spec.layout.value()));
      std::shared_ptr<xla::PjRtLayout> pjrt_layout =
          std::make_shared<xla::PjRtLayout>(std::move(layout));
      layout_ref = xla::ifrt::PjRtLayout::Create(std::move(pjrt_layout));
    }

    TF_ASSIGN_OR_RETURN(
        auto ifrt_array,
        ifrt_client_->MakeArrayFromHostBuffer(
            data.get(), ifrt_dtype, xla::ifrt::Shape(per_shard_shape),
            /*byte_stride=*/std::nullopt,
            xla::ifrt::SingleDeviceSharding::Create(dest_device, memory_kind_),
            std::move(layout_ref),
            xla::ifrt::Client::HostBufferSemantics::kImmutableOnlyDuringCall,
            {}));
    ifrt_arrays.push_back(ifrt_array);
  }

  if (total_transfer != nullptr) {
    std::optional<int64_t> updated_total_transfer =
        xla::OverflowSafeAdd(*total_transfer, xfer_size);
    if (!updated_total_transfer.has_value()) {
      return absl::InvalidArgumentError(
          "Integer overflow calculating total_transfer");
    }
    *total_transfer = *updated_total_transfer;
  }
  if (ifrt_arrays.size() == 1) {
    return std::move(ifrt_arrays[0]);
  }

  return AssembleOpaqueIfrtArray(ifrt_client_, absl::MakeSpan(ifrt_arrays),
                                 global_shape);
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> TensorStoreLoader::Load(
    absl::Span<const TensorStoreLoadSpec> specs, absl::string_view base_dir,
    int64_t* total_transfer) {
  std::vector<xla::ifrt::ArrayRef> results;
  std::vector<int64_t> total_transfers;
  total_transfers.resize(specs.size());
  results.resize(specs.size());

  absl::BlockingCounter count(specs.size());
  absl::Mutex status_mutex;
  absl::Status overall_status;

  for (size_t i = 0; i < specs.size(); ++i) {
    GetLoaderThreadPool()->Schedule([this, &specs, i, base_dir, &results,
                                     &total_transfers, &count, &status_mutex,
                                     &overall_status]() {
      absl::Cleanup decrement_bcount = [&count] { count.DecrementCount(); };
      {
        absl::MutexLock lock(status_mutex);
        if (!overall_status.ok()) {
          return;
        }
      }
      auto task = xla::ifrt::WithCurrentUserContext([&]() -> absl::Status {
        TF_ASSIGN_OR_RETURN(results[i], LoadSingleArray(specs[i], base_dir,
                                                        &total_transfers[i]));
        return absl::OkStatus();
      });
      absl::Status status = task();
      if (!status.ok()) {
        absl::MutexLock lock(status_mutex);
        overall_status.Update(status);
      }
    });
  }
  count.Wait();
  TF_RETURN_IF_ERROR(overall_status);

  if (total_transfer) {
    for (int64_t v : total_transfers) {
      std::optional<int64_t> updated_total_transfer =
          xla::OverflowSafeAdd(*total_transfer, v);
      if (!updated_total_transfer.has_value()) {
        return absl::InvalidArgumentError(
            "Integer overflow aggregating total_transfer");
      }
      *total_transfer = *updated_total_transfer;
    }
  }
  return results;
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client, absl::Span<const TensorStoreLoadSpec> specs,
    absl::string_view base_dir, LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type) {
  std::optional<tensorstore::Context> context;
  if (checkpoint_type == CheckpointType::kOcdbt) {
    TF_ASSIGN_OR_RETURN(tensorstore::Context ocdbt_context, GetOcdbtContext());
    context = ocdbt_context;
    auto context_spec_json = context->spec().ToJson();
    if (!context_spec_json.ok()) {
      return tensorstore::GetStatus(context_spec_json);
    }
    opts.tensorstore_context_spec = *std::move(context_spec_json);
  }

  return LoadFromTensorStoreDefaultImpl(ifrt_client, specs, base_dir,
                                        std::move(opts), std::move(context));
}

}  // namespace neptune

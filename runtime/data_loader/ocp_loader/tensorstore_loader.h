#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "third_party/australis/google/ifrt/ifrt_australis.pb.h"
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "nlohmann/json_fwd.hpp"  // from @nlohmann_json
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla

namespace neptune {

// TensorStoreLoader(ifrt_client).Load(), but with fastpaths for different
// backends.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client,
    const ::proto2::RepeatedPtrField<aux::TensorStoreLoadSpec>& specs,
    std::string_view base_dir, aux::LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

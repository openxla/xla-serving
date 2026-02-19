#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/time/time.h"
#include "third_party/australis/google/ifrt/ifrt_australis.proto.h"
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "third_party/neptune/runtime/data_loader/ocp_loader/common_types.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/memory.h"

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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_TENSORSTORE_LOADER_H_

#include <cstdint>
#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @com_google_absl
#include "absl/log/check.h"  // from @com_google_absl
#include "absl/status/status.h"  // from @com_google_absl
#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl
#include "third_party/australis/google/ifrt/ifrt_australis.pb.h"
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "third_party/json/include/nlohmann/json_fwd.hpp"
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "third_party/protobuf/repeated_ptr_field.h"
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

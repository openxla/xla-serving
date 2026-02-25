#include "third_party/neptune/runtime/data_loader/ocp_loader/tensorstore_loader.h"

#include <optional>
#include <string_view>
#include <utility>
#include <vector>

#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/australis/google/ifrt/ifrt_australis.proto.h"
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "third_party/neptune/runtime/data_loader/ocp_loader/common_types.h"
#include "third_party/neptune/runtime/data_loader/ocp_loader/shared.h"
#include "third_party/protobuf/repeated_ptr_field.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"
#include "third_party/tensorflow/compiler/xla/xla_data.proto.h"
#include "third_party/tensorstore/context.h"
#include "third_party/tensorstore/util/result.h"

namespace neptune {

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client,
    const ::proto2::RepeatedPtrField<aux::TensorStoreLoadSpec>& specs,
    std::string_view base_dir, aux::LoadFromTensorstoreExtraOptions opts,
    CheckpointType checkpoint_type) {
  std::optional<tensorstore::Context> context;
  if (checkpoint_type == CheckpointType::kOcdbt) {
    TF_ASSIGN_OR_RETURN(context, GetOcdbtContext());
    auto context_spec_json = context->spec().ToJson();
    if (!context_spec_json.ok()) {
      return tensorstore::GetStatus(context_spec_json);
    }
    opts.tensorstore_context_spec = *std::move(context_spec_json);
  }

  // Do not set `base_dir` because the path in `specs` already contains the
  // base_dir.
  return aux::LoadFromTensorstore(ifrt_client, specs, /*base_dir=*/"",
                                  std::move(opts));
}

}  // namespace neptune

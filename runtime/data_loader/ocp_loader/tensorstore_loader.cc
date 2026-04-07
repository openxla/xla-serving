#include "runtime/data_loader/ocp_loader/tensorstore_loader.h"  // from @neptune

#include <optional>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "third_party/australis/google/ifrt/tensorstore_loader.h"
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "runtime/data_loader/ocp_loader/shared.h"  // from @neptune
#include "google/protobuf/repeated_ptr_field.h"
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "context.h"  // from @tensorstore
#include "util/result.h"  // from @tensorstore

namespace neptune {
namespace {

}  // namespace

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadFromTensorstore(
    xla::ifrt::Client* ifrt_client, absl::Span<const TensorStoreLoadSpec> specs,
    absl::string_view base_dir, aux::LoadFromTensorstoreExtraOptions opts,
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

  // TODO - b/479524192: Add implementation for external users.
  return absl::UnimplementedError("To be implemented.");
}

}  // namespace neptune

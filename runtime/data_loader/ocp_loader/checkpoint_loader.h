#include "absl/types/span.h"  // from @com_google_absl
#include "runtime/data_loader/ocp_loader/common_types.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

// TODO(qidichen): move or add the unit test from MIRA.
#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_CHECKPOINT_LOADER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_CHECKPOINT_LOADER_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "absl/strings/string_view.h"  // from @com_google_absl
#include "third_party/australis/google/ifrt/ifrt_australis.pb.h"
#include "xla/layout.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla

namespace neptune {

// Struct to describe a potential param to be loaded as a weight array.
struct ArraySpec {
  xla::ifrt::DType dtype;
  xla::ifrt::Shape shape;
  xla::OpSharding sharding;
  std::optional<xla::Layout> layout;
};

// Load a list of weights.
//
// root_path: the root path of the checkpoint.
//
// weight_names: a list of weight names. Each name is a path relative to
// root_path.
//
// weight_specs: a list of ArraySpec that describes the shape and sharding of
// each weight. The size of weight_specs must be the same as weight_names.
//
// device_list: a list of devices that the weights will be loaded to.
//
// client: the ifrt client.
//
// Returns a list of ifrt arrays that contains the loaded weights.
absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> LoadCheckpoint(
    absl::string_view root_path, absl::Span<const std::string> weight_names,
    absl::Span<const ArraySpec> weight_specs,
    const xla::ifrt::DeviceListRef& device_list, xla::ifrt::Client& client,
    CheckpointType checkpoint_type);

// Load a Orbax checkpoint from the given exported model.
//
// checkpoint_path: the absolute path of the checkpoint directory.
//
// weight_names: a list of weight names. Each name is a path relative to
// checkpoint_path.
//
// weight_specs: a list of ArraySpec that describes the shape and sharding of
// each weight. The size of weight_specs must be the same as weight_names.
//
// device_list: a list of devices that the weights will be loaded to.
//
// client: the ifrt client.
//
// Returns a list of ifrt arrays that contains the loaded weights.
// TODO: Maybe return a map for easier implementation of selective
// loading.
absl::StatusOr<
    std::vector<std::pair<std::string /*weight_name*/, xla::ifrt::ArrayRef>>>
LoadCheckpointFromOrbax(absl::string_view checkpoint_path,
                        absl::Span<const std::string> weight_names,
                        absl::Span<const ArraySpec> weight_specs,
                        const xla::ifrt::DeviceListRef& device_list,
                        xla::ifrt::Client& client);

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_CHECKPOINT_LOADER_H_

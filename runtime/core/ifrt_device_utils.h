#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_

#include <optional>
#include <vector>

#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/array.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/device.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/executable.h"

namespace jsv {

// Returns the assigned IFRT devices based on the device assignment attribute.
//
// params:
//   ifrt_client: The ifrt client.
//   num_replicas: The number of replicas.
//   num_cores_per_replica: The number of cores per replica.
//
//   device_assignment: The device assignment array encoded as
//   [x0,y0,z0,core0,x1,y1,z1,core1, ...]. Optional. If not provided, the
//   devices will be assigned based on the default order returned by the IFRT
//   client.
//
// returns:
//   The assigned devices.
absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedIfrtDevices(
    const xla::ifrt::Client& ifrt_client, int num_replicas,
    int num_cores_per_replica,
    std::optional<std::vector<int>> device_assignment);

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_

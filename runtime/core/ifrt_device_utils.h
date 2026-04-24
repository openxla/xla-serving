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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_

#include <optional>
#include <vector>

#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla

namespace neptune {

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

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_IFRT_DEVICE_UTILS_H_

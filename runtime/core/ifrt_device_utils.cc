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

#include "runtime/core/ifrt_device_utils.h"  // from @xla_serving

#include <cstdint>
#include <optional>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/flat_hash_set.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "runtime/core/grid.h"  // from @xla_serving
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/attribute_map.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/host_callback.h"  // from @xla
#include "xla/service/device_assignment.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

static constexpr int kTpuTopologyRank = 4;  // (x, y, z, core).

absl::StatusOr<std::vector<xla::ifrt::Device*>> GetAssignedIfrtDevices(
    const xla::ifrt::Client& ifrt_client, int num_replicas,
    int num_cores_per_replica,
    std::optional<std::vector<int>> device_assignment) {
  const int num_devices = num_replicas * num_cores_per_replica;

  // No device coordinates from ifrt devices. This disallow the mapping from
  // device assignment attribute to ifrt devices.
  bool no_device_coordinates = false;
  for (auto* device : ifrt_client.devices()) {
    auto coords_attr = device->Attributes().Get<std::vector<int64_t>>("coords");
    auto core_on_chip_attr = device->Attributes().Get<int64_t>("core_on_chip");
    if (!coords_attr.ok() || !core_on_chip_attr.ok()) {
      no_device_coordinates = true;
      break;
    }
  }

  // If there is no device assignment attribute or no device coordinates,
  // get the default device assignment from IFRT.
  if (!device_assignment || device_assignment->empty() ||
      no_device_coordinates) {
    TF_ASSIGN_OR_RETURN(xla::DeviceAssignment xla_device_assignment,
                        ifrt_client.GetDefaultDeviceAssignment(
                            num_replicas, num_cores_per_replica));
    VLOG(3) << "Getting default device lists";
    std::vector<xla::ifrt::Device*> devices;
    devices.reserve(num_devices);
    for (int replica_idx = 0; replica_idx < num_replicas; replica_idx++) {
      for (int core_idx = 0; core_idx < num_cores_per_replica; core_idx++) {
        // This relies on the IFRT implementation of GetDefaultDeviceAssignment
        // that keeps device id the same between device assignment  and ifrt
        // device list.
        auto device_id = xla_device_assignment(replica_idx, core_idx);
        TF_ASSIGN_OR_RETURN(
            xla::ifrt::Device * device,
            ifrt_client.LookupDevice(xla::ifrt::DeviceId(device_id)));
        devices.push_back(device);
      }
    }
    return devices;
  }

  // Devices ordered as in the device assignment attribute.
  absl::flat_hash_map<GridCoords, int> devices_from_attribute;

  // Each device is encoded by [x,y,z,c] at the attribute.
  std::vector<int> coord;
  coord.reserve(kTpuTopologyRank);
  int device_index = 0;

  for (auto coord_attr : *device_assignment) {
    coord.push_back(coord_attr);
    if (coord.size() == kTpuTopologyRank) {
      devices_from_attribute.insert(
          {GridCoords(coord[0], coord[1], coord[2], coord[3]), device_index});
      device_index++;
      coord.clear();
    }
  }
  if (!coord.empty()) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment attribute is expected to be a multiple "
                     "of 4, but got ",
                     device_assignment->size()));
  }

  if (devices_from_attribute.size() != num_devices) {
    return absl::FailedPreconditionError(
        absl::StrCat("Device assignment has ", devices_from_attribute.size(),
                     " devices, but expected ", num_devices));
  }

  struct IfrtDeviceGrid {
    xla::ifrt::Device* device;
    GridCoords grid;
    int index_at_attribute;
  };
  std::vector<IfrtDeviceGrid> ifrt_devices;
  ifrt_devices.reserve(num_devices);

  for (auto* device : ifrt_client.devices()) {
    GridCoords grid;
    TF_ASSIGN_OR_RETURN(
        auto coords, device->Attributes().Get<std::vector<int64_t>>("coords"));
    TF_ASSIGN_OR_RETURN(auto core_on_chip,
                        device->Attributes().Get<int64_t>("core_on_chip"));
    grid = GridCoords(coords[0], coords[1], coords[2], core_on_chip);

    auto device_it_from_attribute = devices_from_attribute.find(grid);
    if (device_it_from_attribute == devices_from_attribute.end()) {
      VLOG(1) << "Device coordinates " << grid.ToString()
              << " does not match any TPU device assigned "
              << absl::StrJoin(*device_assignment, " ");
      continue;
    }
    ifrt_devices.push_back(
        {.device = device,
         .grid = grid,
         .index_at_attribute = device_it_from_attribute->second});
  }

  if (ifrt_devices.size() != num_devices) {
    return absl::FailedPreconditionError(absl::StrCat(
        "Match ", ifrt_devices.size(), " devices, but expected ", num_devices));
  }

  // Sort the devices by the order in the device assignment attribute.
  absl::c_sort(ifrt_devices, [&](const auto& lhs, const auto& rhs) {
    return lhs.index_at_attribute < rhs.index_at_attribute;
  });

  std::vector<xla::ifrt::Device*> result;
  result.reserve(ifrt_devices.size());
  for (auto& device_grid : ifrt_devices) {
    result.push_back(device_grid.device);
    VLOG(3) << "Device: " << device_grid.device->DebugString()
            << " is assigned";
  }
  return result;
}

}  // namespace xla_serving

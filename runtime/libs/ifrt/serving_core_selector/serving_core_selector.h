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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

#include <cstddef>
#include <string>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla

namespace neptune {

// Wrapper of a `tsl::ServingDeviceSelector`. Responsible for core selection
// during IFRT TPU execution.
class ServingCoreSelector {
 public:
  ServingCoreSelector(tsl::ServingDeviceSelector* absl_nonnull device_selector,
                      size_t num_cores)
      : device_selector_(device_selector), num_cores_(num_cores) {}

  // Reserves a device for the given `program_fingerprint`. Returns a
  // RAII object that represents the reservation.
  tsl::DeviceReservation ReserveDevice(absl::string_view program_fingerprint);

 private:
  tsl::ServingDeviceSelector* absl_nonnull device_selector_;  // Not owned.
  const size_t num_cores_;

  absl::Mutex execution_counter_mu_;
  // Key: the program fingerprint. Value: the number of times
  // the program has been executed.
  absl::flat_hash_map<std::string, size_t> execution_counter_
      ABSL_GUARDED_BY(execution_counter_mu_);
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

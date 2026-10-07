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

#ifndef XLA_SERVING_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_
#define XLA_SERVING_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

#include <cstddef>
#include <string>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla

namespace xla_serving {

// Wrapper of a `tsl::ServingDeviceSelector`. Responsible for core selection
// during IFRT TPU execution.
class ServingCoreSelector {
 public:
  // If `record_core_selection_by_fingerprint` is true, device reservations are
  // also recorded in the `/xla_serving/runtime/core_selection_count` streamz
  // counter, which is labeled by the program fingerprint. This is intended for
  // tests only: in production the fingerprint is a raw binary digest, which is
  // not valid UTF-8 and breaks consumers of streamz snapshots such as XProf.
  ServingCoreSelector(tsl::ServingDeviceSelector* absl_nonnull device_selector,
                      size_t num_cores,
                      bool record_core_selection_by_fingerprint = false)
      : device_selector_(device_selector),
        num_cores_(num_cores),
        record_core_selection_by_fingerprint_(
            record_core_selection_by_fingerprint) {}

  // Reserves a device for the given `program_fingerprint`. Returns a
  // RAII object that represents the reservation.
  tsl::DeviceReservation ReserveDevice(absl::string_view program_fingerprint);

 private:
  // Records a device reservation in the core selection streamz counters.
  void RecordCoreSelection(absl::string_view program_fingerprint,
                           int device_index);

  tsl::ServingDeviceSelector* absl_nonnull device_selector_;  // Not owned.
  const size_t num_cores_;
  const bool record_core_selection_by_fingerprint_;

  absl::Mutex execution_counter_mu_;
  // Key: the program fingerprint. Value: the number of times
  // the program has been executed.
  absl::flat_hash_map<std::string, size_t> execution_counter_
      ABSL_GUARDED_BY(execution_counter_mu_);
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_LIBS_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

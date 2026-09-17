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

#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving

#include <cstddef>

#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla

namespace xla_serving {

using ::tsl::monitoring::Counter;

static auto* core_selection_counter = Counter<2>::New(
    "/xla_serving/runtime/core_selection_count",
    "Counts device reservations made by `ServingCoreSelector`, broken "
    "down by program fingerprint and device ID.",
    "program_fingerprint", "device_id");
// This counter is a coarse-grained version of `core_selection_counter` that
// only breaks down by device ID. It's added as a workaround for unit tests
// that don't have easy access to the program fingerprint.
static auto* core_selection_counter_by_device_id_only = Counter<1>::New(
    "/xla_serving/runtime/core_selection_count_by_device_id_only",
    "Counts device reservations made by `ServingCoreSelector`, broken "
    "down by device ID only.",
    "device_id");

tsl::DeviceReservation ServingCoreSelector::ReserveDevice(
    absl::string_view program_fingerprint) {
  const size_t execution_count = [this, program_fingerprint]() {
    absl::MutexLock lock(execution_counter_mu_);
    return execution_counter_[program_fingerprint]++;
  }();
  if (execution_count < num_cores_) {
    // The first `num_cores_` executions are used for warmup. We use
    // `execution_count` as device index to iterate through all the TPU cores
    // for the given program. The null `selector` indicates that this
    // `DeviceReservation` isn't made through any selector.
    core_selection_counter
        ->GetCell(program_fingerprint, absl::StrCat(execution_count))
        ->IncrementBy(1);
    core_selection_counter_by_device_id_only
        ->GetCell(absl::StrCat(execution_count))
        ->IncrementBy(1);
    return tsl::DeviceReservation(/*device_index=*/execution_count,
                                  /*selector=*/nullptr);
  }
  tsl::DeviceReservation reservation =
      device_selector_->ReserveDevice(program_fingerprint);
  core_selection_counter
      ->GetCell(program_fingerprint, absl::StrCat(reservation.device_index()))
      ->IncrementBy(1);
  core_selection_counter_by_device_id_only
      ->GetCell(absl::StrCat(reservation.device_index()))
      ->IncrementBy(1);
  return reservation;
}

}  // namespace xla_serving

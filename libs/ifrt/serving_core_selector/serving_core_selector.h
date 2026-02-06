#ifndef THIRD_PARTY_NEPTUNE_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_
#define THIRD_PARTY_NEPTUNE_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

#include <cstddef>
#include <string>

#include "third_party/absl/base/nullability.h"
#include "third_party/absl/base/thread_annotations.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/tensorflow/compiler/xla/tsl/framework/serving_device_selector.h"

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_IFRT_SERVING_CORE_SELECTOR_SERVING_CORE_SELECTOR_H_

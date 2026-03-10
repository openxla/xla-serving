#include "runtime/core/batching/threadsafe_status.h"  // from @neptune

#include <utility>

#include "absl/status/status.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl

namespace jsv {
const absl::Status& ThreadSafeStatus::status() const& {
  absl::MutexLock lock(mutex_);
  return status_;
}

absl::Status ThreadSafeStatus::status() && {
  absl::MutexLock lock(mutex_);
  return std::move(status_);
}

void ThreadSafeStatus::Update(const absl::Status& new_status) {
  if (new_status.ok()) {
    return;
  }

  absl::MutexLock lock(mutex_);
  status_.Update(new_status);
}

void ThreadSafeStatus::Update(absl::Status&& new_status) {
  if (new_status.ok()) {
    return;
  }

  absl::MutexLock lock(mutex_);
  status_.Update(std::forward<absl::Status>(new_status));
}
}  // namespace jsv

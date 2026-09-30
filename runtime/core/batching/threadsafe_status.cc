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

#include "runtime/core/batching/threadsafe_status.h"  // from @xla_serving

#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace xla_serving {
absl::Status ThreadSafeStatus::status() const& {
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
}  // namespace xla_serving

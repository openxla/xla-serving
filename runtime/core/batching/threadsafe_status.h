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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_THREADSAFE_STATUS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_THREADSAFE_STATUS_H_

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp

namespace neptune {
// Wrapper class to allow  concurrent updates on a 'status'.
//
// Example Usage:
//   std::thread threads[2];
//   ThreadSafeStatus thread_safe_status;
//   threads[0] = std::thread([&]() {
//     status.Update(errors::Internal("internal error"));
//   });
//   threads[1] = std::thread([&]() {
//     status.Update(errors::InvalidArgument("invalid argument"));
//   });
//   threads[0].Join();
//   threads[1].Join();
//
//   NOTE:
//   When updated in a multi-threading setup, only the first error is retained.
class ThreadSafeStatus {
 public:
  const absl::Status& status() const& ABSL_LOCKS_EXCLUDED(mutex_);
  absl::Status status() && ABSL_LOCKS_EXCLUDED(mutex_);

  // Retains the first error status: replaces the current status with
  // `new_status` if `new_status` is not OK and the previous status is OK.
  void Update(const absl::Status& new_status) ABSL_LOCKS_EXCLUDED(mutex_);
  void Update(absl::Status&& new_status) ABSL_LOCKS_EXCLUDED(mutex_);

 private:
  mutable absl::Mutex mutex_;
  absl::Status status_ ABSL_GUARDED_BY(mutex_);
};
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_THREADSAFE_STATUS_H_

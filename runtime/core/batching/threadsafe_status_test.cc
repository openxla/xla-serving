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

#include <atomic>
#include <thread>  // NOLINT(build/c++11)
#include <utility>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp

namespace xla_serving {
namespace {

TEST(ThreadSafeStatus, DefaultOk) {
  ThreadSafeStatus status;
  EXPECT_OK(status.status());
}

TEST(ThreadSafeStatus, Update) {
  ThreadSafeStatus status;
  EXPECT_OK(status.status());

  status.Update(absl::FailedPreconditionError("original error"));
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);

  status.Update(absl::OkStatus());
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);

  status.Update(absl::InternalError("new error"));
  EXPECT_EQ(status.status().code(), absl::StatusCode::kFailedPrecondition);
}

TEST(ThreadSafeStatus, Move) {
  ThreadSafeStatus status;
  EXPECT_OK(std::move(status).status());
}

TEST(ThreadSafeStatus, ConcurrentReadAndUpdate) {
  ThreadSafeStatus status;
  std::atomic<bool> done{false};
  std::thread reader([&]() {
    while (!done.load(std::memory_order_relaxed)) {
      absl::Status s = status.status();
      if (!s.ok()) {
        EXPECT_EQ(s.code(), absl::StatusCode::kInternal);
      }
    }
  });

  std::thread updater([&]() {
    for (int i = 0; i < 1000; ++i) {
      status.Update(absl::InternalError("concurrent error"));
    }
    done.store(true, std::memory_order_relaxed);
  });

  updater.join();
  reader.join();
  EXPECT_EQ(status.status().code(), absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace xla_serving

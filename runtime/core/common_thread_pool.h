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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "runtime/core/thread_pool_interface.h"  // from @neptune
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

// An implementation of ThreadPoolInterface that uses a tsl::thread::ThreadPool
// as the underlying thread pool.
class CommonThreadPool : public ThreadPoolInterface {
 public:
  explicit CommonThreadPool(tsl::thread::ThreadPool* thread_pool)
      : underlying_thread_pool_(thread_pool) {};

  void AddTask(std::function<void()> task) override {
    underlying_thread_pool_->Schedule(std::move(task));
  };

  int NumThreads() const override {
    return underlying_thread_pool_->NumThreads();
  }

  absl::Status ParallelFor(
      int64_t total, const SchedulingParams& scheduling_params,
      const std::function<void(int64_t begin, int64_t end)>& fn) override {
    underlying_thread_pool_->ParallelFor(total, scheduling_params, fn);
    return absl::OkStatus();
  }

  Eigen::ThreadPoolInterface* AsEigenThreadPool() const override {
    return underlying_thread_pool_->AsEigenThreadPool();
  }

 private:
  tsl::thread::ThreadPool* underlying_thread_pool_;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_

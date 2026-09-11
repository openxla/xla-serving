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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TFRT_THREAD_POOL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TFRT_THREAD_POOL_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "runtime/core/thread_pool_interface.h"  // from @neptune
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"  // from @org_tensorflow
#include "include/tfrt/host_context/task_function.h"  // from @tf_runtime

namespace xla_serving {

using tensorflow::tfrt_stub::WorkQueueInterface;

// An implementation of ThreadPoolInterface that uses a
// tensorflow::tfrt_stub::WorkQueueInterface as the underlying work queue.
class TfrtThreadPool : public ThreadPoolInterface {
 public:
  explicit TfrtThreadPool(WorkQueueInterface* work_queue)
      : underlying_work_queue_(work_queue) {};

  void AddTask(std::function<void()> task) override {
    underlying_work_queue_->AddTask(tfrt::TaskFunction(std::move(task)));
  };

  int NumThreads() const override {
    return underlying_work_queue_->GetParallelismLevel();
  }

  // Not implemented for TfrtThreadPool.
  absl::Status ParallelFor(
      int64_t total, const SchedulingParams& scheduling_params,
      const std::function<void(int64_t begin, int64_t end)>& fn) override {
    return absl::UnimplementedError(
        "ParallelFor is not implemented for TfrtThreadPool.");
  }

  // Not implemented for TfrtThreadPool.
  Eigen::ThreadPoolInterface* AsEigenThreadPool() const override {
    LOG(WARNING) << "AsEigenThreadPool is not implemented for TfrtThreadPool.";
    return nullptr;
  }

 private:
  tensorflow::tfrt_stub::WorkQueueInterface* underlying_work_queue_;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_TENSORFLOW_TFRT_THREAD_POOL_H_

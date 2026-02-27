#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_

#include <cstdint>
#include <functional>
#include <utility>

#include "third_party/absl/status/status.h"
#include "third_party/neptune/runtime/core/thread_pool_interface.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_COMMON_THREAD_POOL_H_

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_THREAD_POOL_INTERFACE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_THREAD_POOL_INTERFACE_H_

#include <cstdint>
#include <functional>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

using SchedulingParams = tsl::thread::ThreadPool::SchedulingParams;

// This is an interface for thread pools in JSV which allows different
// implementations of thread pools, like TFRT and tsl thread pools.
class ThreadPoolInterface {
 public:
  // Enqueues a task to the underlying thread pool.
  virtual void AddTask(std::function<void()> task) = 0;

  // Returns the number of threads for parallel execution in the thread pool.
  virtual int NumThreads() const = 0;

  // Runs `fn` on `total` units of work in parallel. The number of parallel
  // tasks processing the work is determined by the scheduling parameters.
  virtual absl::Status ParallelFor(
      int64_t total, const SchedulingParams& scheduling_params,
      const std::function<void(int64_t begin, int64_t end)>& fn) = 0;

  // If the underlying ThreadPool implementation is compatible with
  // Eigen::ThreadPoolInterface, returns a non-null pointer.
  virtual Eigen::ThreadPoolInterface* absl_nonnull AsEigenThreadPool()
      const = 0;

  virtual ~ThreadPoolInterface() = default;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_THREAD_POOL_INTERFACE_H_

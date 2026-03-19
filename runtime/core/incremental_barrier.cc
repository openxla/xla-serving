#include "runtime/core/incremental_barrier.h"  // from @neptune

#include <atomic>
#include <utility>

#include "absl/functional/bind_front.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp

namespace neptune {

class InternalIncrementalBarrier {
 public:
  explicit InternalIncrementalBarrier(IncrementalBarrier::DoneCallback callback)
      : left_(1), done_callback_(std::move(callback)) {}

  void operator()() {
    DCHECK_GE(left_.load(std::memory_order_relaxed), 0);

    if (left_.fetch_sub(1, std::memory_order_acq_rel) - 1 == 0) {
      IncrementalBarrier::DoneCallback done_callback =
          std::move(done_callback_);
      delete this;
      done_callback();
    }
  }

  IncrementalBarrier::BarrierCallback Inc() {
    left_.fetch_add(1, std::memory_order_acq_rel);

    // std::bind_front is only available ever since C++20.
    return absl::bind_front(&InternalIncrementalBarrier::operator(), this);
  }

 private:
  std::atomic<int> left_;
  IncrementalBarrier::DoneCallback done_callback_;
};

IncrementalBarrier::IncrementalBarrier(DoneCallback done_callback)
    : internal_barrier_(
          new InternalIncrementalBarrier(std::move(done_callback))) {}

IncrementalBarrier::~IncrementalBarrier() { (*internal_barrier_)(); }

IncrementalBarrier::BarrierCallback IncrementalBarrier::Inc() {
  return internal_barrier_->Inc();
}

}  // namespace neptune

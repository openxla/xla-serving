#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INCREMENTAL_BARRIER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INCREMENTAL_BARRIER_H_

#include <functional>

#include "third_party/absl/functional/any_invocable.h"

namespace jsv {

class InternalIncrementalBarrier;

// BarrierClosure (see
// https://github.com/chromium/chromium/blob/master/base/barrier_closure.h)
// executes a callback after it has been invoked |num_closures| times.
// Plus, `BarrierClosure` is a continuation-passing style abstraction and self-
// deleting.

// IncrementalBarrier is a convenience class to be used in place of a barrier
// closure, which is particularly helpful (e.g. simplify code) because callers
// don't need to calculate the |num_closures| beforehand.
//
// Example Usage:
//   void MakeCalls() {
//     typedef std::function<void()> Callback;
//     typedef std::function<void(Callback)> OtherCallback;
//     Callback done_callback = ...
//     OtherCallback cb1 = ...
//     OtherCallback cb2 = ...
//     std::thread threads[2];
//     {
//         IncrementalBarrier barrier(done_callback);
//         threads[0] = std::thread(cb1(barrier.Inc());
//         threads[1] = std::thread(cb2(barrier.Inc());
//         ... at this moment, `barrier` is incremented twice, and then
//         destructed....
//     }
//     threads[0].join();
//     threads[1].join();
//   }
//
//  `done_callback` will be called when both conditions are true:
//  1) after `barrier` is destructed.
//  2) Each `BarrierCallback` returned by `Inc` is called.
// This class is thread-safe.
class IncrementalBarrier {
 public:
  typedef absl::AnyInvocable<void()> DoneCallback;
  typedef absl::AnyInvocable<void()> BarrierCallback;
  explicit IncrementalBarrier(DoneCallback callback);

  ~IncrementalBarrier();

  // Returns a BarrierCallback (std::function) that individual task call to
  // signal its completeness.
  // The returned BarrierCallback outlives this `IncrementalBarrier` instance.
  // Furthermore, each task should eventually call the returned function, or
  // else done_callback wouldn't be called.
  BarrierCallback Inc();

 private:
  // self-deleting, thereby not owned by 'IncrementalBarrier'.
  InternalIncrementalBarrier* internal_barrier_;
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INCREMENTAL_BARRIER_H_

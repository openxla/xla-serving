#include "third_party/neptune/runtime/core/incremental_barrier.h"

#include <atomic>
#include <utility>

#include "testing/base/public/benchmark.h"
#include "testing/base/public/gunit.h"
#include "third_party/absl/base/thread_annotations.h"
#include "third_party/absl/functional/bind_front.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/synchronization/mutex.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {
namespace {

// A thread-safe counter class.
class Counter {
 public:
  void Increment() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(mu_);
    ++count_;
  }

  int GetCount() ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock l(mu_);
    return count_;
  }

 private:
  absl::Mutex mu_;
  int count_ = 0;
};

TEST(IncrementalBarrierTest, RunInstantlyWhenZeroClosure) {
  Counter counter;
  EXPECT_EQ(counter.GetCount(), 0);
  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(std::move(done_callback));
    EXPECT_EQ(counter.GetCount(), 0);
  }
  EXPECT_EQ(counter.GetCount(), 1);
}

TEST(IncrementalBarrierTest, RunAfterNumClosuresOneNowTwoLater) {
  Counter counter;

  IncrementalBarrier::BarrierCallback bc1, bc2;
  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(std::move(done_callback));

    CHECK_EQ(counter.GetCount(), 0);

    bc1 = barrier.Inc();
    bc2 = barrier.Inc();

    IncrementalBarrier::BarrierCallback bc3 = barrier.Inc();
    bc3();

    CHECK_EQ(counter.GetCount(), 0);
  }

  CHECK_EQ(counter.GetCount(), 0);
  bc1();
  CHECK_EQ(counter.GetCount(), 0);
  bc2();
  CHECK_EQ(counter.GetCount(), 1);
}

TEST(IncrementalBarrierTest, RunAfterNumClosuresConcurrency) {
  const int num_closure = 100, num_thread = 2;
  std::atomic<int> schedule_count{0};
  Counter counter;

  {
    IncrementalBarrier::DoneCallback done_callback =
        absl::bind_front(&Counter::Increment, &counter);
    IncrementalBarrier barrier(std::move(done_callback));

    CHECK_EQ(counter.GetCount(), 0);

    tsl::thread::ThreadPool pool(tsl::Env::Default(), "BarrierClosure",
                                 num_thread);
    for (int i = 0; i < num_closure; ++i) {
      pool.Schedule([&barrier, &schedule_count]() {
        schedule_count.fetch_add(1);
        IncrementalBarrier::BarrierCallback bc = barrier.Inc();

        tsl::Env::Default()->SleepForMicroseconds(100);
        bc();
      });
    }

    CHECK_EQ(counter.GetCount(), 0);
  }

  CHECK_EQ(schedule_count.load(std::memory_order_relaxed), 100);
  CHECK_EQ(counter.GetCount(), 1);
}

#if defined(PLATFORM_GOOGLE)
void BM_FunctionInc(benchmark::State& state) {
  IncrementalBarrier barrier([] {});
  for (auto _ : state) {
    barrier.Inc()();
  }
}

BENCHMARK(BM_FunctionInc);
#endif  // PLATFORM_GOOGLE

}  // namespace
}  // namespace jsv

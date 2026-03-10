#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_FAKE_CLOCK_ENV_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_FAKE_CLOCK_ENV_H_

#include <cstdint>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @com_google_absl
#include "absl/synchronization/mutex.h"  // from @com_google_absl
#include "absl/synchronization/notification.h"  // from @com_google_absl
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/thread_annotations.h"  // from @tsl

namespace jsv {
namespace test_util {

// An Env implementation with a fake clock for NowMicros() and
// SleepForMicroseconds(). The clock doesn't advance on its own; it advances via
// an explicit Advance() method.
// All other Env virtual methods pass through to a wrapped Env.
class FakeClockEnv : public tsl::EnvWrapper {
 public:
  explicit FakeClockEnv(tsl::Env* wrapped);
  ~FakeClockEnv() override = default;

  // Advance the clock by a certain number of microseconds.
  void AdvanceByMicroseconds(int micros);

  // Blocks until there is a sleeping thread that is scheduled to wake up at
  // the given (absolute) time.
  void BlockUntilSleepingThread(uint64_t wake_time);

  // Blocks until there are at least num_threads sleeping.
  void BlockUntilThreadsAsleep(int num_threads);

  // Methods that this class implements.
  uint64_t NowMicros() const override;
  void SleepForMicroseconds(int64_t micros) override;

 private:
  mutable absl::Mutex mu_;

  uint64_t current_time_ ABSL_GUARDED_BY(mu_) = 0;

  struct SleepingThread {
    uint64_t wake_time;
    absl::Notification* wake_notification;
  };
  std::vector<SleepingThread> sleeping_threads_ TF_GUARDED_BY(mu_);

  FakeClockEnv(const FakeClockEnv&) = delete;
  void operator=(const FakeClockEnv&) = delete;
};

}  // namespace test_util
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_FAKE_CLOCK_ENV_H_

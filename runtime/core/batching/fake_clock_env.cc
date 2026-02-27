#include "third_party/neptune/runtime/core/batching/fake_clock_env.h"

#include <cstdint>

#include "third_party/absl/synchronization/mutex.h"
#include "third_party/absl/synchronization/notification.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/env.h"

namespace jsv {
namespace test_util {

FakeClockEnv::FakeClockEnv(Env* wrapped) : EnvWrapper(wrapped) {}

void FakeClockEnv::AdvanceByMicroseconds(int micros) {
  {
    absl::MutexLock l(mu_);
    current_time_ += micros;
    for (auto it = sleeping_threads_.begin(); it != sleeping_threads_.end();) {
      if (current_time_ >= it->wake_time) {
        it->wake_notification->Notify();
        it = sleeping_threads_.erase(it);
      } else {
        ++it;
      }
    }
  }
}

void FakeClockEnv::BlockUntilSleepingThread(uint64_t wake_time) {
  for (;;) {
    {
      absl::MutexLock l(mu_);
      for (auto it = sleeping_threads_.begin(); it != sleeping_threads_.end();
           ++it) {
        if (it->wake_time == wake_time) {
          return;
        }
      }
    }
    EnvWrapper::SleepForMicroseconds(100);
  }
}

void FakeClockEnv::BlockUntilThreadsAsleep(int num_threads) {
  for (;;) {
    {
      absl::MutexLock l(mu_);
      if (num_threads <= sleeping_threads_.size()) {
        return;
      }
    }
    EnvWrapper::SleepForMicroseconds(100);
  }
}

uint64_t FakeClockEnv::NowMicros() const {
  {
    absl::MutexLock l(mu_);
    return current_time_;
  }
}

void FakeClockEnv::SleepForMicroseconds(int64_t micros) {
  if (micros == 0) {
    return;
  }

  absl::Notification wake_notification;
  {
    absl::MutexLock l(mu_);
    sleeping_threads_.push_back({current_time_ + micros, &wake_notification});
  }
  wake_notification.WaitForNotification();
}

}  // namespace test_util
}  // namespace jsv

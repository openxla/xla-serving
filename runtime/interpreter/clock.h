#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CLOCK_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CLOCK_H_

#include "absl/base/no_destructor.h"  // from @com_google_absl
#include "absl/time/clock.h"  // from @com_google_absl
#include "absl/time/time.h"  // from @com_google_absl

namespace jsv {

class Clock {
 public:
  virtual ~Clock() = default;

  virtual absl::Time TimeNow() const = 0;
};

class RealClock : public Clock {
 public:
  ~RealClock() override = default;

  absl::Time TimeNow() const override { return absl::Now(); }

  static const RealClock* Get() {
    static const absl::NoDestructor<RealClock> real_clock;
    return &*real_clock;
  }
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CLOCK_H_

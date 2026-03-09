#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CLOCK_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CLOCK_H_

#include "third_party/absl/base/no_destructor.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"

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

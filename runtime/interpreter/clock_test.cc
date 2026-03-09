#include "third_party/neptune/runtime/interpreter/clock.h"

#include "testing/base/public/gunit.h"
#include "third_party/absl/time/clock.h"
#include "third_party/absl/time/time.h"

namespace jsv {
namespace {

TEST(RealClockTest, TimeNowReturnsCurrentTime) {
  const RealClock* real_clock = RealClock::Get();

  absl::Time time_before = absl::Now();
  absl::Time clock_time = real_clock->TimeNow();
  absl::Time time_after = absl::Now();

  // The time returned by RealClock::TimeNow() should be between the
  // absl::Now() calls made just before and just after.
  EXPECT_GE(clock_time, time_before);
  EXPECT_LE(clock_time, time_after);
}

}  // namespace
}  // namespace jsv

#include "third_party/neptune/runtime/core/batching/periodic_function.h"

#include <algorithm>
#include <cstdint>
#include <string>
#include <utility>

#include "third_party/absl/functional/any_invocable.h"
#include "third_party/absl/log/check.h"
#include "third_party/absl/log/log.h"
#include "third_party/absl/strings/str_cat.h"

namespace jsv {

PeriodicFunction::PeriodicFunction(absl::AnyInvocable<void()> function,
                                   const int64_t interval_micros,
                                   const Options& options)
    : function_(std::move(function)),
      interval_micros_([interval_micros]() -> int64_t {
        if (interval_micros < 0) {
          const std::string error =
              absl::StrCat(" The value of 'interval_micros' should be >= 0: ",
                           interval_micros, ". ");
          DCHECK(false) << error;
          LOG(WARNING) << error << "Resetting it to 0.";
          return 0;
        }
        return interval_micros;
      }()),
      options_(options) {
  thread_.reset(options_.env->StartThread(
      options_.thread_options, options_.thread_name_prefix, [this]() {
        // Record the starting time here instead of in RunLoop.  That way, if
        // there is a delay starting RunLoop, that does not affect the timing
        // of
        // the first function.  (Such a delay can often happen in tests where
        // the test simulates a large time delay immediately after calling
        // Start.)
        RunLoop(options_.env->NowMicros());
      }));
}

PeriodicFunction::~PeriodicFunction() {
  NotifyStop();

  // Waits for thread_ to complete and clean up.
  thread_.reset();
}

void PeriodicFunction::NotifyStop() {
  if (!stop_thread_.HasBeenNotified()) {
    stop_thread_.Notify();
  }
}

void PeriodicFunction::RunLoop(const int64_t start) {
  {
    if (options_.startup_delay_micros > 0) {
      const int64_t deadline = start + options_.startup_delay_micros;
      options_.env->SleepForMicroseconds(deadline - start);
    }

    while (!stop_thread_.HasBeenNotified()) {
      VLOG(3) << "Running function.";
      const int64_t begin = options_.env->NowMicros();
      function_();

      // Take the max() here to guard against time going backwards which
      // sometimes happens in multiproc machines.
      const int64_t end =
          std::max(static_cast<int64_t>(options_.env->NowMicros()), begin);

      // The deadline is relative to when the last function started.
      const int64_t deadline = begin + interval_micros_;

      // We want to sleep until 'deadline'.
      if (deadline > end) {
        if (end > begin) {
          VLOG(3) << "Reducing interval_micros from " << interval_micros_
                  << " to " << (deadline - end);
        }
        options_.env->SleepForMicroseconds(deadline - end);
      } else {
        VLOG(3) << "Function took longer than interval_micros, so not sleeping";
      }
    }
  }
}

}  // namespace jsv

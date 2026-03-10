#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_

#include "runtime/interpreter/clock.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "absl/time/time.h"  // from @com_google_absl

namespace jsv {

struct CancellationContext {
  const Clock* clock = RealClock::Get();
  // The deadline for this request.
  absl::Time deadline = absl::InfiniteFuture();

  bool IsCancelled() const { return deadline < clock->TimeNow(); }
};

struct RunContext {
  CancellationContext cancellation_context;

  static RunContext Create(const RunOptions& run_options) {
    return RunContext{
        .cancellation_context = {.deadline = run_options.deadline}};
  };
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_

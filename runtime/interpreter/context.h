// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_

#include "runtime/interpreter/clock.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune
#include "absl/time/time.h"  // from @abseil-cpp

namespace neptune {

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

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_

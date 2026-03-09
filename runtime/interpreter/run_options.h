#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_RUN_OPTIONS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

#include "third_party/absl/time/time.h"

namespace jsv {

// Per-request options for running a model. This struct is part of the public
// API, and, thus, adding new fields should be done with caution since they may
// be hard to change later.
struct RunOptions {
  // The deadline for this request.
  absl::Time deadline = absl::InfiniteFuture();
};

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

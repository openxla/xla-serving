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

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/stablehlo_utils.h"  // from @neptune
#include "runtime/interpreter/clock.h"  // from @neptune
#include "runtime/interpreter/run_options.h"  // from @neptune

namespace xla_serving {

struct CancellationContext {
  const Clock* clock = RealClock::Get();
  // The deadline for this request.
  absl::Time deadline = absl::InfiniteFuture();

  bool IsCancelled() const { return deadline < clock->TimeNow(); }
};

struct RunContext {
  CancellationContext cancellation_context;
  std::function<void(absl::flat_hash_map<std::string, HostBuffer>)>
      streamed_output_callback = nullptr;
  // TODO: Change this to uint64_t.
  int32_t request_id = -1;
  // `request_ids` is reserved for streaming use cases in batching. When
  // batching is enabled, the size of this vector will be the same as the
  // batch size. Each element in this vector corresponds to the request ID of
  // the same position in the batch.
  std::vector<int32_t> request_ids;
  std::optional<int64_t> callback_id;
  std::vector<Placement> placements;

  static RunContext Create(const RunOptions& run_options,
                           std::vector<Placement> resolved_placements = {},
                           std::optional<int64_t> callback_id = std::nullopt) {
    return RunContext{
        .cancellation_context = {.deadline = run_options.deadline},
        .streamed_output_callback = run_options.streamed_output_callback,
        .request_id = run_options.request_id,
        .callback_id = callback_id,
        .placements = std::move(resolved_placements)};
  }
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_CONTEXT_H_

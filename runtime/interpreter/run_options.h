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

#ifndef XLA_SERVING_RUNTIME_INTERPRETER_RUN_OPTIONS_H_
#define XLA_SERVING_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

#include <cstdint>
#include <functional>
#include <string>

#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving

namespace xla_serving {

// Per-request options for running a model. This struct is part of the public
// API, and, thus, adding new fields should be done with caution since they may
// be hard to change later.
struct RunOptions {
  // The deadline for this request.
  absl::Time deadline = absl::InfiniteFuture();

  // The dynamic streaming output callback for this specific request. The key in
  // the map is the name of the output tensor, and the value is the output
  // buffer.
  std::function<void(absl::flat_hash_map<std::string, HostBuffer>)>
      streamed_output_callback = nullptr;

  // Sparse named output placement map, keyed by output tensor name.
  // Controls whether each returned model output is transferred to host memory
  // (Placement::kHost, producing xla_serving::HostBuffer) or retained on accelerator
  // device memory (Placement::kDevice, wrapping xla::ifrt::Array in
  // xla_serving::DeviceBuffer). Unspecified outputs default to the manifest-defined
  // placement if present, or fall back to Placement::kHost. Note: Input
  // placement is determined automatically by the concrete type of the input
  // value (HostBuffer vs DeviceBuffer) and does not use a placement map.
  absl::flat_hash_map<std::string, Placement> output_placement_map;

  // The unique identifier for this execution request. Internal-only field,
  // populated automatically by the Interpreter. User should not set this field.
  int32_t request_id = -1;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

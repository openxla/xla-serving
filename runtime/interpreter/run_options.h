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
#include "xla/python/ifrt/array.h"  // from @xla

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

  // Pre-allocated device memory for components to write their results into,
  // keyed by the name of the component output each array backs, as it appears
  // in that component's named signature.
  //
  // Output names need no component qualifier: in a keyword-arguments based
  // pipeline, which input aliasing requires, every value a run produces lives
  // in one name-keyed map, so each output name belongs to exactly one
  // component. The interpreter resolves each name to that component before the
  // run and rejects a name that is not an output of any component, or that is
  // produced by a component whose StableHLO function was not exported for
  // input aliasing.
  //
  // Components none of whose outputs appear in the map -- which is all of them
  // by default -- allocate their own output memory, so a caller that wants
  // pre-allocation for one component in a pipeline does not have to supply
  // buffers for the rest. A component any of whose outputs appear must have
  // all of them supplied: input aliasing is all-or-nothing per execution.
  //
  // Each array must match the dtype, shape, sharding and layout of the output
  // it backs, and must be solely owned by the caller: the arrays are donated
  // to the execution and consumed by it. The results come back through the
  // normal return path, backed by this same memory.
  //
  // The names are those of component outputs, not of the pipeline's outputs,
  // because the guarantee is about the memory an accelerator execution writes
  // into. Where a component is followed by a post-processor, the aliased
  // memory holds the accelerator result feeding it rather than the final
  // value the caller gets back.
  //
  // TODO : Support pre-allocated pinned host memory.
  absl::flat_hash_map<std::string, xla::ifrt::ArrayRef>
      input_aliased_output_arrays;

  // The unique identifier for this execution request. Internal-only field,
  // populated automatically by the Interpreter. User should not set this field.
  int32_t request_id = -1;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

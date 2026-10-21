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

  // Caller-provided memory for components' results:
  // `preallocated_output_arrays` for device memory and
  // `preallocated_output_host_buffers` for host memory. Both are keyed by the
  // name of the component output each entry backs, as it appears in that
  // component's named signature, and require a keyword-arguments based
  // pipeline.
  //
  // Output names need no component qualifier: in a keyword-arguments based
  // pipeline every value a run produces lives in one name-keyed map, so each
  // output name belongs to exactly one component. The interpreter resolves
  // each name to that component before the run and rejects a name that is not
  // an output of any component.
  //
  // Any subset of a component's outputs may appear in either map, and an
  // output may appear in both: it is then computed into the array and copied
  // into the host buffer. Outputs in neither map -- which is all of them by
  // default -- get their memory from the runtime as usual.
  //
  // The names are those of component outputs, not of the pipeline's outputs,
  // because the guarantee is about the memory an accelerator execution's
  // results land in. Where a component is followed by a post-processor, that
  // memory holds the accelerator result feeding it rather than the final value
  // the caller gets back.

  // Device memory, written into directly through input aliasing. The
  // component's StableHLO function must have been exported for input aliasing.
  // Each array must match the dtype, shape, sharding and layout of the output
  // it backs, and must be solely owned by the caller.
  //
  // Lifecycle: the array is donated to the execution. It is deleted when the
  // run starts and must never be reused; from then on the runtime manages its
  // device memory. If the output is placed on the device, the returned
  // `DeviceBuffer` is backed by that memory. If it is placed on the host, the
  // result is copied off it and the memory is released after the run.
  absl::flat_hash_map<std::string, xla::ifrt::ArrayRef>
      preallocated_output_arrays;

  // Host memory, which the result is copied into from the device instead of
  // into a buffer the runtime allocates. Pinned (page-locked) memory lets the
  // device write into it directly by DMA; pageable memory works too, but is
  // staged. Works with any component. Each buffer must have the dtype and shape
  // of the output it backs, laid out densely in row-major order, and the output
  // must be replicated or placed on a single device. The output is returned on
  // the host, overriding `output_placement_map`, as a `HostBuffer` backed by
  // the caller's memory. The caller must not touch the buffer until the run
  // completes. The device memory the execution writes into is the output's
  // entry in `preallocated_output_arrays` if it has one, and is allocated by
  // the runtime otherwise; either way it is released after the copy.
  absl::flat_hash_map<std::string, HostBuffer> preallocated_output_host_buffers;

  // The unique identifier for this execution request. Internal-only field,
  // populated automatically by the Interpreter. User should not set this field.
  int32_t request_id = -1;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_INTERPRETER_RUN_OPTIONS_H_

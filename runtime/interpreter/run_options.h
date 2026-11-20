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
  // Each array must match the dtype, shape, sharding (including devices and
  // memory kind) and layout of the output it backs, and must be in GPU or CPU
  // device memory on a PjRt-backed client. TPU cannot write a result into
  // memory the caller keeps, so a run given an array there fails before the
  // execution. No two outputs may be backed by the same memory, and no array
  // may share memory with an input of the same run, such as the `DeviceBuffer`
  // an earlier run returned over it; the run fails in both cases. To carry
  // state such as a KV cache from one run to the next, alternate between two
  // arrays: pass the previous result as the input and the other array as the
  // output.
  //
  // Lifecycle: the caller keeps the array. The run writes the result into its
  // memory in place, and the same array can be passed again to later runs. If
  // the output is placed on the device, the returned `DeviceBuffer` is a
  // different array backed by that same memory, so the next run that is given
  // this array overwrites it. If the output is placed on the host, the result
  // is copied off that memory, which stays with the caller. The memory stays
  // alive while any result backed by it does, even if the caller deletes the
  // array, and until then the array cannot be donated.
  //
  // The runtime waits for pending writes to the array before the run, but it
  // cannot see other uses of that memory, so the caller must ensure that:
  //  - when the array is passed to a run, nothing is still reading or writing
  //    its memory, whether through the array itself, through a `DeviceBuffer`
  //    returned by an earlier run that it backs, or through any other handle
  //    (e.g. an async device-to-host copy of the previous result must have
  //    completed);
  //  - the same array is not passed to runs that overlap in time;
  //  - results returned by earlier runs that it backs are treated as
  //    overwritten once it is passed to a new run.
  // None of these is detected; violating them silently produces wrong data.
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

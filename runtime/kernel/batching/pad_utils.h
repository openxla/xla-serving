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

#ifndef XLA_SERVING_RUNTIME_KERNEL_BATCHING_PAD_UTILS_H_
#define XLA_SERVING_RUNTIME_KERNEL_BATCHING_PAD_UTILS_H_

#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving

namespace xla_serving {

// Padding helper for BatchOptions.pad_variable_length_inputs. When a model
// enables that option, requests in a batch may differ in their non-batch
// dimensions, so this function pads each input buffer to a common shape: every
// non-batch dimension is grown to its maximum across the batch. The batch
// dimension (dim 0) is never padded. The caller is responsible for ignoring the
// padded positions in the model's output.
//
// Returns a new vector of HostBuffers with the padded inputs. Buffers that
// already match the common shape are shallow-copied (no buffer data
// reallocation).
//
// For example:
//   Given inputs with shapes: [1, 2, 3], [2, 3, 1], [1, 1, 4]
//   they are padded to:       [1, 3, 4], [2, 3, 4], [1, 3, 4]
absl::StatusOr<std::vector<HostBuffer>> PadToCommonNonBatchDims(
    absl::Span<const HostBuffer> inputs,
    Allocator* allocator = DefaultAllocator::GetInstance());

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_BATCHING_PAD_UTILS_H_

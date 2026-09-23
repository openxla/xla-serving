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

#ifndef XLA_SERVING_RUNTIME_KERNEL_BATCHING_RAGGED_UTILS_H_
#define XLA_SERVING_RUNTIME_KERNEL_BATCHING_RAGGED_UTILS_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// Concatenates the row splits of the tasks into a single row splits.
// `padding_amount` is the number of padding elements to add to the end of the
// row splits.
// This is used when combining multiple tasks into a single batch.
//
// For example, if the tasks splits are [0, 2, 3] and [0, 1, 3] and the
// padding amount is 4, then the output splits will be [0, 2, 3, 4, 6, 6, 6, 6,
// 6].
template <typename T>
absl::StatusOr<HostBuffer> ConcatRowSplitsTyped(
    absl::Span<const HostBuffer> unbatched_tasks_row_splits, int padding_amount,
    Allocator* allocator);

absl::StatusOr<HostBuffer> ConcatRowSplits(
    absl::Span<const HostBuffer> unbatched_tasks_row_splits, int padding_amount,
    Allocator* allocator);

// This is used when splitting a ragged input task into multiple subtasks i.e.
// when request/task batch is larger than the allowed batch size, we break the
// task into multiple subtasks that can fit into the allowed batch size.
//
// Clones the original splits and modifies the values to be the difference
// between the end and start indices.
// For example, if the original splits are [0, 2, 3, 5] and the start index is
// 1 and the end index is 3, then the output splits will be [0, 1, 3].
absl::StatusOr<HostBuffer> CloneAndModifySplits(
    const HostBuffer& original_splits, int start_idx, int end_idx,
    Allocator* allocator);

// Creates a warmup row splits buffer of size `warmup_batch_size + 1` by
// slicing the first row of `splits_source` (the first two elements) and
// replicating it `warmup_batch_size` times via concatenation.
//
// For example, if `splits_source` is [0, 2, 5] (and corresponding values
// are [10, 20, 30, 40, 50])
// splits [0, 2] corresponding to the first row which has 2 elements: [10, 20]
// If `warmup_batch_size` is 4, then replicating row 0 four times results in
// cumulative offsets [0, 2, 4, 6, 8].
// Values are handled in the `CreateWarmupFlatValues` function.
absl::StatusOr<HostBuffer> CreateWarmupRowSplits(
    const HostBuffer& splits_source, int warmup_batch_size,
    Allocator* allocator);

// Creates a warmup flat values buffer by identifying the number of values in
// row 0 from `splits_source`, slicing `values_source` to that row 0 size, and
// replicating the slice `warmup_batch_size` times via concatenation.
//
// For example, if `splits_source` is [0, 2, 5] and `values_source` is
// [10, 20, 30, 40, 50], row 0 contains the first 2 values [10, 20]. If
// `warmup_batch_size` is 4, replicating row 0 three times results in
// [10, 20, 10, 20, 10, 20, 10, 20].
absl::StatusOr<HostBuffer> CreateWarmupFlatValues(
    const HostBuffer& values_source, const HostBuffer& splits_source,
    int warmup_batch_size, tsl::thread::ThreadPool* thread_pool,
    Allocator* allocator);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_BATCHING_RAGGED_UTILS_H_

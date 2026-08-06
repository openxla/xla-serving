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

#include "runtime/kernel/batching/ragged_utils.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <cstring>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune {

template <typename T>
absl::StatusOr<HostBuffer> ConcatRowSplitsTyped(
    absl::Span<const HostBuffer> unbatched_tasks_row_splits, int padding_amount,
    Allocator* allocator) {
  if (unbatched_tasks_row_splits.empty()) {
    return absl::InvalidArgumentError("empty splits");
  }

  int64_t total_elements = 0;
  // Concat the row splits of the tasks into a single row splits, adding the
  // cumulative offset of each task to the next task.
  // For example, if the unbatched tasks have row splits [0, 2, 3] and [0, 3]
  // the offset of the first task is 3 and the offset of the second task is 6.
  // then the output splits will be [0, 2, 3, 3(0 + 3), 6(3 + 3)].
  for (const auto& splits : unbatched_tasks_row_splits) {
    if (splits.shape().dims().size() != 1) {
      return absl::InvalidArgumentError("splits buffer is not 1D");
    }
    int64_t num_elements = splits.shape().num_elements();
    if (num_elements <= 0) {
      return absl::InvalidArgumentError("split buffer is empty");
    }
    total_elements += num_elements - 1;
  }
  total_elements += 1;
  total_elements += padding_amount;

  HostBuffer output(unbatched_tasks_row_splits[0].dtype(),
                    xla::ifrt::Shape({total_elements}), allocator);
  T* out_data = reinterpret_cast<T*>(output.base());

  int64_t out_idx = 0;
  T cumulative_offset = 0;

  for (size_t i = 0; i < unbatched_tasks_row_splits.size(); ++i) {
    const HostBuffer& splits = unbatched_tasks_row_splits[i];
    const T* flat_splits = reinterpret_cast<const T*>(splits.base());
    int64_t num_elements = splits.shape().num_elements();

    if (i == 0) {
      for (int64_t j = 0; j < num_elements; ++j) {
        out_data[out_idx++] = flat_splits[j];
      }
      cumulative_offset = flat_splits[num_elements - 1];
    } else {
      for (int64_t j = 1; j < num_elements; ++j) {
        out_data[out_idx++] = flat_splits[j] + cumulative_offset;
      }
      cumulative_offset = out_data[out_idx - 1];
    }
  }

  for (int i = 0; i < padding_amount; ++i) {
    out_data[out_idx++] = cumulative_offset;
  }

  return output;
}

template absl::StatusOr<HostBuffer> ConcatRowSplitsTyped<int32_t>(
    absl::Span<const HostBuffer>, int, Allocator*);
template absl::StatusOr<HostBuffer> ConcatRowSplitsTyped<int64_t>(
    absl::Span<const HostBuffer>, int, Allocator*);

absl::StatusOr<HostBuffer> ConcatRowSplits(
    absl::Span<const HostBuffer> unbatched_tasks_row_splits, int padding_amount,
    Allocator* allocator) {
  if (unbatched_tasks_row_splits.empty()) {
    return absl::InvalidArgumentError("empty splits passed to ConcatRowSplits");
  }

  xla::ifrt::DType dtype = unbatched_tasks_row_splits[0].dtype();
  if (dtype.kind() == xla::ifrt::DType::kS32) {
    return ConcatRowSplitsTyped<int32_t>(unbatched_tasks_row_splits,
                                         padding_amount, allocator);
  } else if (dtype.kind() == xla::ifrt::DType::kS64) {
    return ConcatRowSplitsTyped<int64_t>(unbatched_tasks_row_splits,
                                         padding_amount, allocator);
  } else {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported splits dtype: ", dtype));
  }
}

absl::StatusOr<HostBuffer> CloneAndModifySplits(
    const HostBuffer& original_splits, int start_idx, int end_idx,
    Allocator* allocator) {
  int64_t num_elements = end_idx - start_idx + 1;
  HostBuffer new_splits(original_splits.dtype(),
                        xla::ifrt::Shape({num_elements}), allocator);

  if (original_splits.dtype().kind() == xla::ifrt::DType::kS32) {
    auto original_flat = original_splits.flat<int32_t>();
    auto new_flat = new_splits.flat<int32_t>();
    int32_t offset = original_flat(start_idx);
    for (int i = 0; i < num_elements; ++i) {
      new_flat(i) = original_flat(start_idx + i) - offset;
    }
  } else if (original_splits.dtype().kind() == xla::ifrt::DType::kS64) {
    auto original_flat = original_splits.flat<int64_t>();
    auto new_flat = new_splits.flat<int64_t>();
    int64_t offset = original_flat(start_idx);
    for (int i = 0; i < num_elements; ++i) {
      new_flat(i) = original_flat(start_idx + i) - offset;
    }
  } else {
    return absl::InvalidArgumentError("Unsupported splits dtype");
  }
  return new_splits;
}

}  // namespace neptune

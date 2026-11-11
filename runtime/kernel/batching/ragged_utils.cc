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

#include "runtime/kernel/batching/ragged_utils.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_type_utils.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

template <typename T>
absl::StatusOr<HostBuffer> ConcatRowSplitsTyped(
    absl::Span<const HostBuffer> unbatched_tasks_row_splits, int padding_amount,
    Allocator* allocator) {
  if (unbatched_tasks_row_splits.empty()) {
    return absl::InvalidArgumentError("empty splits");
  }

  const xla::ifrt::DType expected_dtype(NativeAsDTypeKind<T>::value);
  int64_t total_elements = 0;
  // Concat the row splits of the tasks into a single row splits, adding the
  // cumulative offset of each task to the next task.
  // For example, if the unbatched tasks have row splits [0, 2, 3] and [0, 3]
  // the offset of the first task is 3 and the offset of the second task is 6.
  // then the output splits will be [0, 2, 3, 3(0 + 3), 6(3 + 3)].
  for (size_t i = 0; i < unbatched_tasks_row_splits.size(); ++i) {
    const HostBuffer& splits = unbatched_tasks_row_splits[i];
    if (!splits.has_data()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Row splits buffer at index ", i, " is uninitialized"));
    }
    if (splits.dtype() != expected_dtype) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Dtypes of all row splits buffers must match expected dtype ",
          expected_dtype, ": dtype[", i, "] = ", splits.dtype()));
    }
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
  if (!unbatched_tasks_row_splits[0].has_data()) {
    return absl::InvalidArgumentError(
        "Row splits buffer at index 0 is uninitialized");
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

absl::StatusOr<HostBuffer> CreateWarmupRowSplits(
    const HostBuffer& splits_source, int warmup_batch_size,
    Allocator* allocator) {
  if (splits_source.shape().dims().empty() ||
      splits_source.shape().dims()[0] < 2) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Cannot use a row_splits buffer with fewer than 2 elements for warmup "
        "batching; got shape ",
        splits_source.shape()));
  }
  HostBuffer first_row_splits;
  if (splits_source.shape().dims()[0] == 2) {
    first_row_splits = splits_source;
  } else {
    absl::StatusOr<HostBuffer> res =
        splits_source.SliceAlongFirstDimension(0, 2, allocator);
    if (!res.ok()) {
      return res.status();
    }
    first_row_splits = std::move(*res);
  }
  std::vector<HostBuffer> tasks_splits;
  tasks_splits.reserve(warmup_batch_size);
  for (int j = 0; j < warmup_batch_size; ++j) {
    tasks_splits.push_back(first_row_splits);
  }
  return ConcatRowSplits(tasks_splits, /*padding_amount=*/0, allocator);
}

absl::StatusOr<HostBuffer> CreateWarmupFlatValues(
    const HostBuffer& values_source, const HostBuffer& splits_source,
    int warmup_batch_size, tsl::thread::ThreadPool* thread_pool,
    Allocator* allocator) {
  if (splits_source.shape().dims().empty() ||
      splits_source.shape().dims()[0] < 2) {
    return absl::InvalidArgumentError(
        "row_splits buffer must have at least 2 elements for warmup.");
  }
  int64_t num_values = 0;
  if (splits_source.dtype().kind() == xla::ifrt::DType::kS32) {
    auto splits_flat = splits_source.flat<int32_t>();
    num_values = splits_flat(1) - splits_flat(0);
  } else if (splits_source.dtype().kind() == xla::ifrt::DType::kS64) {
    auto splits_flat = splits_source.flat<int64_t>();
    num_values = splits_flat(1) - splits_flat(0);
  } else {
    return absl::InvalidArgumentError("Unsupported splits dtype");
  }

  HostBuffer first_row_values;
  if (values_source.shape().dims()[0] == num_values) {
    first_row_values = values_source;
  } else {
    absl::StatusOr<HostBuffer> res = values_source.SliceAlongFirstDimension(
        0, num_values, allocator);
    if (!res.ok()) {
      return res.status();
    }
    first_row_values = std::move(*res);
  }
  std::vector<HostBuffer> to_concatenate;
  to_concatenate.reserve(warmup_batch_size);
  for (int j = 0; j < warmup_batch_size; ++j) {
    to_concatenate.push_back(first_row_values);
  }
  return Concat(thread_pool, to_concatenate, allocator);
}

}  // namespace xla_serving

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

#include "runtime/kernel/batching/pad_utils.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {

namespace {

// Copies HostBuffer elements from `src` into the pre-initialized (zeroed or
// default-constructed) `padded_dst` buffer, overwriting the target coordinates
// while leaving padding positions untouched.
void CopyDataIntoPaddedBuffer(const HostBuffer& src, HostBuffer& padded_dst) {
  const absl::Span<const int64_t> src_dims = src.shape().dims();
  const size_t rank = src_dims.size();

  const absl::InlinedVector<int64_t, 4>& src_byte_strides = *src.byte_strides();
  const absl::InlinedVector<int64_t, 4>& dst_byte_strides =
      *padded_dst.byte_strides();
  const char* src_bytes = static_cast<const char*>(src.base());
  char* dst_bytes = static_cast<char*>(padded_dst.base());

  const bool is_string = src.dtype().kind() == xla::ifrt::DType::Kind::kString;
  const int64_t element_byte_size =
      is_string ? sizeof(tsl::tstring) : *src.dtype().byte_size();

  // Tracks outer indices from dimension 0 to rank - 2. The innermost dimension
  // is omitted because entire rows are copied in one contiguous chunk.
  std::vector<int64_t> coords(rank - 1, 0);
  int64_t src_offset = 0;
  int64_t dst_offset = 0;
  // The innermost dimension is contiguous in memory, representing a 1D row.
  const int64_t elements_per_row = src_dims[rank - 1];
  const int64_t row_bytes = elements_per_row * element_byte_size;
  int64_t total_rows = 1;
  for (size_t d = 0; d < rank - 1; ++d) {
    total_rows *= src_dims[d];
  }

  for (int64_t row = 0; row < total_rows; ++row) {
    if (is_string) {
      // tsl::tstring is not trivially copyable. We must copy elements one by
      // one using tsl::tstring::operator= rather than memcpy to ensure deep
      // copies and prevent double-free bugs when the buffers are destroyed.
      const tsl::tstring* src_strings =
          reinterpret_cast<const tsl::tstring*>(src_bytes + src_offset);
      tsl::tstring* dst_strings =
          reinterpret_cast<tsl::tstring*>(dst_bytes + dst_offset);
      for (int64_t i = 0; i < elements_per_row; ++i) {
        dst_strings[i] = src_strings[i];
      }
    } else {
      // Copy the entire source row.
      std::memcpy(dst_bytes + dst_offset, src_bytes + src_offset, row_bytes);
    }

    // Advance outer coordinates and offsets to the next row (odometer-style).
    // Starting from dimension rank - 2 down to 0, increment dimension d.
    // If it reaches src_dims[d], reset and carry over to dimension d - 1.
    //
    // Example:
    //   For shape [2, 3, 4], coords tracks [batch(=2), row(=3)] across 6 rows:
    //     Batch 0: [0, 0], [0, 1], [0, 2]  --> Copy rows 0, 1, 2
    //     Batch 1: [1, 0], [1, 1], [1, 2]  --> Copy rows 3, 4, 5
    // When dim 1 reaches size 3, it resets to 0, rewinds offsets and carries
    // over to dim 0.
    for (int d = static_cast<int>(rank) - 2; d >= 0; --d) {
      coords[d]++;
      src_offset += src_byte_strides[d];
      dst_offset += dst_byte_strides[d];
      // No carry-over needed if this dimension is still within bounds.
      if (coords[d] < src_dims[d]) {
        break;
      }
      // Rollover: coords[d] reached src_dims[d] and resets to 0.
      coords[d] = 0;
      // Rewind this dimension's offset contribution and carry over to the next
      // dimension.
      //
      // Example:
      //   E = element byte size.
      //   For src shape [2, 3, 4] (byte strides [12E, 4E, 1E]) and
      //       dst shape [2, 4, 6] (byte strides [24E, 6E, 1E])
      //   Consider advancing to row 2 of batch 1:
      //     - coords: [1, 2] --> [1, 3]
      //     - src_offset:
      //       - 20E (batch 1 added 12E, row 0 and 1 added 8E) + 4E = 24E
      //     - dst_offset:
      //       - 36E (batch 1 added 24E, row 0 and 1 added 12E) + 6E = 42E
      //     - Check coords[1] < 3 --> false (reset coords[1] to 0, carry over)
      //       - src_offset:
      //         - 24E - (3 * 4E) = 12E (rewind to start of batch 1)
      //       - dst_offset:
      //         - 42E - (3 * 6E) = 24E (rewind to start of batch 1)
      //     - Roll over to dim 0:
      //     - coords: [1, 0] --> [2, 0]
      //     - src_offset: 12E + 12E = 24E
      //     - dst_offset: 24E + 24E = 48E
      //     - Check coords[0] < 2 --> false (reset coords[0] to 0, all done)
      //       - src_offset: 24E - (2 * 12E) = 0
      //       - dst_offset: 48E - (2 * 24E) = 0
      src_offset -= src_dims[d] * src_byte_strides[d];
      dst_offset -= src_dims[d] * dst_byte_strides[d];
    }
  }
}

// Pads a single HostBuffer to target_shape. Non-batch dimensions (dim >= 1)
// must be greater than or equal to the input's dimensions.
absl::StatusOr<HostBuffer> PadNonBatchDims(const HostBuffer& host_buffer,
                                           const xla::ifrt::Shape& target_shape,
                                           Allocator* allocator) {
  const bool is_string =
      host_buffer.dtype().kind() == xla::ifrt::DType::Kind::kString;
  if (!is_string && !host_buffer.dtype().byte_size().has_value()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Unsupported data type for padding: ", host_buffer.dtype()));
  }
  if (!host_buffer.byte_strides().has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("HostBuffer must have byte strides for padding, got: ",
                     host_buffer.DebugString()));
  }

  // Create a new HostBuffer with the target shape.
  // For the string type, the HostBuffer constructor automatically initializes
  // empty tsl::tstring. For numeric types, we use memset to initialize the
  // memory to zero.
  HostBuffer padded_buffer(host_buffer.dtype(), target_shape, allocator);
  if (!is_string && padded_buffer.total_storage_bytes() > 0) {
    std::memset(padded_buffer.base(), 0, padded_buffer.total_storage_bytes());
  }

  if (host_buffer.shape().num_elements() == 0) {
    return padded_buffer;
  }

  CopyDataIntoPaddedBuffer(host_buffer, padded_buffer);
  return padded_buffer;
}

}  // namespace

absl::StatusOr<std::vector<HostBuffer>> PadToCommonNonBatchDims(
    absl::Span<const HostBuffer> inputs, Allocator* allocator) {
  if (inputs.empty()) {
    return std::vector<HostBuffer>();
  }
  if (allocator == nullptr) {
    allocator = DefaultAllocator::GetInstance();
  }

  const xla::ifrt::DType ref_dtype = inputs.front().dtype();
  const absl::Span<const int64_t> input0_dims = inputs.front().shape().dims();
  // Initialize max_dims to the shape of the first input.
  std::vector<int64_t> max_dims(input0_dims.begin(), input0_dims.end());
  const size_t rank = max_dims.size();

  bool shapes_differ = false;
  for (size_t i = 1; i < inputs.size(); ++i) {
    const HostBuffer& input = inputs[i];
    // Make sure all inputs have the same dtype and rank.
    if (input.dtype() != ref_dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Dtype mismatch across inputs: input 0 has ", ref_dtype,
                       " but input ", i, " has ", input.dtype()));
    }
    if (input.shape().dims().size() != rank) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Rank mismatch across inputs: input 0 has rank ", rank, " but input ",
          i, " has rank ", input.shape().dims().size()));
    }
    // Grow max_dims to the maximum of each non-batch dimension.
    for (size_t d = 1; d < rank; ++d) {
      const int64_t dim = input.shape().dims()[d];
      if (!shapes_differ && dim != input0_dims[d]) {
        shapes_differ = true;
      }
      max_dims[d] = std::max(dim, max_dims[d]);
    }
  }

  // Fast path: no padding is needed. Return shallow copies of all inputs.
  if (!shapes_differ) {
    return std::vector<HostBuffer>(inputs.begin(), inputs.end());
  }

  // Per-input target shape. Dims 1 and up are the common maximum. Dim 0 is the
  // input's own batch size and is never padded.
  std::vector<int64_t> target_dims = max_dims;
  std::vector<HostBuffer> outputs;
  outputs.reserve(inputs.size());
  for (const HostBuffer& input : inputs) {
    bool needs_padding = false;
    for (size_t d = 1; d < rank; ++d) {
      if (input.shape().dims()[d] < max_dims[d]) {
        needs_padding = true;
        break;
      }
    }

    if (!needs_padding) {
      outputs.push_back(input);
      continue;
    }

    // Set batch dimension to the input's own batch size.
    target_dims[0] = input.shape().dims()[0];
    TF_ASSIGN_OR_RETURN(
        HostBuffer padded,
        PadNonBatchDims(input, xla::ifrt::Shape(target_dims), allocator));
    outputs.push_back(std::move(padded));
  }

  return outputs;
}

}  // namespace xla_serving

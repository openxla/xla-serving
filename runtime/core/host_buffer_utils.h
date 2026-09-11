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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_HOST_BUFFER_UTILS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_HOST_BUFFER_UTILS_H_

#include <cstddef>
#include <cstdint>
#include <optional>
#include <string>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "xla/python/ifrt/dtype.h"  // from @xla

namespace xla_serving {

// Creates default strides for a given shape and dtype.
std::optional<absl::InlinedVector<int64_t, 4>> CreateDefaultStrides(
    absl::Span<int64_t const> dims, xla::ifrt::DType dtype);

// Appends a recursive representation of the given data to an output string.
void AppendRecursive(std::string* out, xla::ifrt::DType dtype, const void* data,
                     size_t indent, absl::Span<int64_t const> byte_strides,
                     absl::Span<int64_t const> counts);

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_HOST_BUFFER_UTILS_H_

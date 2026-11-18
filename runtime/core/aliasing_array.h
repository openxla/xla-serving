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

#ifndef XLA_SERVING_RUNTIME_CORE_ALIASING_ARRAY_H_
#define XLA_SERVING_RUNTIME_CORE_ALIASING_ARRAY_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla

namespace xla_serving {

// Whether an execution may write into an aliasing array.
enum class AliasAccess {
  // Only read, like an ordinary execution argument.
  kReadOnly,
  // Written into, like an argument an execution output is aliased onto.
  kWritable,
};

// Returns a new array of `target_client` that aliases the device memory of
// `source`, without copying it.
//
// The new array shares ownership of the memory with `source`: each of its
// buffers shares the reference-counted raw buffer of the corresponding buffer
// of `source`. `source` stays valid, and the memory is released only once
// `source`, the new array, and anything that takes the new array's memory over
// are all gone. Donating the new array to an execution, which hands its memory
// to the execution's output, therefore consumes only the new array and never
// `source`.
//
// The new array has no definition events of its own. If `source` is not ready
// yet, this blocks until it is, so that the new array is ready when returned.
// The caller must not write into the memory through `source` while the new
// array, or anything that took its memory over, may still be reading it, and
// vice versa.
//
// The new array is as mutable as `source`; asking for `AliasAccess::kWritable`
// fails if `source`'s memory is immutable.
//
// Both `source` and `target_client` must be PjRt-backed, and `target_client`
// must be the client of `source` and built on `xla::CommonPjRtClient`.
absl::StatusOr<xla::ifrt::ArrayRef> MakeAliasingArray(
    xla::ifrt::Array& source, xla::ifrt::Client& target_client,
    AliasAccess access);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_ALIASING_ARRAY_H_

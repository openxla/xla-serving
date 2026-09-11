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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_

#include <cstddef>

namespace xla_serving {

// An interface for allocating memory.
class Allocator {
 public:
  // Allocates `size` bytes of memory. The returned pointer is guaranteed to be
  // aligned to a multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  virtual void* Allocate(size_t size, size_t alignment) = 0;

  // Frees the memory allocated at `ptr`.
  virtual void Free(void* ptr, size_t alignment) = 0;

  virtual ~Allocator() = default;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_

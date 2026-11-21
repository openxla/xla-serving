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

#ifndef XLA_SERVING_RUNTIME_ALLOCATOR_H_
#define XLA_SERVING_RUNTIME_ALLOCATOR_H_

#include <cstddef>
#include <memory>

namespace xla_serving {

// An interface for allocating memory.
class Allocator {
 public:
  // Allocates `size` bytes of memory. The returned pointer is guaranteed to be
  // aligned to a multiple of "alignment" bytes.
  // REQUIRES: "alignment" is a power of 2.
  //
  // Prefer `AllocateOwned()` for new callers: implementations backed by
  // per-allocation deleters (such as `PjRtHostAllocator`) incur locking and
  // map-lookup overhead in `Allocate()`/`Free()`.
  virtual void* Allocate(size_t size, size_t alignment) = 0;

  // Frees the memory allocated at `ptr`. Prefer `AllocateOwned()` over raw
  // `Allocate()`/`Free()`.
  virtual void Free(void* ptr, size_t alignment) = 0;

  // Allocates `size` bytes of memory aligned to `alignment` and returns a
  // ref-counted owner whose `.get()` is the allocated buffer and whose deleter
  // releases it.
  //
  // Subclasses backed by allocators that return per-allocation deleters (such
  // as `xla::HostMemoryAllocator`) can override this method to avoid tracking
  // pointer-to-deleter mappings in `Allocate`/`Free`.
  virtual std::shared_ptr<void> AllocateOwned(size_t size, size_t alignment) {
    void* ptr = Allocate(size, alignment);
    if (ptr == nullptr) {
      return nullptr;
    }
    return std::shared_ptr<void>(ptr, [this, alignment](void* p) {
      if (p != nullptr) {
        Free(p, alignment);
      }
    });
  }

  virtual ~Allocator() = default;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_ALLOCATOR_H_

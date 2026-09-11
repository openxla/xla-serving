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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <new>

#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @neptune

namespace xla_serving {

// An allocator that allocates host memory.
class DefaultAllocator : public Allocator {
 public:
  void* Allocate(size_t size, size_t alignment) override {
    return new (std::align_val_t(alignment)) uint8_t[size];
  }

  void Free(void* ptr, size_t alignment) override {
    ::operator delete[](ptr, std::align_val_t(alignment));
  }

  static DefaultAllocator* GetInstance() {
    static absl::NoDestructor<DefaultAllocator> default_allocator;
    return default_allocator.get();
  }

 private:
  friend class absl::NoDestructor<DefaultAllocator>;
  DefaultAllocator() = default;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_

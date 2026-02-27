#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_

#include <cstddef>
#include <cstdint>
#include <new>

#include "third_party/absl/base/no_destructor.h"
#include "third_party/neptune/runtime/core/allocator.h"

namespace jsv {

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

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DEFAULT_ALLOCATOR_H_

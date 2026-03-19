#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_

#include <cstddef>

namespace neptune {

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

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_ALLOCATOR_H_

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

#ifndef XLA_SERVING_RUNTIME_CORE_PJRT_HOST_ALLOCATOR_H_
#define XLA_SERVING_RUNTIME_CORE_PJRT_HOST_ALLOCATOR_H_

#include <cstddef>
#include <memory>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla

namespace xla_serving {

// A unified non-owning `Allocator` backed by a PJRT client's
// `xla::HostMemoryAllocator`.
//
// Works uniformly across PJRT backends (TPU, GPU, and C API plugins) whose
// `PjRtClient::GetHostMemoryAllocator()` exposes a premapped or pinned host
// memory pool. Buffers allocated via `AllocateOwned` (the fast path used by
// `HostBuffer`) bypass locking and forward `xla::HostMemoryAllocator::Deleter`
// directly without retaining the `xla::PjRtClient`.
//
// Lifetime: The caller must ensure the underlying `xla::PjRtClient` /
// `xla::HostMemoryAllocator` outlives both this `PjRtHostAllocator` and every
// buffer allocated from it.
class PjRtHostAllocator : public Allocator {
 public:
  // Non-owning constructor. The caller must ensure `host_memory_allocator`
  // outlives this allocator and all buffers allocated from it.
  explicit PjRtHostAllocator(
      xla::HostMemoryAllocator* absl_nonnull host_memory_allocator);

  // Creates a non-owning `PjRtHostAllocator` backed by `pjrt_client`'s
  // `GetHostMemoryAllocator()`. The caller must ensure `pjrt_client` outlives
  // the returned allocator and all buffers allocated from it.
  static absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> Create(
      xla::PjRtClient* absl_nullable pjrt_client);

  // Creates a non-owning `PjRtHostAllocator` from an IFRT client wrapping a
  // `PjRtCompatibleClient` with a non-null `GetHostMemoryAllocator()`. The
  // caller must ensure `ifrt_client` (and its underlying `xla::PjRtClient`)
  // outlives the returned allocator and all buffers allocated from it.
  static absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> Create(
      xla::ifrt::Client* absl_nullable ifrt_client);

  PjRtHostAllocator(const PjRtHostAllocator&) = delete;
  PjRtHostAllocator& operator=(const PjRtHostAllocator&) = delete;

  OwnedPtr AllocateOwned(size_t size, size_t alignment) override;

  // Unsupported: `xla::HostMemoryAllocator` uses per-allocation deleters and
  // does not support stateless `Free(void*)`. Use `AllocateOwned()` instead.
  // Both methods `LOG(DFATAL)` (with `Allocate` returning `nullptr`).
  void* Allocate(size_t size, size_t alignment) override;
  void Free(void* ptr, size_t alignment) override;

 private:
  xla::HostMemoryAllocator* const absl_nonnull host_memory_allocator_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_PJRT_HOST_ALLOCATOR_H_

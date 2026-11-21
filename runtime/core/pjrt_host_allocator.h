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
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla

namespace xla_serving {

// A unified `Allocator` backed by a PJRT client's `xla::HostMemoryAllocator`.
//
// Works uniformly across PJRT backends (TPU, GPU, and C API plugins) whose
// `PjRtClient::GetHostMemoryAllocator()` exposes a premapped or pinned host
// memory pool. Buffers allocated via `AllocateOwned` (the fast path used by
// `HostBuffer`) bypass locking and retain the underlying `xla::PjRtClient` when
// constructed via `Create(...)` so buffers safely outlive the client.
class PjRtHostAllocator : public Allocator {
 public:
  // Non-owning constructor. The caller must ensure `host_memory_allocator`
  // outlives all buffers allocated from this allocator.
  explicit PjRtHostAllocator(
      xla::HostMemoryAllocator* absl_nonnull host_memory_allocator);

  // Creates a `PjRtHostAllocator` backed by `pjrt_client`'s
  // `GetHostMemoryAllocator()`, retaining `pjrt_client` inside every allocated
  // buffer's owner so buffers safely outlive the client.
  static absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> Create(
      absl_nonnull std::shared_ptr<xla::PjRtClient> pjrt_client);

  // Creates a `PjRtHostAllocator` from an IFRT client wrapping a
  // `PjRtCompatibleClient` with a non-null `GetHostMemoryAllocator()`.
  static absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> Create(
      const std::shared_ptr<xla::ifrt::Client>& ifrt_client);

  PjRtHostAllocator(const PjRtHostAllocator&) = delete;
  PjRtHostAllocator& operator=(const PjRtHostAllocator&) = delete;

  std::shared_ptr<void> AllocateOwned(size_t size, size_t alignment) override;

  // Prefer `AllocateOwned()` over `Allocate()`/`Free()`. Do not use these raw
  // methods on performance-sensitive paths: because `xla::HostMemoryAllocator`
  // has no `Free(void*)`, these acquire `raw_allocations_mu_` and track every
  // allocation in a hash map.
  void* Allocate(size_t size, size_t alignment) override;
  void Free(void* ptr, size_t alignment) override;

 private:
  PjRtHostAllocator(
      std::shared_ptr<xla::PjRtClient> pjrt_client,
      xla::HostMemoryAllocator* absl_nonnull host_memory_allocator);

  std::shared_ptr<xla::PjRtClient> pjrt_client_;
  xla::HostMemoryAllocator* const absl_nonnull host_memory_allocator_;

  // Tracks allocations returned by the raw `Allocate`/`Free` API so their
  // `xla::HostMemoryAllocator::Deleter` (and optional client hold) can be
  // invoked on `Free`. Not touched by `AllocateOwned`.
  absl::Mutex raw_allocations_mu_;
  absl::flat_hash_map<void*, std::shared_ptr<void>> raw_allocations_
      ABSL_GUARDED_BY(raw_allocations_mu_);
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_PJRT_HOST_ALLOCATOR_H_

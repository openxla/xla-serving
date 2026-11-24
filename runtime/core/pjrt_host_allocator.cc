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

#include "runtime/core/pjrt_host_allocator.h"  // from @xla_serving

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <memory>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla

namespace xla_serving {

PjRtHostAllocator::PjRtHostAllocator(
    xla::HostMemoryAllocator* absl_nonnull host_memory_allocator)
    : host_memory_allocator_(host_memory_allocator) {
  CHECK_NE(host_memory_allocator_, nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> PjRtHostAllocator::Create(
    xla::PjRtClient* absl_nullable pjrt_client) {
  if (pjrt_client == nullptr) {
    return absl::InvalidArgumentError("PjRtClient must not be null.");
  }
  xla::HostMemoryAllocator* host_memory_allocator =
      pjrt_client->GetHostMemoryAllocator();
  if (host_memory_allocator == nullptr) {
    return absl::FailedPreconditionError(
        "PjRtClient does not expose a HostMemoryAllocator.");
  }
  return std::make_unique<PjRtHostAllocator>(host_memory_allocator);
}

absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> PjRtHostAllocator::Create(
    xla::ifrt::Client* absl_nullable ifrt_client) {
  if (ifrt_client == nullptr) {
    return absl::InvalidArgumentError("IFRT client must not be null.");
  }
  auto* pjrt_ifrt_client =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleClient>(ifrt_client);
  if (pjrt_ifrt_client == nullptr) {
    return absl::FailedPreconditionError(
        "IFRT client is not a PjRtCompatibleClient.");
  }
  return Create(pjrt_ifrt_client->pjrt_client());
}

Allocator::OwnedPtr PjRtHostAllocator::AllocateOwned(size_t size,
                                                     size_t alignment) {
  if (size == 0) {
    return nullptr;
  }
  // Normalize 0 (e.g., `EIGEN_MAX_ALIGN_BYTES` on unvectorized builds) to 1 to
  // avoid modulo-by-zero below and ensure the `DefaultAllocator` fallback
  // passes a valid power-of-two alignment to `std::align_val_t`.
  const size_t effective_alignment = std::max<size_t>(alignment, 1);
  xla::HostMemoryAllocator::OwnedPtr owned =
      host_memory_allocator_->Allocate(size);
  if (owned != nullptr &&
      (effective_alignment == 1 ||
       reinterpret_cast<uintptr_t>(owned.get()) % effective_alignment == 0)) {
    // Forward `xla::HostMemoryAllocator::Deleter` directly (2 pointers) so it
    // fits in `absl::AnyInvocable`'s inline storage without an extra heap
    // allocation.
    xla::HostMemoryAllocator::Deleter deleter = owned.get_deleter();
    return OwnedPtr(owned.release(), deleter);
  }
  // TODO: Add streamz to record the fallback rate and reasons.
  return DefaultAllocator::GetInstance()->AllocateOwned(size,
                                                        effective_alignment);
}

void* PjRtHostAllocator::Allocate(size_t size, size_t alignment) {
  LOG(DFATAL) << "PjRtHostAllocator::Allocate is unsupported; use "
                 "AllocateOwned() instead.";
  return nullptr;
}

void PjRtHostAllocator::Free(void* ptr, size_t alignment) {
  LOG(DFATAL) << "PjRtHostAllocator::Free is unsupported; use "
                 "AllocateOwned() instead.";
}

}  // namespace xla_serving

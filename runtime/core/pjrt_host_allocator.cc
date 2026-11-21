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

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla

namespace xla_serving {
namespace {

struct PjRtBufferOwner {
  // Declared before `buffer` so `buffer` is destroyed while `pjrt_client` (and
  // its `HostMemoryAllocator`) is still alive.
  std::shared_ptr<xla::PjRtClient> pjrt_client;
  xla::HostMemoryAllocator::OwnedPtr buffer;
};

}  // namespace

PjRtHostAllocator::PjRtHostAllocator(
    xla::HostMemoryAllocator* absl_nonnull host_memory_allocator)
    : PjRtHostAllocator(/*pjrt_client=*/nullptr, host_memory_allocator) {}

PjRtHostAllocator::PjRtHostAllocator(
    std::shared_ptr<xla::PjRtClient> pjrt_client,
    xla::HostMemoryAllocator* absl_nonnull host_memory_allocator)
    : pjrt_client_(std::move(pjrt_client)),
      host_memory_allocator_(host_memory_allocator) {
  CHECK_NE(host_memory_allocator_, nullptr);
}

absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> PjRtHostAllocator::Create(
    absl_nonnull std::shared_ptr<xla::PjRtClient> pjrt_client) {
  if (pjrt_client == nullptr) {
    return absl::InvalidArgumentError("PjRtClient must not be null.");
  }
  xla::HostMemoryAllocator* host_memory_allocator =
      pjrt_client->GetHostMemoryAllocator();
  if (host_memory_allocator == nullptr) {
    return absl::FailedPreconditionError(
        "PjRtClient does not expose a HostMemoryAllocator.");
  }
  return absl::WrapUnique(
      new PjRtHostAllocator(std::move(pjrt_client), host_memory_allocator));
}

absl::StatusOr<std::unique_ptr<PjRtHostAllocator>> PjRtHostAllocator::Create(
    const std::shared_ptr<xla::ifrt::Client>& ifrt_client) {
  if (ifrt_client == nullptr) {
    return absl::InvalidArgumentError("IFRT client must not be null.");
  }
  auto* pjrt_ifrt_client =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleClient>(ifrt_client.get());
  if (pjrt_ifrt_client == nullptr) {
    return absl::FailedPreconditionError(
        "IFRT client is not a PjRtCompatibleClient.");
  }
  return Create(pjrt_ifrt_client->shared_ptr_pjrt_client());
}

std::shared_ptr<void> PjRtHostAllocator::AllocateOwned(size_t size,
                                                       size_t alignment) {
  if (size == 0) {
    return nullptr;
  }
  xla::HostMemoryAllocator::OwnedPtr owned =
      host_memory_allocator_->Allocate(size);
  if (owned != nullptr &&
      reinterpret_cast<uintptr_t>(owned.get()) % alignment == 0) {
    void* base = owned.get();
    if (pjrt_client_ != nullptr) {
      auto holder = std::make_shared<PjRtBufferOwner>(
          PjRtBufferOwner{pjrt_client_, std::move(owned)});
      return std::shared_ptr<void>(std::move(holder), base);
    }
    return std::shared_ptr<void>(std::move(owned));
  }
  return DefaultAllocator::GetInstance()->AllocateOwned(size, alignment);
}

void* PjRtHostAllocator::Allocate(size_t size, size_t alignment) {
  std::shared_ptr<void> owned = AllocateOwned(size, alignment);
  void* ptr = owned.get();
  if (ptr == nullptr) {
    return nullptr;
  }
  absl::MutexLock lock(raw_allocations_mu_);
  raw_allocations_.emplace(ptr, std::move(owned));
  return ptr;
}

void PjRtHostAllocator::Free(void* ptr, size_t alignment) {
  if (ptr == nullptr) {
    return;
  }
  std::shared_ptr<void> to_release;
  {
    absl::MutexLock lock(raw_allocations_mu_);
    auto it = raw_allocations_.find(ptr);
    if (it != raw_allocations_.end()) {
      to_release = std::move(it->second);
      raw_allocations_.erase(it);
    }
  }
  // Release `to_release` outside the lock.
}

}  // namespace xla_serving

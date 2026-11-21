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

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <new>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_matcher.h"  // from @xla_serving
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "xla/pjrt/host_memory_allocator.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "tsl/platform/mem.h"  // from @tsl
#include "tsl/platform/tstring.h"  // from @tsl

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;

class CountingHostMemoryAllocator : public xla::HostMemoryAllocator {
 public:
  explicit CountingHostMemoryAllocator(size_t alignment = 64,
                                       bool fail_allocations = false)
      : alignment_(alignment), fail_allocations_(fail_allocations) {}

  absl_nullable OwnedPtr Allocate(size_t size,
                                  const AllocateOptions& options) override {
    if (size == 0 || fail_allocations_) {
      return nullptr;
    }
    void* ptr = tsl::port::AlignedMalloc(
        size, static_cast<std::align_val_t>(alignment_));
    if (ptr == nullptr) {
      return nullptr;
    }
    alloc_count_.fetch_add(1, std::memory_order_relaxed);
    live_count_.fetch_add(1, std::memory_order_relaxed);
    return OwnedPtr(static_cast<uint8_t*>(ptr),
                    Deleter{/*deleter=*/&Deallocate, /*arg=*/this});
  }

  int alloc_count() const {
    return alloc_count_.load(std::memory_order_relaxed);
  }
  int live_count() const { return live_count_.load(std::memory_order_relaxed); }

 private:
  static void Deallocate(void* ptr, void* arg) {
    auto* self = static_cast<CountingHostMemoryAllocator*>(arg);
    self->live_count_.fetch_sub(1, std::memory_order_relaxed);
    tsl::port::AlignedFree(ptr);
  }

  size_t alignment_;
  bool fail_allocations_;
  std::atomic<int> alloc_count_{0};
  std::atomic<int> live_count_{0};
};

TEST(PjRtHostAllocatorTest, HostBufferUsesAllocateOwnedFastPath) {
  CountingHostMemoryAllocator backend(/*alignment=*/64);
  PjRtHostAllocator allocator(&backend);

  {
    HostBuffer buffer(xla::ifrt::DType(xla::ifrt::DType::kF32),
                      xla::ifrt::Shape({128}), &allocator);
    EXPECT_EQ(backend.alloc_count(), 1);
    EXPECT_EQ(backend.live_count(), 1);
    auto flat = buffer.flat<float>();
    for (int i = 0; i < 128; ++i) {
      flat(i) = static_cast<float>(i);
    }
    std::vector<float> expected(128);
    for (int i = 0; i < 128; ++i) {
      expected[i] = static_cast<float>(i);
    }
    EXPECT_THAT(buffer, HostBufferEq(AsHostBuffer<float>(
                            expected, xla::ifrt::Shape({128}))));
  }
  EXPECT_EQ(backend.live_count(), 0);
}

TEST(PjRtHostAllocatorTest, HostBufferStringDestroysStringsBeforeFree) {
  CountingHostMemoryAllocator backend(/*alignment=*/64);
  PjRtHostAllocator allocator(&backend);

  {
    HostBuffer buffer(xla::ifrt::DType(xla::ifrt::DType::kString),
                      xla::ifrt::Shape({3}), &allocator);
    EXPECT_EQ(backend.alloc_count(), 1);
    EXPECT_EQ(backend.live_count(), 1);
    buffer.flat<tsl::tstring>()(0) = "alpha";
    buffer.flat<tsl::tstring>()(1) = "beta";
    buffer.flat<tsl::tstring>()(2) = "gamma";
    EXPECT_EQ(buffer.flat<tsl::tstring>()(1), "beta");
  }
  EXPECT_EQ(backend.live_count(), 0);
}

TEST(PjRtHostAllocatorTest, RawAllocateAndFreeInvokeBackendDeleter) {
  CountingHostMemoryAllocator backend(/*alignment=*/64);
  PjRtHostAllocator allocator(&backend);

  void* ptr = allocator.Allocate(256, 64);
  ASSERT_NE(ptr, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(ptr) % 64, 0);
  EXPECT_EQ(backend.alloc_count(), 1);
  EXPECT_EQ(backend.live_count(), 1);

  allocator.Free(ptr, 64);
  EXPECT_EQ(backend.live_count(), 0);
}

TEST(PjRtHostAllocatorTest, FallsBackToDefaultAllocatorOnFailureOrMisalign) {
  CountingHostMemoryAllocator failing_backend(/*alignment=*/64,
                                              /*fail_allocations=*/true);
  PjRtHostAllocator failing_allocator(&failing_backend);

  std::shared_ptr<void> fallback = failing_allocator.AllocateOwned(128, 64);
  ASSERT_NE(fallback, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(fallback.get()) % 64, 0);
  EXPECT_EQ(failing_backend.alloc_count(), 0);

  // Request stricter alignment than the backend provides.
  CountingHostMemoryAllocator unaligned_backend(/*alignment=*/16);
  PjRtHostAllocator unaligned_allocator(&unaligned_backend);
  std::shared_ptr<void> aligned = unaligned_allocator.AllocateOwned(128, 4096);
  ASSERT_NE(aligned, nullptr);
  EXPECT_EQ(reinterpret_cast<uintptr_t>(aligned.get()) % 4096, 0);
}

TEST(PjRtHostAllocatorTest, CreateRejectsNullClients) {
  EXPECT_THAT(PjRtHostAllocator::Create(std::shared_ptr<xla::PjRtClient>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
  EXPECT_THAT(PjRtHostAllocator::Create(std::shared_ptr<xla::ifrt::Client>()),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla_serving

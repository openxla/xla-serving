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

#include "runtime/core/aliasing_array.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;
using ::testing::Eq;
using ::testing::HasSubstr;
using ::testing::Ne;

class AliasingArrayTest : public testing::Test {
 protected:
  void SetUp() override {
    TF_ASSERT_OK_AND_ASSIGN(client_, xla::ifrt::test_util::GetClient());
    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "AliasingArrayTest", 2);
  }

  // Returns a ready f32[4] array of `client` holding `values`.
  xla::ifrt::ArrayRef MakeArray(xla::ifrt::Client& client,
                                const std::vector<float>& values) {
    xla::ifrt::Device* const device = client.addressable_devices().front();
    absl::StatusOr<xla::ifrt::ArrayRef> array =
        sharding_internal::MakeArrayFromHostBuffer(
            client, AsHostBuffer<float>(values, xla::ifrt::Shape({4})),
            std::vector<int>{device->Id().value()},
            xla::ifrt::SingleDeviceSharding::Create(device,
                                                    xla::ifrt::MemoryKind()),
            *thread_pool_);
    CHECK_OK(array.status());
    CHECK_OK((*array)->GetReadyFuture().Await());
    return *std::move(array);
  }

  // Returns the contents of the f32[4] `array`.
  static std::vector<float> Read(xla::ifrt::Array& array) {
    std::vector<float> values(4);
    CHECK_OK(array
                 .CopyToHostBuffer(values.data(), /*byte_strides=*/std::nullopt,
                                   xla::ifrt::ArrayCopySemantics::kAlwaysCopy)
                 .Await());
    return values;
  }

  // Returns the device address of the single buffer of `array`.
  static std::uintptr_t Address(xla::ifrt::Array& array) {
    auto* pjrt_array = xla::ifrt::dyn_cast<xla::ifrt::PjRtArray>(&array);
    CHECK(pjrt_array != nullptr);
    CHECK_EQ(pjrt_array->pjrt_buffers().size(), 1);
    xla::PjRtBuffer* const buffer = pjrt_array->pjrt_buffers().front().get();
    absl::StatusOr<std::uintptr_t> address =
        buffer->client()->UnsafeBufferPointer(buffer);
    CHECK_OK(address.status());
    return *address;
  }

  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
};

TEST_F(AliasingArrayTest, AliasesSourceMemoryWithoutCopying) {
  for (const AliasAccess access :
       {AliasAccess::kReadOnly, AliasAccess::kWritable}) {
    xla::ifrt::ArrayRef source = MakeArray(*client_, {1, 2, 3, 4});

    TF_ASSERT_OK_AND_ASSIGN(xla::ifrt::ArrayRef alias,
                            MakeAliasingArray(*source, *client_, access));

    EXPECT_THAT(alias.get(), Ne(source.get()));
    EXPECT_THAT(Address(*alias), Eq(Address(*source)));
    EXPECT_THAT(alias->dtype(), Eq(source->dtype()));
    EXPECT_THAT(alias->shape(), Eq(source->shape()));
    EXPECT_TRUE(alias->sharding().HasSamePartitioning(source->sharding()));
    EXPECT_THAT(Read(*alias), ElementsAre(1, 2, 3, 4));
    EXPECT_FALSE(source->IsDeleted());
  }
}

// Deleting the alias, which is what donating it to an execution does, leaves
// the source and its memory intact.
TEST_F(AliasingArrayTest, DeletingAliasKeepsSource) {
  xla::ifrt::ArrayRef source = MakeArray(*client_, {1, 2, 3, 4});
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef alias,
      MakeAliasingArray(*source, *client_, AliasAccess::kWritable));

  ASSERT_OK(alias->Delete().Await());
  alias.reset();

  EXPECT_FALSE(source->IsDeleted());
  EXPECT_THAT(Read(*source), ElementsAre(1, 2, 3, 4));
  // The source can be aliased again.
  TF_ASSERT_OK_AND_ASSIGN(
      alias, MakeAliasingArray(*source, *client_, AliasAccess::kWritable));
  EXPECT_THAT(Read(*alias), ElementsAre(1, 2, 3, 4));
}

// The alias shares ownership of the memory, so it outlives the source, even
// when the source is deleted explicitly.
TEST_F(AliasingArrayTest, AliasKeepsMemoryAfterSourceIsGone) {
  xla::ifrt::ArrayRef source = MakeArray(*client_, {1, 2, 3, 4});
  const std::uintptr_t address = Address(*source);
  TF_ASSERT_OK_AND_ASSIGN(
      xla::ifrt::ArrayRef alias,
      MakeAliasingArray(*source, *client_, AliasAccess::kReadOnly));

  ASSERT_OK(source->Delete().Await());
  source.reset();

  EXPECT_THAT(Address(*alias), Eq(address));
  EXPECT_THAT(Read(*alias), ElementsAre(1, 2, 3, 4));
}

TEST_F(AliasingArrayTest, RejectsArrayOfAnotherClient) {
  TF_ASSERT_OK_AND_ASSIGN(std::shared_ptr<xla::ifrt::Client> other_client,
                          xla::ifrt::test_util::GetClient());
  ASSERT_THAT(other_client.get(), Ne(client_.get()));
  xla::ifrt::ArrayRef source = MakeArray(*client_, {1, 2, 3, 4});

  EXPECT_THAT(MakeAliasingArray(*source, *other_client, AliasAccess::kReadOnly),
              StatusIs(absl::StatusCode::kUnimplemented,
                       HasSubstr("different client")));
}

}  // namespace
}  // namespace xla_serving

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

#include "runtime/core/host_buffer_test_utils.h"  // from @xla_serving

#include <cstdint>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving {
namespace {

using ::testing::HasSubstr;

TEST(AsHostBufferTest, Basic) {
  ASSERT_NO_FATAL_FAILURE(
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({4})));
  ASSERT_NO_FATAL_FAILURE(
      AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 2})));
}

TEST(AsHostBufferTest, Scalar) {
  ASSERT_NO_FATAL_FAILURE(AsHostBuffer<int32_t>(1));
}

TEST(AsHostBufferDeathTest, MismatchShape) {
  EXPECT_DEATH(AsHostBuffer<int32_t>({1, 2, 3, 4}, xla::ifrt::Shape({2, 3})),
               HasSubstr("does not match the shape"));
}

}  // namespace
}  // namespace xla_serving

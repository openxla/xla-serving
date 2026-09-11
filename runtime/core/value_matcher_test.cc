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

#include "runtime/core/value_matcher.h"  // from @xla_serving

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "runtime/core/value.h"  // from @xla_serving

namespace xla_serving {
namespace {

using ::testing::Eq;
using ::testing::HasSubstr;

TEST(ValueMatcherTest, SameTypeSameContentShallPass) {
  constexpr int kValue = 100;
  Value lhs(kValue);
  EXPECT_THAT(lhs, ValueOfType<int>(Eq(kValue)));
}

TEST(ValueMatcherTest, DifferentTypeShallFail) {
  Value lhs(100.0f);
  constexpr int kValue = 100;

  ::testing::StringMatchResultListener listener;
  ::testing::Matcher<const Value&> matcher = ValueOfType<int>(Eq(kValue));
  EXPECT_FALSE(matcher.MatchAndExplain(lhs, &listener));
  EXPECT_THAT(listener.str(),
              HasSubstr("because its payload type doesn't match"));
}

TEST(ValueMatcherTest, SameTypeDifferentContentsShallFail) {
  Value lhs(102);
  constexpr int kValue = 100;

  ::testing::StringMatchResultListener listener;
  ::testing::Matcher<const Value&> matcher = ValueOfType<int>(Eq(kValue));
  EXPECT_FALSE(matcher.MatchAndExplain(lhs, &listener));

  // The type mismatch message does not exit. Failure is due to the value
  // mismatch.
  EXPECT_TRUE(listener.str().empty());
}
}  // namespace
}  // namespace xla_serving

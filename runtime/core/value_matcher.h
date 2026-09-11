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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_MATCHER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_MATCHER_H_

#include <ostream>

#include "gtest/gtest.h"
#include "runtime/core/value.h"  // from @neptune

namespace xla_serving {
namespace internal {
// Implementation of ValueOfType matcher. See comments of ValueOfType for
// details.
template <typename T>
class ValueOfTypeImpl : public ::testing::MatcherInterface<const Value&> {
 public:
  explicit ValueOfTypeImpl(const ::testing::Matcher<const T&>& inner_matcher)
      : inner_matcher_(inner_matcher) {}

  void DescribeTo(std::ostream* os) const override {
    *os << "is a `Value` with payload type " << typeid(T).name() << " and ";
    inner_matcher_.DescribeTo(os);
  }

  void DescribeNegationTo(std::ostream* os) const override {
    *os << "does not have payload type of " << typeid(T).name()
        << " or ";
    inner_matcher_.DescribeNegationTo(os);
  }
  bool MatchAndExplain(
      const Value& value,
      ::testing::MatchResultListener* listener) const override {
    if (!value.IsType<T>()) {
      *listener << "because its payload type doesn't match "
                << typeid(T).name();
      return false;
    }

    return inner_matcher_.MatchAndExplain(value.Get<T>(), listener);
  }

 private:
  ::testing::Matcher<const T&> inner_matcher_;
};
}  // namespace internal

// A matcher for comparing a Value instance with an expected Type and an
// expected content matcher.
//
// After validating the type of a Value is expected, it content is matched
// by the corresponding inner matcher.
//
// Use this like:
//  Value lhs;
//  constexpr int kTarget = 100;
//  EXPECT_THAT(lhs, ValueOfType<int>(Eq(target)));
//
template <typename T>
class ValueOfType {
 public:
  explicit ValueOfType(const ::testing::Matcher<const T&>& inner_matcher)
      : inner_matcher_(inner_matcher) {}

  // Matchers depend on implicit casts. Do not make explicit.
  operator ::testing::Matcher<const Value&>() const {  // NOLINT
    return ::testing::MakeMatcher(
        new internal::ValueOfTypeImpl<T>(inner_matcher_));
  }

 private:
  const ::testing::Matcher<const T&>& inner_matcher_;
};

}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_VALUE_MATCHER_H_

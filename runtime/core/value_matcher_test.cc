#include "third_party/neptune/runtime/core/value_matcher.h"

#include "testing/base/public/gmock.h"
#include "testing/base/public/gunit.h"
#include "third_party/neptune/runtime/core/value.h"

namespace jsv {
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
}  // namespace jsv

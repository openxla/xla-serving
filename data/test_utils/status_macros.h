#ifndef THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_STATUS_MACROS_H_
#define THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_STATUS_MACROS_H_

// IWYU pragma: always_keep

#ifndef EXPECT_OK
#define EXPECT_OK(expr) EXPECT_TRUE((expr).ok())
#endif

#ifndef ASSERT_OK
#define ASSERT_OK(expr) ASSERT_TRUE((expr).ok())
#endif

#ifndef ASSERT_OK_AND_ASSIGN
#define ASSERT_OK_AND_ASSIGN_CONCAT2(x, y) x##y
#define ASSERT_OK_AND_ASSIGN_CONCAT(x, y) ASSERT_OK_AND_ASSIGN_CONCAT2(x, y)

#define ASSERT_OK_AND_ASSIGN(lhs, rexpr)                                       \
  auto ASSERT_OK_AND_ASSIGN_CONCAT(status_or_, __LINE__) = (rexpr);            \
  ASSERT_TRUE(ASSERT_OK_AND_ASSIGN_CONCAT(status_or_, __LINE__).ok());         \
  lhs = std::move(*ASSERT_OK_AND_ASSIGN_CONCAT(status_or_, __LINE__))
#endif

#endif  // THIRD_PARTY_NEPTUNE_DATA_TEST_UTILS_STATUS_MACROS_H_

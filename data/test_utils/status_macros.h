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

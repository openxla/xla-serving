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

#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAre;

TEST(EmbeddingTableTest, CreateSuccess) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array array(absl::MakeConstSpan(data), Shape({3, 2}));
  ASSERT_OK_AND_ASSIGN(auto table,
                       EmbeddingTable::Create("test_table", std::move(array)));
  EXPECT_EQ(table->Name(), "test_table");
}

TEST(EmbeddingTableTest, CreateFailureInvalidShape) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array array(absl::MakeConstSpan(data), Shape({6}));
  EXPECT_THAT(EmbeddingTable::Create("test_table", std::move(array)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EmbeddingTableTest, CreateFailureEmptyName) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array array(absl::MakeConstSpan(data), Shape({3, 2}));
  EXPECT_THAT(EmbeddingTable::Create("", std::move(array)),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EmbeddingTableTest, LookupSuccess) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array array(absl::MakeConstSpan(data), Shape({3, 2}));
  ASSERT_OK_AND_ASSIGN(auto table,
                       EmbeddingTable::Create("test_table", std::move(array)));

  std::vector<int64_t> ids = {0, 2};
  ASSERT_OK_AND_ASSIGN(Array results, table->Lookup(ids));

  EXPECT_THAT(results.ShapeSpan(), ElementsAre(2, 2));
  EXPECT_THAT(results.AsSpan<float>(), ElementsAre(1.0, 2.0, 5.0, 6.0));
}

TEST(EmbeddingTableTest, LookupFailureIdOutOfRange) {
  float data[] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  Array array(absl::MakeConstSpan(data), Shape({3, 2}));
  ASSERT_OK_AND_ASSIGN(auto table,
                       EmbeddingTable::Create("test_table", std::move(array)));

  std::vector<int64_t> ids = {0, 3};
  EXPECT_THAT(table->Lookup(ids), StatusIs(absl::StatusCode::kInvalidArgument));

  std::vector<int64_t> ids2 = {-1};
  EXPECT_THAT(table->Lookup(ids2),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(EmbeddingTableTest, LookupUnsupportedDType) {
  int32_t data[] = {1, 2, 3, 4, 5, 6};
  Array array(absl::MakeConstSpan(data), Shape({3, 2}));
  ASSERT_OK_AND_ASSIGN(auto table,
                       EmbeddingTable::Create("test_table", std::move(array)));

  std::vector<int64_t> ids = {0, 2};
  EXPECT_THAT(table->Lookup(ids), StatusIs(absl::StatusCode::kInvalidArgument));
}

}  // namespace
}  // namespace xla_serving::data::cpu_embedding

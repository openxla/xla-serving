#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune

#include <cstdint>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data::cpu_embedding {
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
}  // namespace neptune::data::cpu_embedding

#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune

#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

using ::absl_testing::StatusIs;
using ::testing::ElementsAreArray;
using ::testing::HasSubstr;

TEST(FeatureBatchCsrInputTest, Create) {
  std::vector<int64_t> embedding_ids = {0, 1, 2, 3, 4};
  std::vector<int32_t> row_pointers = {0, 2, 4, 5};
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  ASSERT_OK_AND_ASSIGN(FeatureBatchCsrInput input,
                       FeatureBatchCsrInput::Create("feature", embedding_ids,
                                                    row_pointers, weights));

  EXPECT_EQ(input.NumSamples(), 3);
  EXPECT_THAT(input.EmbeddingIds(), ElementsAreArray(embedding_ids));
  EXPECT_THAT(input.RowPointers(), ElementsAreArray(row_pointers));
  EXPECT_THAT(input.Weights(), ElementsAreArray(weights));
}

TEST(FeatureBatchCsrInputTest, InvalidRowPointersSize) {
  std::vector<int64_t> embedding_ids = {0, 1, 2, 3, 4};
  std::vector<int32_t> row_pointers = {0};
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  EXPECT_THAT(
      FeatureBatchCsrInput::Create("feature", embedding_ids, row_pointers,
                                   weights),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Row pointers must have at least 2 elements")));
}

TEST(FeatureBatchCsrInputTest, MismatchedEmbeddingIdsSize) {
  std::vector<int64_t> embedding_ids = {0, 1, 2, 3, 4};
  std::vector<int32_t> row_pointers = {0, 2, 4};
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};

  EXPECT_THAT(FeatureBatchCsrInput::Create("feature", embedding_ids,
                                           row_pointers, weights),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Embedding ids size (5) does not match last "
                                 "element of row pointers (4)")));
}

TEST(FeatureBatchCsrInputTest, UnsortedRowPointers) {
  std::vector<int64_t> embedding_ids = {0, 1, 2, 3, 4};
  std::vector<int32_t> row_pointers = {0, 4, 2, 5};
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  EXPECT_THAT(FeatureBatchCsrInput::Create("feature", embedding_ids,
                                           row_pointers, weights),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Row pointers are not sorted")));
}

TEST(FeatureBatchCsrInputTest, MismatchedWeightsSize) {
  std::vector<int64_t> embedding_ids = {0, 1, 2, 3, 4};
  std::vector<int32_t> row_pointers = {0, 2, 4, 5};
  std::vector<float> weights = {1.0f, 2.0f, 3.0f, 4.0f};

  EXPECT_THAT(
      FeatureBatchCsrInput::Create("feature", embedding_ids, row_pointers,
                                   weights),
      StatusIs(absl::StatusCode::kInvalidArgument,
               HasSubstr("Weights size (4) does not match expected size (5)")));
}

}  // namespace
}  // namespace neptune::data::cpu_embedding

#include "data/cpu_embedding/cc/compute_activations.h"  // from @neptune

#include <cstdint>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointwise;

template <class T>
T GetToler() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-6f;
  } else if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
    return static_cast<Eigen::bfloat16>(1e-2);
  } else {
    static_assert(false, "Unsupported type");
  }
}

template <typename T>
class ComputeActivationsTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(ComputeActivationsTest);

MATCHER_P(Approx, epsilon, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < epsilon;
}

TYPED_TEST_P(ComputeActivationsTest, SumCombine) {
  using T = TypeParam;
  // Embedding table:
  // ID 0: [0.1, 0.2, 0.3]
  // ID 1: [0.4, 0.5, 0.6]
  // ID 2: [0.7, 0.8, 0.9]
  std::vector<T> table_data = {T(0.1), T(0.2), T(0.3), T(0.4), T(0.5),
                               T(0.6), T(0.7), T(0.8), T(0.9)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({3, 3}));
  ASSERT_OK_AND_ASSIGN(auto embedding_table,
                       EmbeddingTable::Create("table", table_array));

  // Batch:
  // Sample 0: IDs [0, 1], Weights [1.0, 1.0] -> [0.1, 0.2, 0.3] + [0.4, 0.5,
  // 0.6] = [0.5, 0.7, 0.9] Sample 1: IDs [2], Weights [1.0] -> [0.7, 0.8, 0.9]
  // Sample 2: IDs [0, 2], Weights [0.5, 2.0] -> 0.5 * [0.1, 0.2, 0.3] + 2.0 *
  // [0.7, 0.8, 0.9]
  //                                      -> [0.05, 0.1, 0.15] + [1.4, 1.6, 1.8]
  //                                      = [1.45, 1.7, 1.95]
  std::vector<int32_t> row_splits = {0, 2, 3, 5};
  std::vector<int64_t> embedding_ids = {0, 1, 2, 0, 2};
  std::vector<float> weights = {1.0, 1.0, 1.0, 0.5, 2.0};
  ASSERT_OK_AND_ASSIGN(FeatureBatchCsrInput feature_batch,
                       FeatureBatchCsrInput::Create("feature", embedding_ids,
                                                    row_splits, weights));

  ASSERT_OK_AND_ASSIGN(Array result, ComputeEmbeddingActivations(
                                         feature_batch, *embedding_table));

  EXPECT_EQ(result.GetDType(), NativeToDType<T>::kDType);
  EXPECT_THAT(result.ShapeSpan(), ElementsAre(3, 3));
  EXPECT_THAT(
      result.AsSpan<T>(),
      Pointwise(Approx(GetToler<T>()), {T(0.5), T(0.7), T(0.9), T(0.7), T(0.8),
                                        T(0.9), T(1.45), T(1.7), T(1.95)}));
}

TYPED_TEST_P(ComputeActivationsTest, MeanCombine) {
  using T = TypeParam;
  // Embedding table:
  // ID 0: [1.0, 2.0]
  // ID 1: [3.0, 4.0]
  std::vector<T> table_data = {T(1.0), T(2.0), T(3.0), T(4.0)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_table,
      EmbeddingTable::Create("table", table_array,
                             EmbeddingTable::CombineMode::kMean));

  // Batch:
  // Sample 0: IDs [0, 1], Weights [0.5, 0.5] -> (0.5 * [1.0, 2.0] + 0.5 *
  // [3.0, 4.0]) / (0.5 + 0.5) = ([0.5, 1.0] + [1.5, 2.0]) / 1.0 = [2.0, 3.0]
  std::vector<int32_t> row_splits = {0, 2};
  std::vector<int64_t> embedding_ids = {0, 1};
  std::vector<float> weights = {0.5, 0.5};
  ASSERT_OK_AND_ASSIGN(FeatureBatchCsrInput feature_batch,
                       FeatureBatchCsrInput::Create("feature", embedding_ids,
                                                    row_splits, weights));

  ASSERT_OK_AND_ASSIGN(Array result, ComputeEmbeddingActivations(
                                         feature_batch, *embedding_table));

  EXPECT_EQ(result.GetDType(), NativeToDType<T>::kDType);
  EXPECT_THAT(result.ShapeSpan(), ElementsAre(1, 2));
  EXPECT_THAT(result.AsSpan<T>(),
              Pointwise(Approx(GetToler<T>()), {T(2.0), T(3.0)}));
}

TYPED_TEST_P(ComputeActivationsTest, SqrtNCombine) {
  using T = TypeParam;
  // Embedding table:
  // ID 0: [1.0, 2.0]
  // ID 1: [3.0, 4.0]
  std::vector<T> table_data = {T(1.0), T(2.0), T(3.0), T(4.0)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(
      auto embedding_table,
      EmbeddingTable::Create("table", table_array,
                             EmbeddingTable::CombineMode::kSqrtN));

  // Batch:
  // Sample 0: IDs [0, 1] -> ([1.0, 2.0] + [3.0, 4.0]) / sqrt(2) =
  // [2.82842712, 4.24264068]
  std::vector<int32_t> row_splits = {0, 2};
  std::vector<int64_t> embedding_ids = {0, 1};
  ASSERT_OK_AND_ASSIGN(
      FeatureBatchCsrInput feature_batch,
      FeatureBatchCsrInput::Create("feature", embedding_ids, row_splits, {}));

  ASSERT_OK_AND_ASSIGN(Array result, ComputeEmbeddingActivations(
                                         feature_batch, *embedding_table));

  EXPECT_EQ(result.GetDType(), NativeToDType<T>::kDType);
  EXPECT_THAT(result.ShapeSpan(), ElementsAre(1, 2));
  EXPECT_THAT(result.AsSpan<T>(),
              Pointwise(Approx(GetToler<T>()), {T(2.82842712), T(4.24264068)}));
}

TYPED_TEST_P(ComputeActivationsTest, SequenceAssembly) {
  using T = TypeParam;
  std::vector<T> table_data(40);
  for (int i = 0; i < 40; ++i) {
    table_data[i] = T(static_cast<float>(i) * 0.1f);
  }
  Array table_array(absl::MakeConstSpan(table_data), Shape({10, 4}));
  ASSERT_OK_AND_ASSIGN(auto embedding_table,
                       EmbeddingTable::Create("table", table_array));

  std::vector<int32_t> row_splits = {0, 2, 4, 5};
  std::vector<int64_t> embedding_ids = {1, 3, 4, 7, 8};
  ASSERT_OK_AND_ASSIGN(
      FeatureBatchCsrInput feature_batch,
      FeatureBatchCsrInput::Create("feature", embedding_ids, row_splits, {}));

  constexpr int64_t kMaxSequenceLength = 5;
  ASSERT_OK_AND_ASSIGN(Array result,
                       LookupEmbeddingSequence(feature_batch, *embedding_table,
                                               kMaxSequenceLength));

  EXPECT_EQ(result.GetDType(), NativeToDType<T>::kDType);
  EXPECT_THAT(result.ShapeSpan(), ElementsAre(3, kMaxSequenceLength, 4));
  EXPECT_THAT(result.AsSpan<T>(),
              Pointwise(Approx(GetToler<T>()),
                        {// Sample sequence 0
                         T(0.4), T(0.5), T(0.6), T(0.7), T(1.2), T(1.3), T(1.4),
                         T(1.5), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0),
                         T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0),
                         // Sample sequence 1
                         T(1.6), T(1.7), T(1.8), T(1.9), T(2.8), T(2.9), T(3.0),
                         T(3.1), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0),
                         T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0),
                         // Sample sequence 2
                         T(3.2), T(3.3), T(3.4), T(3.5), T(0.0), T(0.0), T(0.0),
                         T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0),
                         T(0.0), T(0.0), T(0.0), T(0.0), T(0.0), T(0.0)}));
}

TYPED_TEST_P(ComputeActivationsTest,
             ComputeEmbeddingActivationsWithProvidedBuffer) {
  using T = TypeParam;
  // Embedding table:
  // ID 0: [0.1, 0.2, 0.3]
  // ID 1: [0.4, 0.5, 0.6]
  // ID 2: [0.7, 0.8, 0.9]
  std::vector<T> table_data = {T(0.1), T(0.2), T(0.3), T(0.4), T(0.5),
                               T(0.6), T(0.7), T(0.8), T(0.9)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({3, 3}));
  ASSERT_OK_AND_ASSIGN(auto embedding_table,
                       EmbeddingTable::Create("table", table_array));

  // Batch:
  // Sample 0: IDs [0, 1], Weights [1.0, 1.0] -> [0.1, 0.2, 0.3] + [0.4, 0.5,
  // 0.6] = [0.5, 0.7, 0.9] Sample 1: IDs [2], Weights [1.0] -> [0.7, 0.8, 0.9]
  // Sample 2: IDs [0, 2], Weights [0.5, 2.0] -> 0.5 * [0.1, 0.2, 0.3] + 2.0 *
  // [0.7, 0.8, 0.9]
  //                                      -> [0.05, 0.1, 0.15] + [1.4, 1.6, 1.8]
  //                                      = [1.45, 1.7, 1.95]
  std::vector<int32_t> row_splits = {0, 2, 3, 5};
  std::vector<int64_t> embedding_ids = {0, 1, 2, 0, 2};
  std::vector<float> weights = {1.0, 1.0, 1.0, 0.5, 2.0};
  ASSERT_OK_AND_ASSIGN(FeatureBatchCsrInput feature_batch,
                       FeatureBatchCsrInput::Create("feature", embedding_ids,
                                                    row_splits, weights));
  std::vector<T> buffer(3 * 3);
  ASSERT_OK(ComputeEmbeddingActivations(
      feature_batch.EmbeddingIds(), feature_batch.RowPointers(),
      feature_batch.Weights(), *embedding_table, absl::MakeSpan(buffer)));
  EXPECT_THAT(buffer, Pointwise(Approx(GetToler<T>()),
                                {T(0.5), T(0.7), T(0.9), T(0.7), T(0.8), T(0.9),
                                 T(1.45), T(1.7), T(1.95)}));
}

REGISTER_TYPED_TEST_SUITE_P(ComputeActivationsTest, SumCombine, MeanCombine,
                            SqrtNCombine, SequenceAssembly,
                            ComputeEmbeddingActivationsWithProvidedBuffer);

using MyTypes = ::testing::Types<float, Eigen::bfloat16>;

template <typename T>
std::string TypeName();
template <>
std::string TypeName<float>() {
  return "float32";
}
template <>
std::string TypeName<Eigen::bfloat16>() {
  return "bfloat16";
}

struct TypeNamer {
  template <typename T>
  static std::string GetName(int) {
    return TypeName<T>();
  }
};

INSTANTIATE_TYPED_TEST_SUITE_P(test, ComputeActivationsTest, MyTypes,
                               TypeNamer);

}  // namespace
}  // namespace neptune::data::cpu_embedding

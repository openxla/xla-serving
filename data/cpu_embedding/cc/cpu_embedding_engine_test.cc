// Copyright 2026 Google LLC
// Copyright 2026 Arm Ltd. and affiliates
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

#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointwise;
using ::testing::SizeIs;

template <class T>
T GetToler() {
  if constexpr (std::is_same_v<T, float>) {
    return 1e-6f;
  } else if constexpr (std::is_same_v<T, Eigen::bfloat16>) {
    return static_cast<Eigen::bfloat16>(2e-2);
  } else {
    static_assert(false, "Unsupported type");
  }
}

MATCHER_P(Approx, epsilon, "") {
  return std::abs(static_cast<float>(std::get<0>(arg) - std::get<1>(arg))) <
         epsilon;
}

template <typename T>
class CpuEmbeddingEngineTest : public ::testing::Test {};

TYPED_TEST_SUITE_P(CpuEmbeddingEngineTest);

TYPED_TEST_P(CpuEmbeddingEngineTest, BasicExecute) {
  using T = TypeParam;
  CpuEmbeddingEngine engine;

  // Create and add an embedding table of shape (10, 4).
  std::vector<T> table_data = {
      T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6), T(0.7), T(0.8),
      T(0.9), T(1.0), T(1.1), T(1.2), T(1.3), T(1.4), T(1.5), T(1.6),
      T(1.7), T(1.8), T(1.9), T(2.0), T(2.1), T(2.2), T(2.3), T(2.4),
      T(2.5), T(2.6), T(2.7), T(2.8), T(2.9), T(3.0), T(3.1), T(3.2),
      T(3.3), T(3.4), T(3.5), T(3.6), T(3.7), T(3.8), T(3.9), T(4.0)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({10, 4}));
  ASSERT_OK_AND_ASSIGN(std::shared_ptr<EmbeddingTable> table,
                       EmbeddingTable::Create("test_table", table_array));
  ASSERT_OK(engine.AddLocalTable("test_table", table));

  // Register feature specs.
  ASSERT_OK(engine.RegisterFeatureSpec(
      FeatureSpec{.name = "feature_a", .table_name = "test_table"}));
  ASSERT_OK(engine.RegisterFeatureSpec(
      FeatureSpec{.name = "feature_b", .table_name = "test_table"}));

  // Create feature batches. Each has two samples, with different valency.
  std::vector<int32_t> row_splits_a = {0, 2, 3};
  std::vector<int64_t> embedding_ids_a = {0, 1, 2};
  ASSERT_OK_AND_ASSIGN(
      FeatureBatchCsrInput batch_a,
      FeatureBatchCsrInput::Create("feature_a", embedding_ids_a, row_splits_a,
                                   /*weights=*/{}));

  std::vector<int32_t> row_splits_b = {0, 1, 3};
  std::vector<int64_t> embedding_ids_b = {1, 0, 2};
  ASSERT_OK_AND_ASSIGN(
      FeatureBatchCsrInput batch_b,
      FeatureBatchCsrInput::Create("feature_b", embedding_ids_b, row_splits_b,
                                   /*weights=*/{}));

  std::vector<FeatureBatchCsrInput> batches;
  batches.push_back(std::move(batch_a));
  batches.push_back(std::move(batch_b));

  // Execute
  ASSERT_OK_AND_ASSIGN(std::vector<Array> results, engine.Execute(batches));

  // Verify results
  ASSERT_THAT(results, SizeIs(2));

  // Feature A:
  // Sample 0: ID [0, 1] -> [0.1, 0.2, 0.3, 0.4] + [0.5, 0.6, 0.7, 0.8] = [0.6,
  // 0.8, 1.0, 1.2] Sample 1: ID [2] -> [0.9, 1.0, 1.1, 1.2]
  EXPECT_THAT(results[0].ShapeSpan(), ElementsAre(2, 4));
  EXPECT_THAT(
      results[0].AsSpan<T>(),
      Pointwise(Approx(GetToler<T>()), {T(0.6), T(0.8), T(1.0), T(1.2), T(0.9),
                                        T(1.0), T(1.1), T(1.2)}));

  // Feature B:
  // Sample 0: ID [1] -> [0.5, 0.6, 0.7, 0.8]
  // Sample 1: ID [0, 2] -> [0.1, 0.2, 0.3, 0.4] + [0.9, 1.0, 1.1, 1.2] =
  // [1.0, 1.2, 1.4, 1.6]
  EXPECT_THAT(results[1].ShapeSpan(), ElementsAre(2, 4));
  EXPECT_THAT(
      results[1].AsSpan<T>(),
      Pointwise(Approx(GetToler<T>()), {T(0.5), T(0.6), T(0.7), T(0.8), T(1.0),
                                        T(1.2), T(1.4), T(1.6)}));
}

TYPED_TEST_P(CpuEmbeddingEngineTest, ExecuteWithProvidedBuffer) {
  using T = TypeParam;
  CpuEmbeddingEngine engine;

  // Create and add an embedding table of shape (10, 4).
  std::vector<T> table_data = {
      T(0.1), T(0.2), T(0.3), T(0.4), T(0.5), T(0.6), T(0.7), T(0.8),
      T(0.9), T(1.0), T(1.1), T(1.2), T(1.3), T(1.4), T(1.5), T(1.6),
      T(1.7), T(1.8), T(1.9), T(2.0), T(2.1), T(2.2), T(2.3), T(2.4),
      T(2.5), T(2.6), T(2.7), T(2.8), T(2.9), T(3.0), T(3.1), T(3.2),
      T(3.3), T(3.4), T(3.5), T(3.6), T(3.7), T(3.8), T(3.9), T(4.0)};
  Array table_array(absl::MakeConstSpan(table_data), Shape({10, 4}));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EmbeddingTable> table,
      EmbeddingTable::Create("test_table", table_array,
                             EmbeddingTable::CombineMode::kMean));
  ASSERT_OK(engine.AddLocalTable("test_table", table));

  // Register feature spec.
  ASSERT_OK(engine.RegisterFeatureSpec(
      FeatureSpec{.name = "test_feature", .table_name = "test_table"}));

  // Create feature batches. Each has two samples, with different valency.
  std::vector<int32_t> row_splits = {0, 2, 3, 7};
  std::vector<int64_t> embedding_ids = {0, 1, 2, 4, 6, 8, 4};
  ASSERT_OK_AND_ASSIGN(
      FeatureBatchCsrInput batch,
      FeatureBatchCsrInput::Create("test_feature", embedding_ids, row_splits,
                                   /*weights=*/{}));

  // Execute
  std::vector<T> buffer(3 * 4);  // 3 samples * 4 embedding_dim
  ASSERT_OK(engine.Execute<T>(
      batch.Name(), batch.EmbeddingIds(), batch.RowPointers(), batch.Weights(),
      [&buffer](size_t size) -> absl::Span<T> {
        return absl::MakeSpan(buffer.data(), buffer.size()).subspan(0, size);
      }));
  // Feature:
  // Sample 0: ID [0, 1] -> ([0.1, 0.2, 0.3, 0.4] + [0.5, 0.6, 0.7, 0.8]) / 2 =
  // [0.3, 0.4, 0.5, 0.6]
  // Sample 1: ID [2] -> [0.9, 1.0, 1.1, 1.2]
  // Sample 2: ID [4, 6, 8, 4] -> ([1.7, 1.8, 1.9, 2.0] * 2 +
  // [2.5, 2.6, 2.7, 2.8]
  // + [3.3, 3.4, 3.5, 3.6]) / 4 = [9.2, 9.6, 10.0, 10.4] / 4 = [2.3, 2.4, 2.5,
  // 2.6]
  EXPECT_THAT(buffer,
              Pointwise(Approx(GetToler<T>()),
                        {T(0.3), T(0.4), T(0.5), T(0.6), T(0.9), T(1.0), T(1.1),
                         T(1.2), T(2.3), T(2.4), T(2.5), T(2.6)}));
}

TYPED_TEST_P(CpuEmbeddingEngineTest, ExecuteWithInvalidRowPointersFails) {
  using T = TypeParam;
  CpuEmbeddingEngine engine;

  // Create and add an embedding table of shape (10, 4).
  std::vector<T> table_data(40, T(0.1));
  Array table_array(absl::MakeConstSpan(table_data), Shape({10, 4}));
  ASSERT_OK_AND_ASSIGN(
      std::shared_ptr<EmbeddingTable> table,
      EmbeddingTable::Create("test_table", table_array,
                             EmbeddingTable::CombineMode::kMean));
  ASSERT_OK(engine.AddLocalTable("test_table", table));

  // Register feature spec.
  ASSERT_OK(engine.RegisterFeatureSpec(
      FeatureSpec{.name = "test_feature", .table_name = "test_table"}));

  // Create invalid row pointers.
  std::vector<int64_t> embedding_ids = {0, 1, 2, 4};
  std::vector<T> buffer(3 * 4);
  auto allocator = [&buffer](size_t size) -> absl::Span<T> {
    return absl::MakeSpan(buffer.data(), buffer.size()).subspan(0, size);
  };

  // 1. Non-monotonic row pointers
  std::vector<int32_t> rp_non_monotonic = {0, 2, 1, 4};
  EXPECT_FALSE(engine
                   .Execute<T>("test_feature", embedding_ids, rp_non_monotonic,
                               {}, allocator)
                   .ok());

  // 2. Negative row pointers
  std::vector<int32_t> rp_negative = {-1, 1, 3, 4};
  EXPECT_FALSE(
      engine
          .Execute<T>("test_feature", embedding_ids, rp_negative, {}, allocator)
          .ok());

  // 3. Out of bounds row pointers (final element > embedding_ids.size())
  std::vector<int32_t> rp_out_of_bounds = {0, 1, 3, 5};
  EXPECT_FALSE(engine
                   .Execute<T>("test_feature", embedding_ids, rp_out_of_bounds,
                               {}, allocator)
                   .ok());

  // 4. Weights size mismatch (too small)
  std::vector<int32_t> rp_ok = {0, 1, 3, 4};
  std::vector<float> weights_small = {1.0, 0.5};  // needs 4 elements
  EXPECT_FALSE(engine
                   .Execute<T>("test_feature", embedding_ids, rp_ok,
                               weights_small, allocator)
                   .ok());
}

REGISTER_TYPED_TEST_SUITE_P(CpuEmbeddingEngineTest, BasicExecute,
                            ExecuteWithProvidedBuffer,
                            ExecuteWithInvalidRowPointersFails);

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

INSTANTIATE_TYPED_TEST_SUITE_P(test, CpuEmbeddingEngineTest, MyTypes,
                               TypeNamer);

absl::Status AddFeature(
    CpuEmbeddingEngine* engine, absl::string_view name,
    absl::Span<const float> data, int64_t dimension,
    EmbeddingTable::CombineMode mode = EmbeddingTable::CombineMode::kSum) {
  const std::string feature_name(name);
  const std::string table_name = feature_name + "_table";
  Array array(
      data, Shape({static_cast<int64_t>(data.size()) / dimension, dimension}));
  auto table = EmbeddingTable::Create(table_name, std::move(array), mode);
  if (!table.ok()) return table.status();
  absl::Status status = engine->AddLocalTable(table_name, std::move(*table));
  if (!status.ok()) return status;
  return engine->RegisterFeatureSpec(
      FeatureSpec{.name = feature_name, .table_name = table_name});
}

// Covers grouped execution across dimensions, modes, and weights.
TEST(CpuEmbeddingEngineBatchTest, ExecutesMixedDimensionsModesAndWeights) {
  CpuEmbeddingEngine engine(CpuEmbeddingEngine::Options{.max_cpu_slots = 2});
  std::vector<float> mean_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> sqrt_data = {2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f};
  ASSERT_OK(AddFeature(&engine, "mean", mean_data, 2,
                       EmbeddingTable::CombineMode::kMean));
  ASSERT_OK(AddFeature(&engine, "sqrt", sqrt_data, 3,
                       EmbeddingTable::CombineMode::kSqrtN));
  const std::vector<int64_t> ids = {0, 1};
  const std::vector<int32_t> rows = {0, 2};
  const std::vector<float> weights = {0.25f, 0.75f};
  std::vector<float> mean_output(2, -11.0f);
  std::vector<float> sqrt_output(3, -13.0f);
  std::vector<CpuEmbeddingEngine::BatchRequest> requests;
  requests.push_back({
      .feature_name = "mean",
      .embedding_ids = ids,
      .row_pointers = rows,
      .weights = weights,
      .output_allocator =
          [&](size_t size) {
            return absl::MakeSpan(mean_output).subspan(0, size);
          },
  });
  requests.push_back({
      .feature_name = "sqrt",
      .embedding_ids = ids,
      .row_pointers = rows,
      .weights = {},
      .output_allocator =
          [&](size_t size) {
            return absl::MakeSpan(sqrt_output).subspan(0, size);
          },
  });

  ASSERT_OK(engine.ExecuteBatch(absl::MakeSpan(requests)));

  EXPECT_THAT(mean_output, ElementsAre(2.5f, 3.5f));
  const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
  EXPECT_THAT(sqrt_output,
              Pointwise(Approx(1e-6f), {3.0f * inv_sqrt_2, 7.0f * inv_sqrt_2,
                                        11.0f * inv_sqrt_2}));
}

// Covers whole-group feature resolution before allocation.
TEST(CpuEmbeddingEngineBatchTest, ResolvesAllRequestsBeforeAllocation) {
  CpuEmbeddingEngine engine(CpuEmbeddingEngine::Options{.max_cpu_slots = 2});
  std::vector<float> table_data = {1.0f, 2.0f};
  ASSERT_OK(AddFeature(&engine, "feature", table_data, 2));
  const std::vector<int64_t> ids = {0};
  const std::vector<int32_t> rows = {0, 1};
  std::vector<float> output0(2, 29.0f);
  std::vector<float> output1(2, 31.0f);
  int allocation_count = 0;
  std::vector<CpuEmbeddingEngine::BatchRequest> requests;
  requests.push_back({
      .feature_name = "feature",
      .embedding_ids = ids,
      .row_pointers = rows,
      .weights = {},
      .output_allocator =
          [&](size_t size) {
            ++allocation_count;
            return absl::MakeSpan(output0).subspan(0, size);
          },
  });
  requests.push_back({
      .feature_name = "missing",
      .embedding_ids = ids,
      .row_pointers = rows,
      .weights = {},
      .output_allocator =
          [&](size_t size) {
            ++allocation_count;
            return absl::MakeSpan(output1).subspan(0, size);
          },
  });

  EXPECT_THAT(engine.ExecuteBatch(absl::MakeSpan(requests)),
              absl_testing::StatusIs(absl::StatusCode::kNotFound,
                                     testing::HasSubstr("missing")));
  EXPECT_EQ(allocation_count, 0);
  EXPECT_THAT(output0, ElementsAre(29.0f, 29.0f));
  EXPECT_THAT(output1, ElementsAre(31.0f, 31.0f));
}

// Covers snapshot lifetime across a re-entrant Clear().
TEST(CpuEmbeddingEngineBatchTest, SnapshotSurvivesClearDuringAllocation) {
  CpuEmbeddingEngine engine(CpuEmbeddingEngine::Options{.max_cpu_slots = 2});
  std::vector<float> table_data = {1.0f, 2.0f};
  ASSERT_OK(AddFeature(&engine, "feature", table_data, 2));
  const std::vector<int64_t> ids = {0};
  const std::vector<int32_t> rows = {0, 1};
  std::vector<float> output(2, -1.0f);
  int allocation_count = 0;
  std::vector<CpuEmbeddingEngine::BatchRequest> requests;
  requests.push_back({
      .feature_name = "feature",
      .embedding_ids = ids,
      .row_pointers = rows,
      .weights = {},
      .output_allocator =
          [&](size_t size) {
            ++allocation_count;
            engine.Clear();
            return absl::MakeSpan(output).subspan(0, size);
          },
  });

  ASSERT_OK(engine.ExecuteBatch(absl::MakeSpan(requests)));

  EXPECT_THAT(output, ElementsAre(1.0f, 2.0f));
  EXPECT_THAT(engine.ExecuteBatch(absl::MakeSpan(requests)),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
  EXPECT_EQ(allocation_count, 1);
}

}  // namespace
}  // namespace xla_serving::data::cpu_embedding

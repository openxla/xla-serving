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

#include "data/cpu_embedding/cc/embedding_bag.h"  // from @xla_serving

#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding {
namespace {

using ::testing::ElementsAre;
using ::testing::Pointwise;

MATCHER_P(Approx, epsilon, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < epsilon;
}

absl::StatusOr<std::unique_ptr<EmbeddingTable>> MakeTable(
    absl::Span<const float> data, int64_t dimension,
    EmbeddingTable::CombineMode mode = EmbeddingTable::CombineMode::kSum) {
  return EmbeddingTable::Create(
      "table", Array(data, Shape({static_cast<int64_t>(data.size()) / dimension,
                                   dimension})),
      mode);
}

absl::AnyInvocable<absl::Span<float>(size_t)> MakeAllocator(
    std::vector<float>* output) {
  return [output](size_t size) {
    return absl::MakeSpan(*output).subspan(0, size);
  };
}

internal::EmbeddingBagRequest MakeRequest(
    absl::Span<const int64_t> ids, absl::Span<const int32_t> rows,
    const EmbeddingTable* table, absl::Span<const float> weights,
    absl::AnyInvocable<absl::Span<float>(size_t)>* allocator) {
  return {.embedding_ids = ids,
          .row_pointers = rows,
          .embedding_table = table,
          .weights = weights,
          .output_allocator = allocator};
}

// Covers portable auto dispatch and fresh-output semantics.
TEST(EmbeddingBagBatchTest, AutoUsesPortableWithFreshOutputSemantics) {
  std::vector<float> table_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f};
  ASSERT_OK_AND_ASSIGN(auto table,
                       MakeTable(table_data, 3,
                                 EmbeddingTable::CombineMode::kMean));
  const std::vector<int64_t> ids = {0, 2, 1};
  const std::vector<int32_t> rows = {0, 2, 2, 3};
  const std::vector<float> weights = {0.5f, 1.5f, 2.0f};
  std::vector<float> output(9, -99.0f);
  internal::EmbeddingBagBackend selected =
      internal::EmbeddingBagBackend::kAuto;

  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      ids, rows, weights, *table, absl::MakeSpan(output),
      internal::EmbeddingBagBackend::kAuto, &selected));

  EXPECT_EQ(selected, internal::EmbeddingBagBackend::kPortable);
  EXPECT_THAT(output, ElementsAre(5.5f, 6.5f, 7.5f, 0.0f, 0.0f, 0.0f,
                                  4.0f, 5.0f, 6.0f));
}

// Covers mixed dimensions and combiners on one worker pool.
TEST(EmbeddingBagBatchTest, ExecutesMixedRequestsOnOnePool) {
  std::vector<float> table2_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> table3_data = {2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f};
  ASSERT_OK_AND_ASSIGN(auto table2, MakeTable(table2_data, 2));
  ASSERT_OK_AND_ASSIGN(auto table3,
                       MakeTable(table3_data, 3,
                                 EmbeddingTable::CombineMode::kSqrtN));
  const std::vector<int64_t> ids0 = {0, 1, 1};
  const std::vector<int32_t> rows0 = {0, 2, 3};
  const std::vector<int64_t> ids1 = {0, 1};
  const std::vector<int32_t> rows1 = {0, 2};
  std::vector<float> output0(4, -1.0f);
  std::vector<float> output1(3, -1.0f);
  auto allocate0 = MakeAllocator(&output0);
  auto allocate1 = MakeAllocator(&output1);
  std::vector<internal::EmbeddingBagRequest> requests = {
      MakeRequest(ids0, rows0, table2.get(), {}, &allocate0),
      MakeRequest(ids1, rows1, table3.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
      absl::MakeSpan(requests), internal::EmbeddingBagBackend::kPortable,
      &pool));

  EXPECT_THAT(output0, ElementsAre(4.0f, 6.0f, 3.0f, 4.0f));
  const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
  EXPECT_THAT(output1,
              Pointwise(Approx(1e-6f),
                        {3.0f * inv_sqrt_2, 7.0f * inv_sqrt_2,
                         11.0f * inv_sqrt_2}));
}

// Covers whole-group validation before allocation.
TEST(EmbeddingBagBatchTest, ValidatesWholeGroupBeforeAllocating) {
  std::vector<float> table_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, 2));
  const std::vector<int64_t> valid_ids = {0};
  const std::vector<int64_t> invalid_ids = {2};
  const std::vector<int32_t> rows = {0, 1};
  std::vector<float> output0(2, 17.0f);
  std::vector<float> output1(2, 19.0f);
  int allocation_count = 0;
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate0 =
      [&](size_t size) {
        ++allocation_count;
        return absl::MakeSpan(output0).subspan(0, size);
      };
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate1 =
      [&](size_t size) {
        ++allocation_count;
        return absl::MakeSpan(output1).subspan(0, size);
      };
  std::vector<internal::EmbeddingBagRequest> requests = {
      MakeRequest(valid_ids, rows, table.get(), {}, &allocate0),
      MakeRequest(invalid_ids, rows, table.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(requests),
                  internal::EmbeddingBagBackend::kPortable, &pool),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("request 1")));
  EXPECT_EQ(allocation_count, 0);
  EXPECT_THAT(output0, ElementsAre(17.0f, 17.0f));
  EXPECT_THAT(output1, ElementsAre(19.0f, 19.0f));
}

// Covers rejection of aliased request outputs before writes.
TEST(EmbeddingBagBatchTest, RejectsAliasedOutputsBeforeWriting) {
  std::vector<float> table_data = {1.0f, 2.0f};
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, 2));
  const std::vector<int64_t> ids = {0};
  const std::vector<int32_t> rows = {0, 1};
  std::vector<float> shared_output(2, 23.0f);
  auto allocate0 = MakeAllocator(&shared_output);
  auto allocate1 = MakeAllocator(&shared_output);
  std::vector<internal::EmbeddingBagRequest> requests = {
      MakeRequest(ids, rows, table.get(), {}, &allocate0),
      MakeRequest(ids, rows, table.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(requests),
                  internal::EmbeddingBagBackend::kPortable, &pool),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("overlap")));
  EXPECT_THAT(shared_output, ElementsAre(23.0f, 23.0f));
}

// Covers rejection of output-weight aliasing before writes.
TEST(EmbeddingBagBatchTest, RejectsOutputInputAliasingBeforeWriting) {
  std::vector<float> table_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, 2));
  const std::vector<int64_t> ids = {0, 1};
  const std::vector<int32_t> rows = {0, 2};
  std::vector<float> weights = {0.25f, 0.75f};
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate =
      [&](size_t) { return absl::MakeSpan(weights); };
  auto request = MakeRequest(ids, rows, table.get(), weights, &allocate);
  Eigen::ThreadPool pool(2);

  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(&request, 1),
                  internal::EmbeddingBagBackend::kPortable, &pool),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("weight storage")));
  EXPECT_THAT(weights, ElementsAre(0.25f, 0.75f));
}

// Covers pool re-entry without deadlock.
TEST(EmbeddingBagBatchTest, PoolReentryExecutesInline) {
  std::vector<float> table_data = {1.0f, 2.0f};
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, 2));
  const std::vector<int64_t> ids = {0};
  const std::vector<int32_t> rows = {0, 1};
  std::vector<float> output(2, -1.0f);
  auto allocate = MakeAllocator(&output);
  auto request = MakeRequest(ids, rows, table.get(), {}, &allocate);
  Eigen::ThreadPool pool(2);
  std::promise<void> blocker_started;
  std::future<void> blocker_ready = blocker_started.get_future();
  std::promise<void> release_blocker;
  std::shared_future<void> blocker_release =
      release_blocker.get_future().share();
  std::promise<absl::Status> completion;
  std::future<absl::Status> result = completion.get_future();

  pool.Schedule([&] {
    blocker_started.set_value();
    blocker_release.wait();
  });
  if (blocker_ready.wait_for(std::chrono::seconds(5)) !=
      std::future_status::ready) {
    release_blocker.set_value();
    FAIL() << "Thread-pool blocker did not start";
  }
  pool.Schedule([&] {
    completion.set_value(internal::ExecuteEmbeddingBagBatch(
        absl::MakeSpan(&request, 1),
        internal::EmbeddingBagBackend::kPortable, &pool));
  });

  const bool completed_while_other_worker_blocked =
      result.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  release_blocker.set_value();

  EXPECT_TRUE(completed_while_other_worker_blocked);
  ASSERT_EQ(result.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);
  ASSERT_OK(result.get());
  EXPECT_THAT(output, ElementsAre(1.0f, 2.0f));
}

}  // namespace
}  // namespace xla_serving::data::cpu_embedding

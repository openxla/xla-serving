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
      "table",
      Array(data,
            Shape({static_cast<int64_t>(data.size()) / dimension, dimension})),
      mode);
}

absl::AnyInvocable<absl::Span<float>(size_t)> MakeAllocator(
    std::vector<float>* output) {
  return [output](size_t size) {
    return absl::MakeSpan(*output).subspan(0, size);
  };
}

internal::EmbeddingBagInput MakeRequest(
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
  ASSERT_OK_AND_ASSIGN(
      auto table, MakeTable(table_data, 3, EmbeddingTable::CombineMode::kMean));
  const std::vector<int64_t> ids = {0, 2, 1};
  const std::vector<int32_t> rows = {0, 2, 2, 3};
  const std::vector<float> weights = {0.5f, 1.5f, 2.0f};
  std::vector<float> output(9, -99.0f);
  auto alloc = MakeAllocator(&output);
  internal::EmbeddingBagBackend selected = internal::EmbeddingBagBackend::kAuto;

  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), weights, &alloc),
      internal::EmbeddingBagBackend::kAuto, &selected));

  EXPECT_EQ(selected, internal::EmbeddingBagBackend::kPortable);
  EXPECT_THAT(output, ElementsAre(5.5f, 6.5f, 7.5f, 0.0f, 0.0f, 0.0f, 4.0f,
                                  5.0f, 6.0f));
}

// Covers mixed dimensions and combiners on one worker pool.
TEST(EmbeddingBagBatchTest, ExecutesMixedRequestsOnOnePool) {
  std::vector<float> table2_data = {1.0f, 2.0f, 3.0f, 4.0f};
  std::vector<float> table3_data = {2.0f, 4.0f, 6.0f, 1.0f, 3.0f, 5.0f};
  ASSERT_OK_AND_ASSIGN(auto table2, MakeTable(table2_data, 2));
  ASSERT_OK_AND_ASSIGN(
      auto table3,
      MakeTable(table3_data, 3, EmbeddingTable::CombineMode::kSqrtN));
  const std::vector<int64_t> ids0 = {0, 1, 1};
  const std::vector<int32_t> rows0 = {0, 2, 3};
  const std::vector<int64_t> ids1 = {0, 1};
  const std::vector<int32_t> rows1 = {0, 2};
  std::vector<float> output0(4, -1.0f);
  std::vector<float> output1(3, -1.0f);
  auto allocate0 = MakeAllocator(&output0);
  auto allocate1 = MakeAllocator(&output1);
  std::vector<internal::EmbeddingBagInput> inputs = {
      MakeRequest(ids0, rows0, table2.get(), {}, &allocate0),
      MakeRequest(ids1, rows1, table3.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
      absl::MakeSpan(inputs), internal::EmbeddingBagBackend::kPortable, &pool,
      /*inline_work_threshold=*/0));

  EXPECT_THAT(output0, ElementsAre(4.0f, 6.0f, 3.0f, 4.0f));
  const float inv_sqrt_2 = 1.0f / std::sqrt(2.0f);
  EXPECT_THAT(output1,
              Pointwise(Approx(1e-6f), {3.0f * inv_sqrt_2, 7.0f * inv_sqrt_2,
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
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate0 = [&](size_t size) {
    ++allocation_count;
    return absl::MakeSpan(output0).subspan(0, size);
  };
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate1 = [&](size_t size) {
    ++allocation_count;
    return absl::MakeSpan(output1).subspan(0, size);
  };
  std::vector<internal::EmbeddingBagInput> inputs = {
      MakeRequest(valid_ids, rows, table.get(), {}, &allocate0),
      MakeRequest(invalid_ids, rows, table.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(inputs),
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
  std::vector<internal::EmbeddingBagInput> inputs = {
      MakeRequest(ids, rows, table.get(), {}, &allocate0),
      MakeRequest(ids, rows, table.get(), {}, &allocate1),
  };
  Eigen::ThreadPool pool(2);

  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(inputs),
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
  absl::AnyInvocable<absl::Span<float>(size_t)> allocate = [&](size_t) {
    return absl::MakeSpan(weights);
  };
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
  std::promise<void> blocker_started;
  std::future<void> blocker_ready = blocker_started.get_future();
  std::promise<void> release_blocker;
  std::shared_future<void> blocker_release =
      release_blocker.get_future().share();
  std::promise<absl::Status> completion;
  std::future<absl::Status> result = completion.get_future();
  Eigen::ThreadPool pool(2);

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
        absl::MakeSpan(&request, 1), internal::EmbeddingBagBackend::kPortable,
        &pool, /*inline_work_threshold=*/0));
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

// Covers small groups running on the caller thread while every pool worker is
// busy.
TEST(EmbeddingBagBatchTest, SmallGroupExecutesInlineOnCaller) {
  std::vector<float> table_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(std::unique_ptr<EmbeddingTable> table,
                       MakeTable(table_data, 2));
  const std::vector<int64_t> ids = {0, 1};
  const std::vector<int32_t> rows = {0, 1, 2};
  std::vector<float> output(4, -1.0f);
  auto allocate = MakeAllocator(&output);
  auto request = MakeRequest(ids, rows, table.get(), {}, &allocate);
  constexpr int kThreads = 2;
  std::promise<void> release_workers;
  std::shared_future<void> workers_release =
      release_workers.get_future().share();
  std::vector<std::promise<void>> worker_started(kThreads);
  Eigen::ThreadPool pool(kThreads);
  for (int worker = 0; worker < kThreads; ++worker) {
    pool.Schedule([&, worker] {
      worker_started[worker].set_value();
      workers_release.wait();
    });
  }
  for (auto& started : worker_started) {
    if (started.get_future().wait_for(std::chrono::seconds(5)) !=
        std::future_status::ready) {
      release_workers.set_value();
      FAIL() << "Thread-pool blockers did not start";
    }
  }

  std::future<absl::Status> result = std::async(std::launch::async, [&] {
    return internal::ExecuteEmbeddingBagBatch(
        absl::MakeSpan(&request, 1), internal::EmbeddingBagBackend::kPortable,
        &pool);
  });
  const bool completed_while_workers_blocked =
      result.wait_for(std::chrono::seconds(5)) == std::future_status::ready;
  release_workers.set_value();

  EXPECT_TRUE(completed_while_workers_blocked);
  ASSERT_OK(result.get());
  EXPECT_THAT(output, ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
}

// Covers auto gating and exact FP32 addition order across all ID blocks.
TEST(EmbeddingBagSveTest, AutoDispatchIsGuardedAndPreservesAdditionOrder) {
  constexpr int kRows = 4;
  constexpr int kEmbeddingDim = 128;
  const float row_values[kRows] = {1.0e20f, 1.25f, -1.0e20f, 0.33333334f};
  std::vector<float> table_data(kRows * kEmbeddingDim);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kEmbeddingDim; ++column) {
      table_data[row * kEmbeddingDim + column] =
          row_values[row] + static_cast<float>(column % 3) * 0.03125f;
    }
  }
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, kEmbeddingDim));
  std::vector<int64_t> ids;
  for (int index = 0; index < 37; ++index) ids.push_back(index % kRows);
  // 31 IDs exercise the 16/8/4/2/1 block decomposition.
  const std::vector<int32_t> rows = {0, 31, 37};
  std::vector<float> portable(2 * kEmbeddingDim, -1.0f);
  std::vector<float> automatic(2 * kEmbeddingDim, -2.0f);
  auto portable_alloc = MakeAllocator(&portable);
  auto automatic_alloc = MakeAllocator(&automatic);
  internal::EmbeddingBagBackend selected = internal::EmbeddingBagBackend::kAuto;

  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), {}, &portable_alloc),
      internal::EmbeddingBagBackend::kPortable));
  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), {}, &automatic_alloc),
      internal::EmbeddingBagBackend::kAuto, &selected));

  const internal::EmbeddingBagBackend expected =
      internal::IsEmbeddingBagSveAvailable() &&
              internal::EmbeddingBagSveVectorLengthBytes() ==
                  internal::kAutoSveVectorLengthBytes
          ? internal::EmbeddingBagBackend::kSve
          : internal::EmbeddingBagBackend::kPortable;
  EXPECT_EQ(selected, expected);
  EXPECT_EQ(automatic, portable);

  if (internal::IsEmbeddingBagSveAvailable()) {
    std::vector<float> forced(2 * kEmbeddingDim, -3.0f);
    auto forced_alloc = MakeAllocator(&forced);
    ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
        MakeRequest(ids, rows, table.get(), {}, &forced_alloc),
        internal::EmbeddingBagBackend::kSve));
    EXPECT_EQ(forced, portable);
  }
}

TEST(EmbeddingBagSveTest, ForcedUnweightedFastSlicesAndTailMatchPortable) {
  if (!internal::IsEmbeddingBagSveAvailable()) {
    GTEST_SKIP() << "SVE is unavailable";
  }

  const int embedding_dim = 8 * (internal::EmbeddingBagSveVectorLengthBytes() /
                                 static_cast<int>(sizeof(float))) +
                            1;
  constexpr int kRows = 4;
  const float row_values[kRows] = {1.0e20f, 1.25f, -1.0e20f, 0.33333334f};
  std::vector<float> table_data(kRows * embedding_dim);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < embedding_dim; ++column) {
      table_data[row * embedding_dim + column] =
          row_values[row] + static_cast<float>(column % 3) * 0.03125f;
    }
  }
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, embedding_dim));
  std::vector<int64_t> ids;
  for (int index = 0; index < 17; ++index) ids.push_back(index % kRows);
  const std::vector<int32_t> rows = {0, 17};
  std::vector<float> portable(embedding_dim, -1.0f);
  std::vector<float> forced(embedding_dim, -2.0f);
  auto portable_alloc = MakeAllocator(&portable);
  auto forced_alloc = MakeAllocator(&forced);

  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), {}, &portable_alloc),
      internal::EmbeddingBagBackend::kPortable));
  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), {}, &forced_alloc),
      internal::EmbeddingBagBackend::kSve));
  EXPECT_EQ(forced, portable);
}

// Covers forced SVE tails, weights, normalization, and unavailable hardware.
TEST(EmbeddingBagSveTest, ForcedBackendIsGuardedAndDimensionAgnostic) {
  constexpr int kEmbeddingDim = 7;
  std::vector<float> table_data(3 * kEmbeddingDim);
  for (size_t index = 0; index < table_data.size(); ++index) {
    table_data[index] = static_cast<float>(index + 1) * 0.125f;
  }
  ASSERT_OK_AND_ASSIGN(
      auto table,
      MakeTable(table_data, kEmbeddingDim, EmbeddingTable::CombineMode::kMean));
  const std::vector<int64_t> ids = {0, 2, 1};
  const std::vector<int32_t> rows = {0, 2, 2, 3};
  const std::vector<float> weights = {0.25f, 1.25f, 0.5f};
  std::vector<float> portable(3 * kEmbeddingDim, -1.0f);
  std::vector<float> forced(3 * kEmbeddingDim, 41.0f);
  auto portable_alloc = MakeAllocator(&portable);
  auto forced_alloc = MakeAllocator(&forced);
  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), weights, &portable_alloc),
      internal::EmbeddingBagBackend::kPortable));

  const absl::Status status = internal::ComputeEmbeddingActivationsF32(
      MakeRequest(ids, rows, table.get(), weights, &forced_alloc),
      internal::EmbeddingBagBackend::kSve);
  if (internal::IsEmbeddingBagSveAvailable()) {
    ASSERT_OK(status);
    EXPECT_THAT(forced, Pointwise(Approx(1e-6f), portable));
  } else {
    EXPECT_THAT(status,
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
    EXPECT_THAT(forced, testing::Each(testing::FloatEq(41.0f)));
  }
}

// Covers absolute CSR offsets for threaded work starting mid-batch.
TEST(EmbeddingBagSveTest, ForcedBatchCoversThreadedNonzeroBagRanges) {
  constexpr int kRows = 4;
  constexpr int kEmbeddingDim = 128;
  constexpr int kBags = 16;
  std::vector<float> table_data(kRows * kEmbeddingDim);
  for (int row = 0; row < kRows; ++row) {
    for (int column = 0; column < kEmbeddingDim; ++column) {
      table_data[row * kEmbeddingDim + column] =
          static_cast<float>((row + 1) * 128 + column);
    }
  }
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, kEmbeddingDim));
  std::vector<int64_t> ids;
  std::vector<int32_t> rows;
  for (int bag = 0; bag < kBags; ++bag) {
    rows.push_back(bag);
    ids.push_back(bag % kRows);
  }
  rows.push_back(kBags);
  std::vector<float> portable(kBags * kEmbeddingDim, -1.0f);
  std::vector<float> forced(kBags * kEmbeddingDim, -3.0f);
  auto allocate_portable = MakeAllocator(&portable);
  auto allocate_forced = MakeAllocator(&forced);
  auto portable_request =
      MakeRequest(ids, rows, table.get(), {}, &allocate_portable);
  auto forced_request =
      MakeRequest(ids, rows, table.get(), {}, &allocate_forced);
  Eigen::ThreadPool pool(2);

  ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
      absl::MakeSpan(&portable_request, 1),
      internal::EmbeddingBagBackend::kPortable, &pool,
      /*inline_work_threshold=*/0));
  const absl::Status status = internal::ExecuteEmbeddingBagBatch(
      absl::MakeSpan(&forced_request, 1), internal::EmbeddingBagBackend::kSve,
      &pool, /*inline_work_threshold=*/0);

  if (internal::IsEmbeddingBagSveAvailable()) {
    ASSERT_OK(status);
    EXPECT_EQ(forced, portable);
  } else {
    EXPECT_THAT(status,
                absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition));
    EXPECT_THAT(forced, testing::Each(testing::FloatEq(-3.0f)));
  }
}

TEST(EmbeddingBagBatchTest, Int32IdsMatchInt64AcrossAllUnrollSizesAndModes) {
  for (int rows : {16, 1500}) {
    for (int embedding_dim : {7, 32, 64, 128}) {
      std::vector<float> table_data(rows * embedding_dim);
      for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < embedding_dim; ++col) {
          table_data[row * embedding_dim + col] =
              static_cast<float>(((row % 31) + 1) * 0.25f + col * 0.03125f);
        }
      }
      const std::vector<int> bag_sizes = {
          0, 1, 2, 3, 4, 7, 8, 13, 16, 31, 100, 1, 1, 1, 1};
      std::vector<int32_t> row_pointers = {0};
      std::vector<int64_t> ids_i64;
      std::vector<int32_t> ids_i32;
      for (int bag_size : bag_sizes) {
        for (int i = 0; i < bag_size; ++i) {
          const int id = (i * 7 + bag_size * 13) % rows;
          ids_i64.push_back(id);
          ids_i32.push_back(id);
        }
        row_pointers.push_back(static_cast<int32_t>(ids_i64.size()));
      }
      const size_t num_bags = bag_sizes.size();

      for (auto mode : {EmbeddingTable::CombineMode::kSum,
                        EmbeddingTable::CombineMode::kMean,
                        EmbeddingTable::CombineMode::kSqrtN}) {
        ASSERT_OK_AND_ASSIGN(auto table,
                             MakeTable(table_data, embedding_dim, mode));
        std::vector<float> out_i64(num_bags * embedding_dim, -1.0f);
        std::vector<float> out_i32(num_bags * embedding_dim, -2.0f);
        auto alloc_i64 = MakeAllocator(&out_i64);
        auto alloc_i32 = MakeAllocator(&out_i32);

        internal::EmbeddingBagInput req_i64 =
            MakeRequest(ids_i64, row_pointers, table.get(), {}, &alloc_i64);
        internal::EmbeddingBagInput req_i32 = {
            .embedding_ids = {},
            .row_pointers = row_pointers,
            .embedding_table = table.get(),
            .weights = {},
            .output_allocator = &alloc_i32,
            .embedding_ids_i32 = ids_i32,
        };
        Eigen::ThreadPool pool(4);
        ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
            absl::MakeSpan(&req_i64, 1),
            internal::EmbeddingBagBackend::kPortable, &pool,
            /*inline_work_threshold=*/0));
        ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
            absl::MakeSpan(&req_i32, 1),
            internal::EmbeddingBagBackend::kPortable, &pool,
            /*inline_work_threshold=*/0));
        EXPECT_EQ(out_i32, out_i64);
      }
    }
  }
}

TEST(EmbeddingBagBatchTest, Int32IdsValidateBoundsAndIsolation) {
  std::vector<float> table_data = {1.0f, 2.0f, 3.0f, 4.0f};
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, 2));
  const std::vector<int32_t> oob_ids = {0, -1};
  const std::vector<int32_t> rows = {0, 2};
  std::vector<float> output(2, 5.0f);
  auto allocate = MakeAllocator(&output);
  internal::EmbeddingBagInput bad_req = {
      .embedding_ids = {},
      .row_pointers = rows,
      .embedding_table = table.get(),
      .weights = {},
      .output_allocator = &allocate,
      .embedding_ids_i32 = oob_ids,
  };
  Eigen::ThreadPool pool(2);
  EXPECT_THAT(internal::ExecuteEmbeddingBagBatch(
                  absl::MakeSpan(&bad_req, 1),
                  internal::EmbeddingBagBackend::kPortable, &pool),
              absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                                     testing::HasSubstr("out of bounds")));

  const std::vector<int64_t> valid_i64 = {0, 1};
  const std::vector<int32_t> valid_i32 = {0, 1};
  internal::EmbeddingBagInput both_req = {
      .embedding_ids = valid_i64,
      .row_pointers = rows,
      .embedding_table = table.get(),
      .weights = {},
      .output_allocator = &allocate,
      .embedding_ids_i32 = valid_i32,
  };
  EXPECT_THAT(
      internal::ExecuteEmbeddingBagBatch(
          absl::MakeSpan(&both_req, 1),
          internal::EmbeddingBagBackend::kPortable, &pool),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument,
                             testing::HasSubstr("both int64 and int32")));
}

TEST(EmbeddingBagBatchTest, ConcurrentCallersSharePoolWithoutRaceOrDeadlock) {
  constexpr int kRows = 1024;
  constexpr int kDim = 128;
  constexpr int kBags = 64;
  std::vector<float> table_data(kRows * kDim);
  for (int i = 0; i < kRows * kDim; ++i) {
    table_data[i] = static_cast<float>((i % 29) * 0.125f);
  }
  ASSERT_OK_AND_ASSIGN(auto table, MakeTable(table_data, kDim));

  std::vector<int32_t> row_pointers(kBags + 1);
  std::vector<int32_t> ids(kBags * 4);
  for (int b = 0; b <= kBags; ++b) row_pointers[b] = b * 4;
  for (int i = 0; i < kBags * 4; ++i) ids[i] = (i * 17) % kRows;

  std::vector<float> expected(kBags * kDim, 0.0f);
  auto expected_alloc = MakeAllocator(&expected);
  internal::EmbeddingBagInput ref_req = {
      .embedding_ids = {},
      .row_pointers = row_pointers,
      .embedding_table = table.get(),
      .weights = {},
      .output_allocator = &expected_alloc,
      .embedding_ids_i32 = ids,
  };
  ASSERT_OK(internal::ComputeEmbeddingActivationsF32(
      ref_req, internal::EmbeddingBagBackend::kPortable));

  Eigen::ThreadPool pool(4);
  constexpr int kCallers = 12;
  constexpr int kIterations = 25;
  std::vector<std::future<void>> futures;
  futures.reserve(kCallers);
  for (int c = 0; c < kCallers; ++c) {
    futures.push_back(std::async(std::launch::async, [&]() {
      for (int iter = 0; iter < kIterations; ++iter) {
        std::vector<float> out(kBags * kDim, -1.0f);
        auto alloc = MakeAllocator(&out);
        internal::EmbeddingBagInput req = {
            .embedding_ids = {},
            .row_pointers = row_pointers,
            .embedding_table = table.get(),
            .weights = {},
            .output_allocator = &alloc,
            .embedding_ids_i32 = ids,
        };
        ASSERT_OK(internal::ExecuteEmbeddingBagBatch(
            absl::MakeSpan(&req, 1), internal::EmbeddingBagBackend::kAuto,
            &pool, /*inline_work_threshold=*/0));
        EXPECT_EQ(out, expected);
      }
    }));
  }
  for (auto& f : futures) {
    f.get();
  }
}

}  // namespace
}  // namespace xla_serving::data::cpu_embedding

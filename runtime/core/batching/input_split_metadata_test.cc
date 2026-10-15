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

#include "runtime/core/batching/input_split_metadata.h"  // from @xla_serving

#include <tuple>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/container/fixed_array.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp

namespace xla_serving {
namespace internal {
namespace {

using ::testing::ElementsAreArray;

TEST(InputSplitUtilTest, Basic) {
  for (const auto& batch_task_param :
       {std::tuple<int /* input_size */, int /* open_batch_remaining_slot */,
                   int /* batch_size_limit */, int /* expected_num_batches */,
                   int /* expected_num_new_batches */,
                   int /* expected_head_batch_task_size */,
                   int /* expected_tail_batch_task_size */>{5, 1, 1, 5, 4, 1,
                                                            1},
        {10, 3, 4, 3, 2, 3, 3},
        {20, 5, 6, 4, 3, 5, 3},
        {30, 0, 11, 3, 3, 11, 8},
        {5, 6, 8, 1, 0, 5, 5}}) {
    const int input_size = std::get<0>(batch_task_param);
    const int open_batch_remaining_slot = std::get<1>(batch_task_param);
    const int batch_size_limit = std::get<2>(batch_task_param);
    const int expected_num_batches = std::get<3>(batch_task_param);
    const int expected_head_batch_task_size = std::get<5>(batch_task_param);
    const int expected_tail_batch_task_size = std::get<6>(batch_task_param);

    // The number of remaining slots should be smaller than or equal to
    // batch_size_limit; whearas we allow one input (of `input_size`) to span
    // over multiple batches.
    ASSERT_LE(open_batch_remaining_slot, batch_size_limit);

    InputSplitMetadata input_split_metadata(
        input_size, open_batch_remaining_slot, batch_size_limit);
    EXPECT_EQ(input_split_metadata.task_sizes().size(), expected_num_batches);

    absl::FixedArray<int> expected_task_sizes(expected_num_batches);
    for (int i = 0; i < expected_num_batches; i++) {
      if (i == 0) {
        expected_task_sizes[i] = expected_head_batch_task_size;
      } else if (i == expected_num_batches - 1) {
        expected_task_sizes[i] = expected_tail_batch_task_size;
      } else {
        expected_task_sizes[i] = batch_size_limit;
      }
    }

    EXPECT_THAT(input_split_metadata.task_sizes(),
                ElementsAreArray(expected_task_sizes));
    EXPECT_EQ(input_split_metadata.DebugString(),
              absl::StrJoin(expected_task_sizes, ", "));
  }
}
}  // namespace
}  // namespace internal
}  // namespace xla_serving

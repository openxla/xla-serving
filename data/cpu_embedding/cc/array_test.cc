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

#include "data/cpu_embedding/cc/array.h"  // from @xla_serving

#include <any>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"

namespace xla_serving::data::cpu_embedding {
namespace {

TEST(ArrayTest, ConstructFromSpan) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f};
  Shape shape({2, 2});
  Array arr(absl::MakeConstSpan(data), shape);

  EXPECT_EQ(arr.GetDType(), DType::kFloat32);
  EXPECT_EQ(arr.GetShape().dims.size(), 2);
  EXPECT_EQ(arr.GetShape().dims[0], 2);
  EXPECT_EQ(arr.GetShape().dims[1], 2);

  auto span = arr.AsSpan<float>();
  EXPECT_EQ(span.size(), 4);
  EXPECT_EQ(span[0], 1.0f);
  EXPECT_EQ(span[3], 4.0f);
}

TEST(ArrayTest, WrapWithAny) {
  std::vector<int32_t> data = {10, 20, 30};
  auto data_ptr = data.data();
  Shape shape({3});

  // Wrap unowned
  Array arr = Array::Wrap(data_ptr, shape, std::any(data));

  EXPECT_EQ(arr.GetDType(), DType::kInt32);
  EXPECT_EQ(arr.Data(), data_ptr);
  EXPECT_EQ(arr.AsSpan<int32_t>()[1], 20);
}

TEST(ArrayTest, AsTensor) {
  std::vector<float> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  Shape shape({2, 3});
  Array arr(absl::MakeConstSpan(data), shape);

  auto tensor = arr.AsTensor<float, 2>();
  EXPECT_EQ(tensor.dimension(0), 2);
  EXPECT_EQ(tensor.dimension(1), 3);
  EXPECT_EQ(tensor(0, 0), 1.0f);
  EXPECT_EQ(tensor(1, 2), 6.0f);
}

TEST(ArrayTest, ReleaseBufferOwned) {
  std::vector<float> data = {1.1f, 2.2f};
  Shape shape({2});
  Array arr(absl::MakeConstSpan(data), shape);

  std::string released = std::move(arr).ReleaseBuffer();
  EXPECT_EQ(released.size(), 2 * sizeof(float));
  const float* released_ptr = reinterpret_cast<const float*>(released.data());
  EXPECT_EQ(released_ptr[0], 1.1f);
  EXPECT_EQ(released_ptr[1], 2.2f);
}

TEST(ArrayTest, ReleaseBufferShared) {
  std::vector<float> data = {1.1f, 2.2f};
  Shape shape({2});
  Array arr(absl::MakeConstSpan(data), shape);

  // Create another reference to the owner to trigger copy in ReleaseBuffer
  Array arr2 = arr;

  std::string released = std::move(arr).ReleaseBuffer();
  EXPECT_EQ(released.size(), 2 * sizeof(float));

  // Verify data is still there in arr2
  EXPECT_EQ(arr2.AsSpan<float>()[0], 1.1f);
}

}  // namespace
}  // namespace xla_serving::data::cpu_embedding

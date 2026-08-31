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

#include "data/cpu_embedding/cc/compiled_custom_combiner.h"  // from @neptune

#include <cmath>
#include <cstdint>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

using ::testing::Pointwise;

MATCHER_P(Approx, epsilon, "") {
  return std::abs(std::get<0>(arg) - std::get<1>(arg)) < epsilon;
}

TEST(CompiledCustomCombinerTest, FunctionCustomCombinerExecution) {
  CompiledCustomCombiner::Options options;
  options.tile_size = 2;
  options.max_valency = 2;
  options.embedding_dim = 3;
  options.output_dim = 3;

  auto sum_fn = [](absl::Span<const int32_t> valencies,
                   absl::Span<const float> vectors,
                   absl::Span<const float> weights,
                   absl::Span<float> output) -> absl::Status {
    for (int i = 0; i < 2; ++i) {
      for (int d = 0; d < 3; ++d) {
        float sum = 0.0f;
        for (int v = 0; v < valencies[i]; ++v) {
          sum += vectors[i * 6 + v * 3 + d];
        }
        output[i * 3 + d] = sum;
      }
    }
    return absl::OkStatus();
  };

  ASSERT_OK_AND_ASSIGN(
      auto combiner,
      CompiledCustomCombiner::CreateFromFunction(sum_fn, options));

  std::vector<int32_t> valencies = {2, 1};
  std::vector<float> vectors = {
      // Sample 0, vector 0
      1.0f, 2.0f, 3.0f,
      // Sample 0, vector 1
      4.0f, 5.0f, 6.0f,
      // Sample 1, vector 0
      10.0f, 20.0f, 30.0f,
      // Sample 1, vector 1 (padding)
      0.0f, 0.0f, 0.0f,
  };
  std::vector<float> weights;
  std::vector<float> output(6, 0.0f);

  ASSERT_OK(combiner->ExecuteTile(absl::MakeSpan(valencies),
                                 absl::MakeSpan(vectors),
                                 absl::MakeSpan(weights),
                                 absl::MakeSpan(output)));

  EXPECT_THAT(output, Pointwise(Approx(1e-6f),
                               {5.0f, 7.0f, 9.0f, 10.0f, 20.0f, 30.0f}));
}

TEST(CompiledCustomCombinerTest, XlaCustomCombinerStableHloExecution) {
  // StableHLO MLIR module that reduces (sums) along dimension 1 (max_valency):
  // (valencies: tensor<2xi32>, vectors: tensor<2x2x2xf32>) -> tensor<2x2xf32>
  constexpr char kStableHloMlir[] = R"mlir(
module @custom_combiner {
  func.func @main(%valencies: tensor<2xi32>, %vectors: tensor<2x2x2xf32>) -> tensor<2x2xf32> {
    %c0 = stablehlo.constant dense<0.000000e+00> : tensor<f32>
    %sum = stablehlo.reduce(%vectors init: %c0) across dimensions = [1] : (tensor<2x2x2xf32>, tensor<f32>) -> tensor<2x2xf32>
      reducer(%arg0: tensor<f32>, %arg1: tensor<f32>) {
        %res = stablehlo.add %arg0, %arg1 : tensor<f32>
        stablehlo.return %res : tensor<f32>
      }
    return %sum : tensor<2x2xf32>
  }
}
)mlir";

  CompiledCustomCombiner::Options options;
  options.tile_size = 2;
  options.max_valency = 2;
  options.embedding_dim = 2;
  options.output_dim = 2;

  ASSERT_OK_AND_ASSIGN(
      auto combiner,
      CompiledCustomCombiner::CompileFromStableHlo(
          kStableHloMlir,
          composite::transforms::cpu_embedding::CustomCombinerSpec::
              STABLEHLO_TEXT,
          options));

  std::vector<int32_t> valencies = {2, 1};
  std::vector<float> vectors = {
      // Sample 0: [1, 2] + [3, 4] = [4, 6]
      1.0f, 2.0f,
      3.0f, 4.0f,
      // Sample 1: [5, 6] + [0, 0] = [5, 6]
      5.0f, 6.0f,
      0.0f, 0.0f,
  };
  std::vector<float> weights;
  std::vector<float> output(4, 0.0f);

  ASSERT_OK(combiner->ExecuteTile(absl::MakeSpan(valencies),
                                 absl::MakeSpan(vectors),
                                 absl::MakeSpan(weights),
                                 absl::MakeSpan(output)));

  EXPECT_THAT(output, Pointwise(Approx(1e-6f), {4.0f, 6.0f, 5.0f, 6.0f}));
}

}  // namespace
}  // namespace neptune::data::cpu_embedding

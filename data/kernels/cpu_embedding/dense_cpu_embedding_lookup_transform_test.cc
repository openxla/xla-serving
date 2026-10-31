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

#include "data/kernels/cpu_embedding/dense_cpu_embedding_lookup_transform.h"  // from @xla_serving

#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/test_utils/shape.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace {

class TestCpuEmbeddingEngineResource : public CpuEmbeddingEngineResource {
 public:
  using CpuEmbeddingEngineResource::CpuEmbeddingEngineResource;

  void RegisterEmbeddingDim(absl::string_view feature_name,
                            int32_t embedding_dim) {
    feature_to_embedding_dim_[std::string(feature_name)] = embedding_dim;
  }
};

TEST(DenseCpuEmbeddingLookupTransformTest, BasicDenseLookup) {
  // 1. Setup Resource and Engine
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  // Create 5x4 table:
  std::vector<float> table_data = {
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
      2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f,
  };
  cpu_embedding::Shape table_shape({5, 4});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table,
                       cpu_embedding::EmbeddingTable::Create(
                           "table1", std::move(table_array),
                           cpu_embedding::EmbeddingTable::CombineMode::kSum));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feat1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feat1", 4);

  // 2. Setup Transform with unified pad_value=0
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      DenseCpuEmbeddingLookupTransform::CreateForFeature(resource, "feat1",
                                                         /*pad_value=*/0));

  // 3. Prepare Input Tensor: shape [2, 4]
  std::vector<int32_t> ids_vec = {
      1, 2, 3, 0,  // Row 0: 1 + 2 + 3 + 0 = 6.0
      4, 1, 2, 0,  // Row 1: 4 + 1 + 2 + 0 = 7.0
  };
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2, 4}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  // 4. Execute Transform
  std::vector<xla_serving::HostBuffer> inputs;
  inputs.push_back(std::move(ids_hb));
  ASSERT_OK_AND_ASSIGN(auto outputs, (*transform)(std::move(inputs)));
  ASSERT_EQ(outputs.size(), 1);
  const auto& out = outputs[0];

  EXPECT_EQ(out.shape().dims()[0], 2);
  EXPECT_EQ(out.shape().dims()[1], 4);

  const float* out_ptr = reinterpret_cast<const float*>(out.base());
  for (int i = 0; i < 4; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], 6.0f);
  }
  for (int i = 4; i < 8; ++i) {
    EXPECT_FLOAT_EQ(out_ptr[i], 7.0f);
  }
}

TEST(DenseCpuEmbeddingLookupTransformTest, MultiTableParallelLookup) {
  // 1. Setup Resource with 2 tables
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  // Table 1 (Dim 4)
  std::vector<float> table1_data = {
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f, 2.0f, 2.0f,
  };
  ASSERT_OK_AND_ASSIGN(
      auto table1, cpu_embedding::EmbeddingTable::Create(
                       "table1",
                       cpu_embedding::Array(absl::MakeConstSpan(table1_data),
                                            cpu_embedding::Shape({3, 4})),
                       cpu_embedding::EmbeddingTable::CombineMode::kSum));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table1)));

  // Table 2 (Dim 2)
  std::vector<float> table2_data = {
      0.0f, 0.0f, 10.0f, 10.0f, 20.0f, 20.0f,
  };
  ASSERT_OK_AND_ASSIGN(
      auto table2, cpu_embedding::EmbeddingTable::Create(
                       "table2",
                       cpu_embedding::Array(absl::MakeConstSpan(table2_data),
                                            cpu_embedding::Shape({3, 2})),
                       cpu_embedding::EmbeddingTable::CombineMode::kSum));
  EXPECT_OK(engine->AddLocalTable("table2", std::move(table2)));

  cpu_embedding::FeatureSpec spec1;
  spec1.name = "feat1";
  spec1.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec1));
  resource->RegisterEmbeddingDim("feat1", 4);

  cpu_embedding::FeatureSpec spec2;
  spec2.name = "feat2";
  spec2.table_name = "table2";
  EXPECT_OK(engine->RegisterFeatureSpec(spec2));
  resource->RegisterEmbeddingDim("feat2", 2);

  // 2. Setup Multi-Table Transform
  ASSERT_OK_AND_ASSIGN(auto transform,
                       DenseCpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feat1", "feat2"},
                           /*pad_value=*/0));

  // 3. Prepare Inputs
  // Feat1: shape [2, 3] -> rows [1, 2, 0] (3.0) and [2, 2, 0] (4.0)
  std::vector<int32_t> feat1_ids = {1, 2, 0, 2, 2, 0};
  xla_serving::HostBuffer hb1(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                      xla::ifrt::Shape({2, 3}));
  absl::c_copy(feat1_ids, reinterpret_cast<int32_t*>(hb1.base()));

  // Feat2: shape [2, 2] -> rows [1, 0] (10.0) and [2, 1] (30.0)
  std::vector<int32_t> feat2_ids = {1, 0, 2, 1};
  xla_serving::HostBuffer hb2(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                      xla::ifrt::Shape({2, 2}));
  absl::c_copy(feat2_ids, reinterpret_cast<int32_t*>(hb2.base()));

  // 4. Execute Multi-Table in single call
  std::vector<xla_serving::HostBuffer> inputs;
  inputs.push_back(std::move(hb1));
  inputs.push_back(std::move(hb2));

  ASSERT_OK_AND_ASSIGN(auto outputs, (*transform)(std::move(inputs)));
  ASSERT_EQ(outputs.size(), 2);

  // Check Feat1 output: [2, 4]
  EXPECT_EQ(outputs[0].shape().dims()[0], 2);
  EXPECT_EQ(outputs[0].shape().dims()[1], 4);
  const float* out1 = reinterpret_cast<const float*>(outputs[0].base());
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out1[i], 3.0f);
  for (int i = 4; i < 8; ++i) EXPECT_FLOAT_EQ(out1[i], 4.0f);

  // Check Feat2 output: [2, 2]
  EXPECT_EQ(outputs[1].shape().dims()[0], 2);
  EXPECT_EQ(outputs[1].shape().dims()[1], 2);
  const float* out2 = reinterpret_cast<const float*>(outputs[1].base());
  EXPECT_FLOAT_EQ(out2[0], 10.0f);
  EXPECT_FLOAT_EQ(out2[1], 10.0f);
  EXPECT_FLOAT_EQ(out2[2], 30.0f);
  EXPECT_FLOAT_EQ(out2[3], 30.0f);
}

TEST(DenseCpuEmbeddingLookupTransformTest, SerializationAndDeserialization) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  resource->RegisterEmbeddingDim("feat1", 4);
  resource->RegisterEmbeddingDim("feat2", 8);

  ASSERT_OK_AND_ASSIGN(auto orig,
                       DenseCpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feat1", "feat2"},
                           /*pad_value=*/-1));

  ASSERT_OK_AND_ASSIGN(auto proto_config, orig->ToProto());
  EXPECT_EQ(proto_config.op_name(), "DenseCpuEmbeddingLookupTransform");

  ASSERT_OK_AND_ASSIGN(
      auto deserialized_base,
      TransformBase::FromProto(proto_config, /*resources=*/{resource}));
  auto* deserialized =
      dynamic_cast<DenseCpuEmbeddingLookupTransform*>(deserialized_base.get());
  ASSERT_NE(deserialized, nullptr);

  EXPECT_EQ(deserialized->feature_names(),
            (std::vector<std::string>{"feat1", "feat2"}));
  EXPECT_EQ(deserialized->pad_value(), -1);
}

TEST(DenseCpuEmbeddingLookupTransformTest, ShapeInference) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  resource->RegisterEmbeddingDim("feat1", 16);
  resource->RegisterEmbeddingDim("feat2", 32);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       DenseCpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feat1", "feat2"},
                           /*pad_value=*/0));

  api::ShapeInferenceContext context;
  const api::HostBufferSpec spec1{xla::ifrt::DType(xla::ifrt::DType::kS32),
                                  xla::ifrt::Shape({64, 4})};
  const api::HostBufferSpec spec2{xla::ifrt::DType(xla::ifrt::DType::kS32),
                                  xla::ifrt::Shape({64, 8})};

  std::vector<api::ShapeInferenceValue> inputs = {
      api::ShapeInferenceValue(spec1), api::ShapeInferenceValue(spec2)};
  context.inputs = absl::MakeConstSpan(inputs);

  ASSERT_OK(transform->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 2);
  EXPECT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape, xla::ifrt::Shape({64, 16}));
  EXPECT_TRUE(context.outputs[1].IsSpec());
  EXPECT_EQ(context.outputs[1].spec().shape, xla::ifrt::Shape({64, 32}));
}

TEST(DenseCpuEmbeddingLookupTransformTest, ShapeInferenceWithUnknownBatch) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  resource->RegisterEmbeddingDim("feat1", 16);
  resource->RegisterEmbeddingDim("feat2", 32);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       DenseCpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feat1", "feat2"},
                           /*pad_value=*/0));

  // An unknown batch dimension is forwarded. A scalar has no batch dimension,
  // so the output's is unknown.
  std::vector<api::ShapeInferenceValue> inputs = {
      api::ShapeInferenceValue(
          api::HostBufferSpec{xla::ifrt::DType(xla::ifrt::DType::kS32),
                              test_utils::PartialShapeOrDie(
                                  {api::Dim::Unknown(), api::Dim::Static(4)})}),
      api::ShapeInferenceValue(
          api::HostBufferSpec{xla::ifrt::DType(xla::ifrt::DType::kS32),
                              api::PartialShape(xla::ifrt::Shape({}))})};
  api::ShapeInferenceContext context;
  context.inputs = absl::MakeConstSpan(inputs);

  ASSERT_OK(transform->PropagateSignature(context));
  ASSERT_EQ(context.outputs.size(), 2);
  ASSERT_TRUE(context.outputs[0].IsSpec());
  EXPECT_EQ(context.outputs[0].spec().shape,
            test_utils::PartialShapeOrDie(
                {api::Dim::Unknown(), api::Dim::Static(16)}));
  ASSERT_TRUE(context.outputs[1].IsSpec());
  EXPECT_EQ(context.outputs[1].spec().shape,
            test_utils::PartialShapeOrDie(
                {api::Dim::Unknown(), api::Dim::Static(32)}));
}

TEST(DenseCpuEmbeddingLookupTransformTest, S64InputDenseLookup) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {
      0.0f, 0.0f, 0.0f, 0.0f, 1.0f, 1.0f, 1.0f, 1.0f, 2.0f, 2.0f,
      2.0f, 2.0f, 3.0f, 3.0f, 3.0f, 3.0f, 4.0f, 4.0f, 4.0f, 4.0f,
  };
  cpu_embedding::Shape table_shape({5, 4});
  ASSERT_OK_AND_ASSIGN(auto table,
                       cpu_embedding::EmbeddingTable::Create(
                           "table1",
                           cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                                table_shape),
                           cpu_embedding::EmbeddingTable::CombineMode::kSum));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feat1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feat1", 4);

  ASSERT_OK_AND_ASSIGN(
      auto transform,
      DenseCpuEmbeddingLookupTransform::CreateForFeature(resource, "feat1",
                                                         /*pad_value=*/0));

  std::vector<int64_t> ids_vec = {
      1, 2, 3, 0,
      4, 1, 2, 0,
  };
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                         xla::ifrt::Shape({2, 4}));
  absl::c_copy(ids_vec, reinterpret_cast<int64_t*>(ids_hb.base()));

  std::vector<xla_serving::HostBuffer> inputs;
  inputs.push_back(std::move(ids_hb));
  ASSERT_OK_AND_ASSIGN(auto outputs, (*transform)(std::move(inputs)));
  ASSERT_EQ(outputs.size(), 1);
  const auto& out = outputs[0];

  EXPECT_EQ(out.shape().dims()[0], 2);
  EXPECT_EQ(out.shape().dims()[1], 4);

  const float* out_ptr = reinterpret_cast<const float*>(out.base());
  for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(out_ptr[i], 6.0f);
  for (int i = 4; i < 8; ++i) EXPECT_FLOAT_EQ(out_ptr[i], 7.0f);
}

TEST(DenseCpuEmbeddingLookupTransformTest, AllPaddedSequenceFallback) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {
      0.0f, 0.0f, 1.0f, 1.0f,
  };
  ASSERT_OK_AND_ASSIGN(auto table,
                       cpu_embedding::EmbeddingTable::Create(
                           "table1",
                           cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                                cpu_embedding::Shape({2, 2})),
                           cpu_embedding::EmbeddingTable::CombineMode::kSum));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feat1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feat1", 2);

  ASSERT_OK_AND_ASSIGN(
      auto transform,
      DenseCpuEmbeddingLookupTransform::CreateForFeature(resource, "feat1",
                                                         /*pad_value=*/0));

  // Sequence of all padding tokens: [0, 0, 0]
  std::vector<int32_t> ids_vec = {0, 0, 0};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({1, 3}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<xla_serving::HostBuffer> inputs;
  inputs.push_back(std::move(ids_hb));
  ASSERT_OK_AND_ASSIGN(auto outputs, (*transform)(std::move(inputs)));
  ASSERT_EQ(outputs.size(), 1);
  const auto& out = outputs[0];

  EXPECT_EQ(out.shape().dims()[0], 1);
  EXPECT_EQ(out.shape().dims()[1], 2);
  const float* out_ptr = reinterpret_cast<const float*>(out.base());
  EXPECT_FLOAT_EQ(out_ptr[0], 0.0f);
  EXPECT_FLOAT_EQ(out_ptr[1], 0.0f);
}

TEST(DenseCpuEmbeddingLookupTransformTest, ErrorHandling) {
  // Null resource check
  EXPECT_THAT(
      DenseCpuEmbeddingLookupTransform::Create(nullptr, {"feat1"}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Empty feature names vector check
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  EXPECT_THAT(
      DenseCpuEmbeddingLookupTransform::Create(resource, {}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Empty feature name string check
  EXPECT_THAT(
      DenseCpuEmbeddingLookupTransform::Create(resource, {""}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  ASSERT_OK_AND_ASSIGN(
      auto transform,
      DenseCpuEmbeddingLookupTransform::CreateForFeature(resource, "feat1"));

  // Input count mismatch in operator()
  EXPECT_THAT(
      (*transform)({}),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Input count mismatch in Execute
  api::ExecuteContext exec_ctx;
  EXPECT_THAT(
      transform->Execute(exec_ctx),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Input count mismatch in PropagateSignature
  api::ShapeInferenceContext sig_ctx;
  EXPECT_THAT(
      transform->PropagateSignature(sig_ctx),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Non-HostBufferSpec input in PropagateSignature
  std::vector<api::ShapeInferenceValue> non_spec_inputs = {
      api::ShapeInferenceValue()};
  sig_ctx.inputs = absl::MakeConstSpan(non_spec_inputs);
  EXPECT_THAT(
      transform->PropagateSignature(sig_ctx),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Empty shape / rank-0 input check
  xla_serving::HostBuffer rank0_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                           xla::ifrt::Shape({}));
  std::vector<xla_serving::HostBuffer> rank0_inputs;
  rank0_inputs.push_back(std::move(rank0_hb));
  EXPECT_THAT(
      (*transform)(std::move(rank0_inputs)),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Invalid dtype (e.g. F32 IDs)
  xla_serving::HostBuffer f32_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                         xla::ifrt::Shape({2, 2}));
  std::vector<xla_serving::HostBuffer> invalid_inputs;
  invalid_inputs.push_back(std::move(f32_hb));
  EXPECT_THAT(
      (*transform)(std::move(invalid_inputs)),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));

  // Dimension overflow check (> INT32_MAX)
  int64_t huge_dim = static_cast<int64_t>(std::numeric_limits<int32_t>::max()) + 1;
  xla_serving::HostBuffer overflow_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                              xla::ifrt::Shape({huge_dim, 1}));
  std::vector<xla_serving::HostBuffer> overflow_inputs;
  overflow_inputs.push_back(std::move(overflow_hb));
  EXPECT_THAT(
      (*transform)(std::move(overflow_inputs)),
      absl_testing::StatusIs(absl::StatusCode::kInvalidArgument));
}

// Registers a 3x2 table and feature "feat1" on `resource` and returns a
// transform over it with pad_value=0.
std::unique_ptr<DenseCpuEmbeddingLookupTransform> CreateSingleFeatureTransform(
    const std::shared_ptr<TestCpuEmbeddingEngineResource>& resource,
    int32_t max_sequence_length,
    cpu_embedding::CustomCombinerFn custom_combiner) {
  auto* engine = resource->Get();
  std::vector<float> table_data = {0.0f, 0.0f, 1.0f, 1.0f, 2.0f, 2.0f};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({3, 2}));
  auto table = cpu_embedding::EmbeddingTable::Create(
      "table1", std::move(table_array),
      cpu_embedding::EmbeddingTable::CombineMode::kSum, custom_combiner);
  CHECK_OK(table.status());
  CHECK_OK(engine->AddLocalTable("table1", std::move(*table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feat1";
  spec.table_name = "table1";
  spec.max_sequence_length = max_sequence_length;
  CHECK_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feat1", 2);
  auto transform = DenseCpuEmbeddingLookupTransform::CreateForFeature(
      resource, "feat1", /*pad_value=*/0);
  CHECK_OK(transform.status());
  return std::move(*transform);
}

std::vector<xla_serving::HostBuffer> MakeIdsInput() {
  static constexpr int32_t kIds[] = {1, 2, 2, 0};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2, 2}));
  absl::c_copy(kIds, ids_hb.flat<int32_t>().data());
  std::vector<xla_serving::HostBuffer> inputs;
  inputs.push_back(std::move(ids_hb));
  return inputs;
}

TEST(DenseCpuEmbeddingLookupTransformTest, SequenceFeatureIsUnimplemented) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto transform = CreateSingleFeatureTransform(
      resource, /*max_sequence_length=*/2, /*custom_combiner=*/nullptr);
  EXPECT_THAT((*transform)(MakeIdsInput()),
              absl_testing::StatusIs(absl::StatusCode::kUnimplemented,
                                     testing::HasSubstr("non-sequence")));
}

void NoOpCombiner(const int64_t*, int64_t, const int32_t*, int64_t,
                  const float*, int64_t, const float*, int64_t, int64_t,
                  float*) {}

TEST(DenseCpuEmbeddingLookupTransformTest, CustomCombinerIsRejected) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto transform = CreateSingleFeatureTransform(
      resource, /*max_sequence_length=*/0, &NoOpCombiner);
  EXPECT_THAT((*transform)(MakeIdsInput()),
              absl_testing::StatusIs(absl::StatusCode::kFailedPrecondition,
                                     testing::HasSubstr("custom combiner")));
}

}  // namespace
}  // namespace composite
}  // namespace xla_serving::data


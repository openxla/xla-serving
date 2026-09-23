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

#include "data/kernels/cpu_embedding/cpu_embedding_lookup_transform.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/cpu_embedding_engine.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_spec.h"  // from @xla_serving
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace {

using ::testing::ElementsAre;
using ::testing::FloatEq;

class TestCpuEmbeddingEngineResource : public CpuEmbeddingEngineResource {
 public:
  using CpuEmbeddingEngineResource::CpuEmbeddingEngineResource;

  void RegisterEmbeddingDim(absl::string_view feature_name,
                            int32_t embedding_dim) {
    feature_to_embedding_dim_[std::string(feature_name)] = embedding_dim;
  }
};

TEST(CpuEmbeddingLookupTransformTest, Execute) {
  // 1. Setup Resource and Engine
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  // Create a 2x3 table:
  // [[1.0, 2.0, 3.0],
  //  [4.0, 5.0, 6.0]]
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));

  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  // 2. Setup Transform
  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"}));

  // 3. Prepare Inputs
  std::vector<int64_t> ids_vec = {0, 0, 1};
  std::vector<int32_t> rp_vec = {0, 1, 3};
  std::vector<float> w_vec = {1.0, 0.5, 0.5};

  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                         xla::ifrt::Shape({3}));
  absl::c_copy(ids_vec, reinterpret_cast<int64_t*>(ids_hb.base()));

  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({3}));
  absl::c_copy(rp_vec, reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer w_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                       xla::ifrt::Shape({3}));
  absl::c_copy(w_vec, reinterpret_cast<float*>(w_hb.base()));

  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(w_hb)}));

  api::ExecuteContext context;
  context.inputs = inputs;

  // 4. Execute
  EXPECT_OK(transform->Execute(context));

  // 5. Verify Outputs
  ASSERT_EQ(context.outputs.size(), 1);
  auto out_vector = context.outputs[0].Get<std::vector<xla_serving::HostBuffer>>();
  ASSERT_EQ(out_vector.size(), 1);
  xla_serving::HostBuffer out_hb = out_vector[0];
  EXPECT_EQ(out_hb.dtype().kind(), xla::ifrt::DType::Kind::kF32);
  EXPECT_EQ(out_hb.shape(), xla::ifrt::Shape({2, 3}));

  const float* out_data = reinterpret_cast<const float*>(out_hb.base());
  std::vector<float> output_vec(out_data, out_data + 6);

  // Expected output (BatchSize=2, EmbeddingDim=3):
  // Row 0: 1.0 * [1.0, 2.0, 3.0] = [1.0, 2.0, 3.0]
  // Row 1: 0.5 * [1.0, 2.0, 3.0] + 0.5 * [4.0, 5.0, 6.0] = [2.5, 3.5, 4.5]
  EXPECT_THAT(output_vec,
              ElementsAre(FloatEq(1.0), FloatEq(2.0), FloatEq(3.0),
                          FloatEq(2.5), FloatEq(3.5), FloatEq(4.5)));
}

TEST(CpuEmbeddingLookupTransformTest, DenseExecute) {
  // 1. Setup Resource and Engine
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  // Create a 2x3 table
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  // 2. Setup Transform with dense_lookup = true
  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  // 3. Prepare Inputs: 2D ID tensor of shape [2, 2] (batch_size=2, valency=2)
  // Row 0: look up IDs [0, 1]
  // Row 1: look up IDs [1, 0]
  std::vector<int32_t> ids_vec = {0, 1, 1, 0};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2, 2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<api::Value> dense_inputs;
  dense_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));

  api::ExecuteContext context;
  context.inputs = dense_inputs;

  // 4. Execute
  EXPECT_OK(transform->Execute(context));

  // 5. Verify Outputs
  ASSERT_EQ(context.outputs.size(), 1);
  auto out_vector = context.outputs[0].Get<std::vector<xla_serving::HostBuffer>>();
  ASSERT_EQ(out_vector.size(), 1);
  xla_serving::HostBuffer out_hb = out_vector[0];
  EXPECT_EQ(out_hb.dtype().kind(), xla::ifrt::DType::Kind::kF32);
  EXPECT_EQ(out_hb.shape(), xla::ifrt::Shape({2, 3}));

  const float* out_data = reinterpret_cast<const float*>(out_hb.base());
  std::vector<float> output_vec(out_data, out_data + 6);

  // Expected output with SUM combiner (default for table):
  // Row 0: [1.0, 2.0, 3.0] + [4.0, 5.0, 6.0] = [5.0, 7.0, 9.0]
  // Row 1: [4.0, 5.0, 6.0] + [1.0, 2.0, 3.0] = [5.0, 7.0, 9.0]
  EXPECT_THAT(output_vec,
              ElementsAre(FloatEq(5.0), FloatEq(7.0), FloatEq(9.0),
                          FloatEq(5.0), FloatEq(7.0), FloatEq(9.0)));
}

TEST(CpuEmbeddingLookupTransformTest, MultiLookupSuccess) {
  // 1. Setup JCE Engine with 2 tables
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> tableA_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape shapeA({2, 3});
  auto arrayA = cpu_embedding::Array(
      absl::MakeConstSpan(tableA_data.data(), tableA_data.size()), shapeA);
  ASSERT_OK_AND_ASSIGN(auto tableA,
                       cpu_embedding::EmbeddingTable::Create(
                           "tableA", std::move(arrayA),
                           cpu_embedding::EmbeddingTable::CombineMode::kMean));
  EXPECT_OK(engine->AddLocalTable("tableA", std::move(tableA)));

  std::vector<float> tableB_data = {7.0, 8.0, 9.0, 10.0, 11.0, 12.0};
  cpu_embedding::Shape shapeB({2, 3});
  auto arrayB = cpu_embedding::Array(
      absl::MakeConstSpan(tableB_data.data(), tableB_data.size()), shapeB);
  ASSERT_OK_AND_ASSIGN(auto tableB,
                       cpu_embedding::EmbeddingTable::Create(
                           "tableB", std::move(arrayB),
                           cpu_embedding::EmbeddingTable::CombineMode::kMean));
  EXPECT_OK(engine->AddLocalTable("tableB", std::move(tableB)));

  cpu_embedding::FeatureSpec specA;
  specA.name = "tableA";
  specA.table_name = "tableA";
  EXPECT_OK(engine->RegisterFeatureSpec(specA));

  cpu_embedding::FeatureSpec specB;
  specB.name = "tableB";
  specB.table_name = "tableB";
  EXPECT_OK(engine->RegisterFeatureSpec(specB));

  resource->RegisterEmbeddingDim("tableA", 3);
  resource->RegisterEmbeddingDim("tableB", 3);
  // 2. Setup Transform with 2 features
  ASSERT_OK_AND_ASSIGN(auto transform, CpuEmbeddingLookupTransform::Create(
                                           resource, std::vector<std::string>{
                                                         "tableA", "tableB"}));

  // 3. Prepare Inputs
  // Table A inputs
  std::vector<int64_t> idsA = {0, 0, 1};
  std::vector<int32_t> rpA = {0, 1, 3};
  std::vector<float> wA = {1.0, 0.5, 0.5};

  xla_serving::HostBuffer hb_idsA(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                          xla::ifrt::Shape({3}));
  absl::c_copy(idsA, reinterpret_cast<int64_t*>(hb_idsA.base()));

  xla_serving::HostBuffer hb_rpA(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({3}));
  absl::c_copy(rpA, reinterpret_cast<int32_t*>(hb_rpA.base()));

  xla_serving::HostBuffer hb_wA(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                        xla::ifrt::Shape({3}));
  absl::c_copy(wA, reinterpret_cast<float*>(hb_wA.base()));

  // Table B inputs
  std::vector<int64_t> idsB = {1, 1};
  std::vector<int32_t> rpB = {0, 2};
  std::vector<float> wB = {0.5, 0.5};

  xla_serving::HostBuffer hb_idsB(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                          xla::ifrt::Shape({2}));
  absl::c_copy(idsB, reinterpret_cast<int64_t*>(hb_idsB.base()));

  xla_serving::HostBuffer hb_rpB(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(rpB, reinterpret_cast<int32_t*>(hb_rpB.base()));

  xla_serving::HostBuffer hb_wB(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                        xla::ifrt::Shape({2}));
  absl::c_copy(wB, reinterpret_cast<float*>(hb_wB.base()));

  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(
      std::vector<xla_serving::HostBuffer>{std::move(hb_idsA), std::move(hb_idsB)}));
  inputs.push_back(api::Value(
      std::vector<xla_serving::HostBuffer>{std::move(hb_rpA), std::move(hb_rpB)}));
  inputs.push_back(api::Value(
      std::vector<xla_serving::HostBuffer>{std::move(hb_wA), std::move(hb_wB)}));

  api::ExecuteContext context;
  context.inputs = std::move(inputs);

  // 4. Execute
  EXPECT_OK(transform->Execute(context));
  ASSERT_EQ(context.outputs.size(), 1);
  auto out_vector = context.outputs[0].Get<std::vector<xla_serving::HostBuffer>>();
  ASSERT_EQ(out_vector.size(), 2);

  // 5. Verify Table A Output
  xla_serving::HostBuffer outA = std::move(out_vector[0]);
  EXPECT_EQ(outA.shape(), xla::ifrt::Shape({2, 3}));
  const float* dataA = reinterpret_cast<const float*>(outA.base());
  std::vector<float> vecA(dataA, dataA + 6);
  EXPECT_THAT(vecA, ElementsAre(FloatEq(1.0), FloatEq(2.0), FloatEq(3.0),
                                FloatEq(2.5), FloatEq(3.5), FloatEq(4.5)));

  // 6. Verify Table B Output
  xla_serving::HostBuffer outB = std::move(out_vector[1]);
  EXPECT_EQ(outB.shape(), xla::ifrt::Shape({1, 3}));
  const float* dataB = reinterpret_cast<const float*>(outB.base());
  std::vector<float> vecB(dataB, dataB + 3);
  EXPECT_THAT(vecB, ElementsAre(FloatEq(10.0), FloatEq(11.0), FloatEq(12.0)));
}

TEST(CpuEmbeddingLookupTransformTest, ScalarRowPointersFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"}));

  std::vector<int64_t> ids_vec = {0};
  std::vector<float> w_vec = {1.0};

  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
                         xla::ifrt::Shape({1}));
  absl::c_copy(ids_vec, reinterpret_cast<int64_t*>(ids_hb.base()));

  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({}));
  int32_t rp_val = 0;
  absl::c_copy(absl::MakeConstSpan(&rp_val, 1),
               reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer w_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                       xla::ifrt::Shape({1}));
  absl::c_copy(w_vec, reinterpret_cast<float*>(w_hb.base()));

  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{std::move(w_hb)}));

  api::ExecuteContext context;
  context.inputs = inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr(
                  "row_pointers HostBuffer must be at least 1D (got scalar)."));
}

TEST(CpuEmbeddingLookupTransformTest, DuplicateFeaturesFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");

  auto transform_or = CpuEmbeddingLookupTransform::Create(
      resource, std::vector<std::string>{"feature1", "feature1"});
  EXPECT_FALSE(transform_or.ok());
}

TEST(CpuEmbeddingLookupTransformTest, DuplicateOutputsFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");

  auto transform_or = CpuEmbeddingLookupTransform::Create(
      resource, std::vector<std::string>{"feature1", "feature2"},
      /*dense_lookup=*/false, std::vector<std::string>{"out1", "out1"});
  EXPECT_FALSE(transform_or.ok());
}

TEST(CpuEmbeddingLookupTransformTest, OutputFeatureNamesSizeMismatchFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");

  auto transform_or = CpuEmbeddingLookupTransform::Create(
      resource, std::vector<std::string>{"feature1", "feature2"},
      /*dense_lookup=*/false, std::vector<std::string>{"out1"});
  EXPECT_FALSE(transform_or.ok());
}

TEST(CpuEmbeddingLookupTransformTest, NonMonotonicRowPointersFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  std::vector<int64_t> ids_vec = {0, 0, 1};
  std::vector<int32_t> rp_vec = {0, 2, 1};
  std::vector<float> w_vec = {1.0, 0.5, 0.5};

  auto allocator = [](size_t size) {
    static std::vector<float> alloc_buffer;
    alloc_buffer.resize(size);
    return absl::MakeSpan(alloc_buffer);
  };

  EXPECT_THAT(
      engine->Execute<float>("feature1", ids_vec, rp_vec, w_vec, allocator),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Row pointers must be monotonic.")));
}

TEST(CpuEmbeddingLookupTransformTest, NegativeRowPointersFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  std::vector<int64_t> ids_vec = {0, 0, 1};
  std::vector<int32_t> rp_vec = {-1, 1, 3};
  std::vector<float> w_vec = {1.0, 0.5, 0.5};

  auto allocator = [](size_t size) {
    static std::vector<float> alloc_buffer;
    alloc_buffer.resize(size);
    return absl::MakeSpan(alloc_buffer);
  };

  EXPECT_THAT(
      engine->Execute<float>("feature1", ids_vec, rp_vec, w_vec, allocator),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr("Row pointers must be non-negative.")));
}

TEST(CpuEmbeddingLookupTransformTest, OutOfBoundsRowPointersFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  std::vector<int64_t> ids_vec = {0, 0, 1};
  std::vector<int32_t> rp_vec = {0, 1, 4};
  std::vector<float> w_vec = {1.0, 0.5, 0.5, 0.5};

  auto allocator = [](size_t size) {
    static std::vector<float> alloc_buffer;
    alloc_buffer.resize(size);
    return absl::MakeSpan(alloc_buffer);
  };

  EXPECT_THAT(
      engine->Execute<float>("feature1", ids_vec, rp_vec, w_vec, allocator),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Final element of row pointers exceeds embedding IDs size.")));
}

TEST(CpuEmbeddingLookupTransformTest, WeightsSizeMismatchFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();

  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0};
  cpu_embedding::Shape table_shape({2, 3});
  auto table_array =
      cpu_embedding::Array(absl::MakeConstSpan(table_data), table_shape);
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));

  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 3);

  std::vector<int64_t> ids_vec = {0, 0, 1};
  std::vector<int32_t> rp_vec = {0, 1, 3};
  std::vector<float> w_vec = {1.0, 0.5};

  auto allocator = [](size_t size) {
    static std::vector<float> alloc_buffer;
    alloc_buffer.resize(size);
    return absl::MakeSpan(alloc_buffer);
  };

  EXPECT_THAT(
      engine->Execute<float>("feature1", ids_vec, rp_vec, w_vec, allocator),
      absl_testing::StatusIs(
          absl::StatusCode::kInvalidArgument,
          testing::HasSubstr(
              "Weights size is less than the final element of row pointers.")));
}

TEST(CpuEmbeddingLookupTransformTest, DenseExecuteIntegerOverflowFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  resource->RegisterEmbeddingDim("feature1", 2000);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  void* dummy_ptr = reinterpret_cast<void*>(0x100000000);
  xla_serving::HostBuffer ids_hb(
      dummy_ptr, xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
      xla::ifrt::Shape({2000000, 2000}), std::nullopt, nullptr);

  std::vector<api::Value> dense_inputs;
  dense_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));

  api::ExecuteContext context;
  context.inputs = dense_inputs;

  EXPECT_THAT(transform->Execute(context),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Batch volume (batch_size * valency) "
                                     "exceeds INT32_MAX bounds.")));
}

TEST(CpuEmbeddingLookupTransformTest, DenseExecuteBatchSizeOverflowFails) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  resource->RegisterEmbeddingDim("feature1", 1);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  void* dummy_ptr = reinterpret_cast<void*>(0x100000000);
  xla_serving::HostBuffer ids_hb(
      dummy_ptr, xla::ifrt::DType(xla::ifrt::DType::Kind::kS64),
      xla::ifrt::Shape({2147483648LL, 1}), std::nullopt, nullptr);

  std::vector<api::Value> dense_inputs;
  dense_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  dense_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));

  api::ExecuteContext context;
  context.inputs = dense_inputs;

  EXPECT_THAT(transform->Execute(context),
              absl_testing::StatusIs(
                  absl::StatusCode::kInvalidArgument,
                  testing::HasSubstr("Batch size exceeds INT32_MAX bounds.")));
}

TEST(CpuEmbeddingLookupTransformTest, ExecuteMissingInputsReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({1, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  api::ExecuteContext context;
  context.inputs = {};  // Missing inputs

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      std::string(status.message()),
      testing::HasSubstr(
          "Expected 3 input(s) when calling Execute, but got 0 instead."));
}

TEST(CpuEmbeddingLookupTransformTest, ExecuteInvalidIdTypeReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({1, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  // 3. Prepare Inputs: 1D ID tensor with FLOAT data!
  std::vector<float> ids_vec = {0.0};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                         xla::ifrt::Shape({1, 1}));
  absl::c_copy(ids_vec, reinterpret_cast<float*>(ids_hb.base()));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("Embedding IDs must be int32 or int64."));
}
TEST(CpuEmbeddingLookupTransformTest,
     ExecuteInvalidRowPointersSizeReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/false));

  std::vector<int32_t> ids_vec = {0, 1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<int32_t> rp_vec = {0};
  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({1}));
  absl::c_copy(rp_vec, reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer weights_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({0}));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(weights_hb)}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      std::string(status.message()),
      testing::HasSubstr("Row pointers must have at least 2 elements."));
}

TEST(CpuEmbeddingLookupTransformTest,
     ExecuteInvalidRowPointersFirstElementReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/false));

  std::vector<int32_t> ids_vec = {0, 1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<int32_t> rp_vec = {1, 2};
  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({2}));
  absl::c_copy(rp_vec, reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer weights_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({0}));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(weights_hb)}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("First element of row pointers must be 0."));
}

TEST(CpuEmbeddingLookupTransformTest,
     ExecuteInvalidRowPointersLastElementReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/false));

  std::vector<int32_t> ids_vec = {0, 1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<int32_t> rp_vec = {0, 3};
  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({2}));
  absl::c_copy(rp_vec, reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer weights_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({0}));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(weights_hb)}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      std::string(status.message()),
      testing::HasSubstr("does not match last element of row pointers"));
}

TEST(CpuEmbeddingLookupTransformTest, ExecuteRowPointersNotSortedReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/false));

  std::vector<int32_t> ids_vec = {0, 1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<int32_t> rp_vec_unsorted = {0, 3, 2};
  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({3}));
  absl::c_copy(rp_vec_unsorted, reinterpret_cast<int32_t*>(rp_hb.base()));

  xla_serving::HostBuffer weights_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({0}));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(weights_hb)}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("Row pointers must be monotonic."));
}

TEST(CpuEmbeddingLookupTransformTest, ExecuteWeightsSizeMismatchReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/false));

  std::vector<int32_t> ids_vec = {0, 1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({2}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<int32_t> rp_vec = {0, 2};
  xla_serving::HostBuffer rp_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                        xla::ifrt::Shape({2}));
  absl::c_copy(rp_vec, reinterpret_cast<int32_t*>(rp_hb.base()));

  std::vector<float> weights_vec = {1.0f, 1.0f, 1.0f};
  xla_serving::HostBuffer weights_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({3}));
  absl::c_copy(weights_vec, reinterpret_cast<float*>(weights_hb.base()));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(rp_hb)}));
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(weights_hb)}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(
      std::string(status.message()),
      testing::HasSubstr("Weights size (3) does not match expected size (2)"));
}
TEST(CpuEmbeddingLookupTransformTest, ExecuteNegativeEmbeddingIdReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  std::vector<int32_t> ids_vec = {-1};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({1, 1}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("is out of bounds [0, 2)"));
}

TEST(CpuEmbeddingLookupTransformTest,
     ExecuteOutOfRangeEmbeddingIdReturnsError) {
  auto resource = std::make_shared<TestCpuEmbeddingEngineResource>(
      "engine_res", std::vector<Asset>{}, "CpuEmbeddingEngineResource");
  auto* engine = resource->Get();
  std::vector<float> table_data = {1.0, 2.0, 3.0, 4.0};
  auto table_array = cpu_embedding::Array(absl::MakeConstSpan(table_data),
                                          cpu_embedding::Shape({2, 2}));
  ASSERT_OK_AND_ASSIGN(auto table, cpu_embedding::EmbeddingTable::Create(
                                       "table1", std::move(table_array)));
  EXPECT_OK(engine->AddLocalTable("table1", std::move(table)));
  cpu_embedding::FeatureSpec spec;
  spec.name = "feature1";
  spec.table_name = "table1";
  EXPECT_OK(engine->RegisterFeatureSpec(spec));
  resource->RegisterEmbeddingDim("feature1", 2);

  ASSERT_OK_AND_ASSIGN(auto transform,
                       CpuEmbeddingLookupTransform::Create(
                           resource, std::vector<std::string>{"feature1"},
                           /*dense_lookup=*/true));

  std::vector<int32_t> ids_vec = {2};
  xla_serving::HostBuffer ids_hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                         xla::ifrt::Shape({1, 1}));
  absl::c_copy(ids_vec, reinterpret_cast<int32_t*>(ids_hb.base()));

  std::vector<api::Value> test_inputs;
  test_inputs.push_back(
      api::Value(std::vector<xla_serving::HostBuffer>{std::move(ids_hb)}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  test_inputs.push_back(api::Value(std::vector<xla_serving::HostBuffer>{}));
  api::ExecuteContext context;
  context.inputs = test_inputs;

  auto status = transform->Execute(context);
  EXPECT_FALSE(status.ok());
  EXPECT_THAT(std::string(status.message()),
              testing::HasSubstr("is out of bounds [0, 2)"));
}
}  // namespace
}  // namespace composite
}  // namespace xla_serving::data

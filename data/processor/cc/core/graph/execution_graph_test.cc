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

#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace core {
namespace {

class TestTransform : public TransformBase {
 public:
  absl::Status Execute(api::ExecuteContext& context) override {
    return absl::OkStatus();
  }
  absl::Span<const api::TypeInfo> InputTypes() const override { return {}; }
  absl::Span<const api::TypeInfo> OutputTypes() const override { return {}; }

  // DebugString is used to log any helpful state about the class you might want
  // logged when manually debugging.
  absl::string_view DebugString() const override {
    return "TestTransform; This transform does not do anything.";
  }

  // DebugName is used to identify nodes in the graph. Use this to debug your
  // graph connections.
  std::string DebugName() const override { return "TestTransform"; }

  absl::StatusOr<TransformConfig> ToProto() const override {
    return absl::UnimplementedError("ToProto not implemented");
  }
};

TEST(ExecutionGraphTest, CreateEmpty) {
  ASSERT_OK_AND_ASSIGN(ExecutionGraph graph, ExecutionGraph::Create({}));
  EXPECT_TRUE(graph.empty());
  EXPECT_EQ(graph.size(), 0);
}

TEST(ExecutionGraphTest, CreateFromVector) {
  std::vector<std::shared_ptr<TransformBase>> transforms;
  transforms.push_back(std::make_shared<TestTransform>());
  transforms.push_back(std::make_shared<TestTransform>());

  ASSERT_OK_AND_ASSIGN(ExecutionGraph graph,
                       ExecutionGraph::Create(std::move(transforms)));
  EXPECT_FALSE(graph.empty());
  EXPECT_EQ(graph.size(), 2);

  EXPECT_EQ(graph[0].node_index, 0);
  EXPECT_EQ(graph[0].name, "TestTransform_0");

  EXPECT_EQ(graph[1].node_index, 1);
  EXPECT_EQ(graph[1].name, "TestTransform_1");
}

TEST(ExecutionGraphTest, AddNode) {
  ExecutionGraph graph;

  int idx0 = graph.AddNode("first", std::make_shared<TestTransform>(), {});
  EXPECT_EQ(graph.size(), 1);
  EXPECT_EQ(idx0, 0);
  EXPECT_EQ(graph[0].name, "first");

  int idx1 = graph.AddNode("second", std::make_shared<TestTransform>(), {});
  EXPECT_EQ(graph.size(), 2);
  EXPECT_EQ(idx1, 1);
  EXPECT_EQ(graph[1].name, "second");
}

TEST(ExecutionGraphTest, ConstAccess) {
  std::vector<std::shared_ptr<TransformBase>> transforms;
  transforms.push_back(std::make_shared<TestTransform>());
  ASSERT_OK_AND_ASSIGN(ExecutionGraph graph,
                       ExecutionGraph::Create(std::move(transforms)));

  EXPECT_FALSE(graph.empty());
  EXPECT_EQ(graph.size(), 1);
  EXPECT_EQ(graph[0].name, "TestTransform_0");
}

TEST(ExecutionGraphTest, GraphEdgeValidation) {
  EXPECT_EQ(GraphEdge::GraphInput(-1).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(GraphEdge::NodeOutput(-1, 0).status().code(),
            absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(GraphEdge::NodeOutput(0, -1).status().code(),
            absl::StatusCode::kInvalidArgument);
}

TEST(ExecutionGraphTest, GetNumInputsAndOutputs) {
  ExecutionGraph graph;
  EXPECT_EQ(graph.GetNumInputs(), 0);
  EXPECT_EQ(graph.GetNumOutputs(), 0);

  graph.AddNode(
      "first", std::make_shared<TestTransform>(),
      {GraphEdge::GraphInput(0).value(), GraphEdge::GraphInput(2).value()});
  EXPECT_EQ(graph.GetNumInputs(), 3);
  EXPECT_EQ(graph.GetNumOutputs(), 0);

  graph.SetOutputs({GraphEdge::GraphInput(4).value()});
  EXPECT_EQ(graph.GetNumInputs(), 5);
  EXPECT_EQ(graph.GetNumOutputs(), 1);
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace neptune::data

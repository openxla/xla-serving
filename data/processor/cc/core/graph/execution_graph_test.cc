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

#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data {
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

TEST(ExecutionGraphTest, GetNumReferencedInputsAndOutputs) {
  ExecutionGraph graph;
  EXPECT_EQ(graph.GetNumReferencedInputs(), 0);
  EXPECT_EQ(graph.GetNumOutputs(), 0);

  graph.AddNode(
      "first", std::make_shared<TestTransform>(),
      {GraphEdge::GraphInput(0).value(), GraphEdge::GraphInput(2).value()});
  EXPECT_EQ(graph.GetNumReferencedInputs(), 3);
  EXPECT_EQ(graph.GetNumOutputs(), 0);

  graph.SetOutputs({GraphEdge::GraphInput(4).value()});
  EXPECT_EQ(graph.GetNumReferencedInputs(), 5);
  EXPECT_EQ(graph.GetNumOutputs(), 1);
}

TEST(ExecutionGraphTest, GetNumInputsWithoutSetNumInputsDies) {
  ExecutionGraph graph;
  graph.AddNode("first", std::make_shared<TestTransform>(),
                {GraphEdge::GraphInput(0).value()});
  EXPECT_EQ(graph.declared_num_inputs(), std::nullopt);
  EXPECT_EQ(graph.GetNumReferencedInputs(), 1);
  EXPECT_DEATH(graph.GetNumInputs(), "SetNumInputs");
}

TEST(ExecutionGraphTest, DeclaredNumInputsIncludesUnreadTrailingInputs) {
  ExecutionGraph graph;
  graph.AddNode("first", std::make_shared<TestTransform>(),
                {GraphEdge::GraphInput(0).value()});
  graph.SetNumInputs(3);
  EXPECT_EQ(graph.declared_num_inputs(), 3);
  EXPECT_EQ(graph.GetNumReferencedInputs(), 1);
  EXPECT_EQ(graph.GetNumInputs(), 3);
}

TEST(ExecutionGraphTest, DeclaredZeroInputs) {
  ExecutionGraph graph;
  graph.SetNumInputs(0);
  EXPECT_EQ(graph.declared_num_inputs(), 0);
  EXPECT_EQ(graph.GetNumInputs(), 0);
}

TEST(ExecutionGraphTest, ClearResetsDeclaredNumInputs) {
  ExecutionGraph graph;
  graph.AddNode("first", std::make_shared<TestTransform>(),
                {GraphEdge::GraphInput(0).value()});
  graph.SetNumInputs(3);
  GraphIOStructureProto io_structure;
  io_structure.mutable_input_structure()->set_type(PyTreeProto::LEAF);
  graph.SetIOStructure(io_structure);

  graph.clear();
  EXPECT_EQ(graph.declared_num_inputs(), std::nullopt);
  EXPECT_EQ(graph.io_structure().input_structure().type(),
            PyTreeProto::INVALID);
  EXPECT_DEATH(graph.GetNumInputs(), "SetNumInputs");
}

TEST(ExecutionGraphTest, DeclaredNumInputsFallsBackToIOStructureAndUpdates) {
  ExecutionGraph graph;

  GraphIOStructureProto io1;
  io1.mutable_input_structure()->set_type(PyTreeProto::LEAF);
  io1.mutable_captured_structure()->set_type(PyTreeProto::LEAF);
  graph.SetIOStructure(io1);
  EXPECT_EQ(graph.declared_num_inputs(), 2);
  EXPECT_EQ(graph.GetNumInputs(), 2);

  GraphIOStructureProto io2;
  io2.mutable_input_structure()->set_type(PyTreeProto::TUPLE);
  io2.mutable_input_structure()->add_children()->set_type(PyTreeProto::LEAF);
  io2.mutable_input_structure()->add_children()->set_type(PyTreeProto::LEAF);
  io2.mutable_input_structure()->add_children()->set_type(PyTreeProto::LEAF);
  graph.SetIOStructure(io2);
  EXPECT_EQ(graph.declared_num_inputs(), 3);
  EXPECT_EQ(graph.GetNumInputs(), 3);
}

TEST(ExecutionGraphTest, SetNumInputsTakesPrecedenceOverIOStructure) {
  ExecutionGraph graph;
  GraphIOStructureProto io;
  io.mutable_input_structure()->set_type(PyTreeProto::TUPLE);
  io.mutable_input_structure()->add_children()->set_type(PyTreeProto::LEAF);
  io.mutable_input_structure()->add_children()->set_type(PyTreeProto::LEAF);

  graph.SetNumInputs(5);
  graph.SetIOStructure(io);
  EXPECT_EQ(graph.declared_num_inputs(), 5);
  EXPECT_EQ(graph.GetNumInputs(), 5);

  ExecutionGraph graph2;
  graph2.SetIOStructure(io);
  EXPECT_EQ(graph2.GetNumInputs(), 2);
  graph2.SetNumInputs(1);
  graph2.SetIOStructure(io);
  EXPECT_EQ(graph2.declared_num_inputs(), 1);
  EXPECT_EQ(graph2.GetNumInputs(), 1);
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

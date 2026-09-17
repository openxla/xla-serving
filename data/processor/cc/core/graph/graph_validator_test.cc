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

#include "data/processor/cc/core/graph/graph_validator.h"  // from @xla_serving

#include <memory>
#include <string>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {
namespace {

using ::absl_testing::StatusIs;
using ::testing::HasSubstr;

class IntToIntTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<IntToIntTransform>> Create() {
    return std::make_unique<IntToIntTransform>();
  }
  int operator()(int x) { return x; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    return TransformConfig();
  }
};
ND_REGISTER_TRANSFORM(IntToIntTransform, "IntToIntTransform");

class StringToStringTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<StringToStringTransform>> Create() {
    return std::make_unique<StringToStringTransform>();
  }
  std::string operator()(std::string x) { return x; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    return TransformConfig();
  }
};
ND_REGISTER_TRANSFORM(StringToStringTransform, "StringToStringTransform");

TEST(GraphValidatorTest, EmptyGraphFails) {
  ExecutionGraph graph;
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected empty graph")));
}

TEST(GraphValidatorTest, ValidGraphSucceeds) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 0).value()});
  EXPECT_OK(GraphValidator::Validate(graph));
}

TEST(GraphValidatorTest, WrongInputEdgeCountFails) {
  ExecutionGraph graph;
  graph.AddNode("node0", std::make_shared<IntToIntTransform>(), {});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("expects 1 inputs, but has 0 input edges")));
}

TEST(GraphValidatorTest, TopologicallyUnsortedNodeOutputFails) {
  ExecutionGraph graph;
  // Node 0 depends on node 0
  graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                {GraphEdge::NodeOutput(0, 0).value()});
  EXPECT_THAT(
      GraphValidator::Validate(graph),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Graph is not topologically sorted or has invalid edge")));
}

TEST(GraphValidatorTest, InvalidNodeOutputPortIndexFails) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  // Node 1 references out-of-bounds port 1 on node 0 (which only has port 0)
  graph.AddNode("node1", std::make_shared<IntToIntTransform>(),
                {GraphEdge::NodeOutput(node0, 1).value()});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid port index: node 'node1'")));
}

TEST(GraphValidatorTest, TypeMismatchFails) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("int_node", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  // Connect int output from node0 to string input of node1
  graph.AddNode("str_node", std::make_shared<StringToStringTransform>(),
                {GraphEdge::NodeOutput(node0, 0).value()});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type mismatch: node 'str_node'")));
}

TEST(GraphValidatorTest, InvalidGraphOutputNodeIndexFails) {
  ExecutionGraph graph;
  graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(5, 0).value()});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("references invalid node index 5")));
}

TEST(GraphValidatorTest, InvalidGraphOutputPortIndexFails) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 5).value()});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("references invalid port index 5")));
}

TEST(GraphValidatorTest, NegativeGraphOutputNodeIndexFails) {
  ExecutionGraph graph;
  graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({{GraphEdge::SourceType::kNodeOutput, -1, 0}});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("references invalid node index -1")));
}

TEST(GraphValidatorTest, NegativeGraphOutputPortIndexFails) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({{GraphEdge::SourceType::kNodeOutput, node0, -1}});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("references invalid port index -1")));
}

TEST(GraphValidatorTest, NegativeInternalEdgeSourceIndexFails) {
  ExecutionGraph graph;
  graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                {{GraphEdge::SourceType::kNodeOutput, -1, 0}});
  EXPECT_THAT(
      GraphValidator::Validate(graph),
      StatusIs(
          absl::StatusCode::kInvalidArgument,
          HasSubstr("Graph is not topologically sorted or has invalid edge")));
}

TEST(GraphValidatorTest, NegativeInternalEdgePortIndexFails) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("node0", std::make_shared<IntToIntTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.AddNode("node1", std::make_shared<IntToIntTransform>(),
                {{GraphEdge::SourceType::kNodeOutput, node0, -1}});
  EXPECT_THAT(GraphValidator::Validate(graph),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Invalid port index: node 'node1'")));
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

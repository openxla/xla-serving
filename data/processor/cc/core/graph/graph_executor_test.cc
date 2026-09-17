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

#include "data/processor/cc/core/graph/graph_executor.h"  // from @xla_serving

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/type_info.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace neptune::data {
namespace composite {
namespace core {
namespace {

class IncrementTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<IncrementTransform>> Create() {
    return std::make_unique<IncrementTransform>();
  }
  int operator()(int a) const { return a + 1; }
  absl::string_view DebugString() const override {
    return "IncrementTransform(int value) => value + 1;";
  }
  std::string DebugName() const override { return "IncrementTransform"; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    return absl::UnimplementedError("ToProto not implemented");
  }
};

ND_REGISTER_TRANSFORM(IncrementTransform, "IncrementTransform");

class SumTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<SumTransform>> Create() {
    return std::make_unique<SumTransform>();
  }
  int operator()(int a, int b) const { return a + b; }
  absl::string_view DebugString() const override {
    return "SumTransform(int a, int b) => a + b;";
  }
  std::string DebugName() const override { return "SumTransform"; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    return absl::UnimplementedError("ToProto not implemented");
  }
};

ND_REGISTER_TRANSFORM(SumTransform, "SumTransform");

class SplitTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<SplitTransform>> Create() {
    return std::make_unique<SplitTransform>();
  }
  absl::Status Execute(api::ExecuteContext& context) override {
    if (context.inputs.size() != 1) {
      return absl::InvalidArgumentError("Expected 1 input");
    }
    int val = context.inputs[0].Get<int>();
    api::Value out1;
    api::Value out2;
    out1.Set(val + 1);
    out2.Set(val + 2);
    context.outputs.push_back(std::move(out1));
    context.outputs.push_back(std::move(out2));
    return absl::OkStatus();
  }
  absl::Span<const api::TypeInfo> InputTypes() const override {
    static const std::vector<api::TypeInfo> types = {
        api::CreateTypeInfo<int>()};
    return absl::MakeConstSpan(types);
  }
  absl::Span<const api::TypeInfo> OutputTypes() const override {
    static const std::vector<api::TypeInfo> types = {
        api::CreateTypeInfo<int>(), api::CreateTypeInfo<int>()};
    return absl::MakeConstSpan(types);
  }
  absl::string_view DebugString() const override {
    return "SplitTransform(int a) => (a + 1, a + 2);";
  }
  std::string DebugName() const override { return "SplitTransform"; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    return absl::UnimplementedError("ToProto not implemented");
  }
};

ND_REGISTER_TRANSFORM(SplitTransform, "SplitTransform");

TEST(GraphExecutorTest, SingleThreadedSynchronousExecution) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  int node1 = graph.AddNode("inc1", std::make_shared<IncrementTransform>(),
                            {GraphEdge::NodeOutput(node0, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node1, 0).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(10));

  api::ExecuteContext context;
  context.inputs = inputs;

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].Get<int>(), 12);
}

TEST(GraphExecutorTest, NodeWithMultipleInputs) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  int node1 = graph.AddNode("inc1", std::make_shared<IncrementTransform>(),
                            {GraphEdge::GraphInput(1).value()});
  int node2 = graph.AddNode("sum2", std::make_shared<SumTransform>(),
                            {GraphEdge::NodeOutput(node0, 0).value(),
                             GraphEdge::NodeOutput(node1, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node2, 0).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  std::vector<api::Value> inputs = {api::Value(10), api::Value(20)};

  api::ExecuteContext context;
  context.inputs = inputs;

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].Get<int>(), 32);  // (10+1) + (20+1) = 32
}

TEST(GraphExecutorTest, NodeWithMultipleOutputs) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("split0", std::make_shared<SplitTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 0).value(),
                    GraphEdge::NodeOutput(node0, 1).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  std::vector<api::Value> inputs = {api::Value(10)};

  api::ExecuteContext context;
  context.inputs = inputs;

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 2);
  EXPECT_EQ(context.outputs[0].Get<int>(), 11);
  EXPECT_EQ(context.outputs[1].Get<int>(), 12);
}

TEST(GraphExecutorTest, GraphWithConstantInput) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::Constant(api::Value(5))});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 0).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  api::ExecuteContext context;  // No inputs in context

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].Get<int>(), 6);
}

TEST(GraphExecutorTest, GraphWithConstantOutput) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 0).value(),
                    GraphEdge::Constant(api::Value(42))});

  SingleThreadedSynchronousGraphExecutor executor;
  std::vector<api::Value> inputs = {api::Value(10)};

  api::ExecuteContext context;
  context.inputs = inputs;

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 2);
  EXPECT_EQ(context.outputs[0].Get<int>(), 11);
  EXPECT_EQ(context.outputs[1].Get<int>(), 42);
}

TEST(GraphExecutorTest, GraphWithPassThroughInputs) {
  ExecutionGraph graph;
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node0, 0).value(),
                    GraphEdge::GraphInput(0).value(),
                    GraphEdge::GraphInput(1).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  std::vector<api::Value> inputs = {api::Value(10), api::Value(20)};

  api::ExecuteContext context;
  context.inputs = inputs;

  ASSERT_OK(executor.Execute(graph, context));
  ASSERT_EQ(context.outputs.size(), 3);
  EXPECT_EQ(context.outputs[0].Get<int>(), 11);
  EXPECT_EQ(context.outputs[1].Get<int>(), 10);
  EXPECT_EQ(context.outputs[2].Get<int>(), 20);
}

TEST(GraphExecutorTest, GraphWithCyclicDependencyError) {
  ExecutionGraph graph;
  // node0 takes input from node1 (which is added later)
  int node0 = graph.AddNode("inc0", std::make_shared<IncrementTransform>(),
                            {GraphEdge::NodeOutput(1, 0).value()});
  // node1 takes input from node0
  int node1 = graph.AddNode("inc1", std::make_shared<IncrementTransform>(),
                            {GraphEdge::NodeOutput(node0, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node1, 0).value()});

  SingleThreadedSynchronousGraphExecutor executor;
  api::ExecuteContext context;

  // This should fail during execution because node0 tries to access node1's
  // output which has not been evaluated yet.
  absl::Status status = executor.Execute(graph, context);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), absl::StatusCode::kInternal);
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace neptune::data

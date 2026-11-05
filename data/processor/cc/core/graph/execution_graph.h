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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

// Represents an edge in the execution graph (inputs or outputs).
struct GraphEdge {
  enum class SourceType {
    kGraphInput,
    kNodeOutput,
    kConstant,
  };

  SourceType type;
  // Index of the graph input (if kGraphInput) OR node index in the graph (if
  // kNodeOutput)
  int source_index;
  // Output port index of the producer node (only if kNodeOutput)
  int port_index;
  // Constant value (only if kConstant)
  api::Value constant_value;

  static absl::StatusOr<GraphEdge> GraphInput(int index) {
    if (index < 0) {
      return absl::InvalidArgumentError(
          absl::StrCat("GraphInput index must be non-negative, got ", index));
    }
    return GraphEdge{SourceType::kGraphInput, index, -1, api::Value()};
  }

  static absl::StatusOr<GraphEdge> NodeOutput(int node_index, int port_index) {
    if (node_index < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "NodeOutput node_index must be non-negative, got ", node_index));
    }
    if (port_index < 0) {
      return absl::InvalidArgumentError(absl::StrCat(
          "NodeOutput port_index must be non-negative, got ", port_index));
    }
    return GraphEdge{SourceType::kNodeOutput, node_index, port_index,
                     api::Value()};
  }

  static GraphEdge Constant(api::Value value) {
    return {SourceType::kConstant, -1, -1, std::move(value)};
  }
};

// A node in the execution graph (DAG).
struct GraphNode {
  // The index of the node in the graph (its position in the nodes vector).
  int node_index;
  // The debug-friendly display name for the node.
  std::string name;
  std::shared_ptr<TransformBase> transform;
  // The sources for each input of this transform.
  // Must match the size of transform->InputTypes().
  std::vector<GraphEdge> input_edges;
  // Pre-constructed trace names to avoid latency in execution loop.
  std::string collect_node_inputs_trace_name;
  std::string execute_trace_name;
};

// Represents an execution graph, which can be a Directed Acyclic Graph (DAG)
// of transforms.
//
// The graph is composed of:
// - `GraphNode`: Represents a processing step, containing a `TransformBase`
//   and references to its input edges.
// - `GraphEdge`: Represents a data dependency (link) between nodes. An edge
//   can source from a graph input, another node's output port, or a constant
//   value.
// - Outputs: A list of `GraphEdge`s defining the outputs of the graph, which
//   are returned after the graph is fully executed.
//
// The graph execution order is determined by the topological order of the
// nodes, which is currently expected to be the order they are stored in the
// `nodes_` vector (i.e. nodes must be added in topological order).
class ExecutionGraph {
 public:
  // Left for backwards compatibility to support simple linear graph creation.
  static absl::StatusOr<ExecutionGraph> Create(
      std::vector<std::shared_ptr<TransformBase>>&& transforms) {
    ExecutionGraph graph;
    for (auto& transform : transforms) {
      std::string transform_name =
          absl::StrCat(transform->DebugName(), "_", graph.size());
      RETURN_IF_ERROR(graph.push_back(transform_name, transform));
    }

    if (!graph.empty()) {
      std::vector<GraphEdge> outputs;
      int last_node_idx = graph.size() - 1;
      int output_size = graph.nodes().back().transform->OutputTypes().size();
      outputs.reserve(output_size);
      for (int i = 0; i < output_size; ++i) {
        ASSIGN_OR_RETURN(auto edge, GraphEdge::NodeOutput(last_node_idx, i));
        outputs.push_back(std::move(edge));
      }
      graph.SetOutputs(std::move(outputs));
    }
    return graph;
  }

  int AddNode(std::string name, std::shared_ptr<TransformBase> transform,
              std::vector<GraphEdge> input_edges) {
    int index = nodes_.size();
    std::string collect_node_inputs_trace_name =
        absl::StrCat("CollectNodeInputs: ", name);
    std::string execute_trace_name = absl::StrCat("Execute: ", name);
    nodes_.push_back(GraphNode{index, std::move(name), std::move(transform),
                               std::move(input_edges),
                               std::move(collect_node_inputs_trace_name),
                               std::move(execute_trace_name)});
    return index;
  }

  void SetOutputs(std::vector<GraphEdge> outputs) {
    outputs_ = std::move(outputs);
  }

  void SetIOStructure(GraphIOStructureProto s) {
    io_structure_ = std::move(s);
  }

  // Only here for backwards compatibility. Assumes all inputs of the previous
  // transform are fed into this transform.
  absl::Status push_back(std::string name,
                         std::shared_ptr<TransformBase> transform) {
    std::vector<GraphEdge> input_edges;
    int input_size = transform->InputTypes().size();
    input_edges.reserve(input_size);
    if (empty()) {
      for (int i = 0; i < input_size; ++i) {
        ASSIGN_OR_RETURN(auto edge, GraphEdge::GraphInput(i));
        input_edges.push_back(std::move(edge));
      }
      SetNumInputs(input_size);
    } else {
      int prev_node_idx = size() - 1;
      for (int i = 0; i < input_size; ++i) {
        ASSIGN_OR_RETURN(auto edge, GraphEdge::NodeOutput(prev_node_idx, i));
        input_edges.push_back(std::move(edge));
      }
    }
    int node_idx = AddNode(std::move(name), transform, std::move(input_edges));

    std::vector<GraphEdge> outputs;
    int output_size = transform->OutputTypes().size();
    outputs.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
      ASSIGN_OR_RETURN(auto edge, GraphEdge::NodeOutput(node_idx, i));
      outputs.push_back(std::move(edge));
    }
    SetOutputs(std::move(outputs));

    return absl::OkStatus();
  }

  // Declares the number of input arguments to the execution graph. Unlike the
  // inferred count, it includes trailing inputs that no node or graph output
  // reads, so that the graph keeps the arity it was built with (e.g. a traced
  // function with an unused trailing argument). `GraphValidator` rejects graph
  // input edges whose index is not below the declared count.
  void SetNumInputs(size_t num_inputs) { num_inputs_ = num_inputs; }

  // Returns the declared number of inputs (from `SetNumInputs`, or from
  // `io_structure()` if `SetNumInputs` was not called), or `std::nullopt` if
  // neither was provided.
  std::optional<size_t> declared_num_inputs() const {
    if (num_inputs_.has_value()) {
      return num_inputs_;
    }
    if (io_structure_.input_structure().type() != PyTreeProto::INVALID) {
      return CountLeaves(io_structure_.input_structure()) +
             CountLeaves(io_structure_.captured_structure());
    }
    return std::nullopt;
  }

  // Returns the number of input arguments to the execution graph: the declared
  // count if any, otherwise the inferred one (see `GetNumReferencedInputs`).
  size_t GetNumInputs() const {
    if (std::optional<size_t> declared = declared_num_inputs();
        declared.has_value()) {
      return *declared;
    }
    // TODO: clean up the num_input inference path. num_inputs should be set
    // when building the graph. For now, we leave the inference path so that
    // affected users can switch from FATAL to ERROR here and move on.
    LOG(FATAL) << "ExecutionGraph::GetNumInputs() is inferring num_inputs "
                  "from graph edges because SetNumInputs() was not called.";
    return GetNumReferencedInputs();
  }

  // Returns one more than the largest graph input index read by a node or a
  // graph output, or 0 if no graph input is read.
  size_t GetNumReferencedInputs() const {
    int max_input_idx = -1;
    for (const auto& node : nodes_) {
      for (const auto& edge : node.input_edges) {
        if (edge.type == GraphEdge::SourceType::kGraphInput) {
          max_input_idx = std::max(max_input_idx, edge.source_index);
        }
      }
    }
    for (const auto& edge : outputs_) {
      if (edge.type == GraphEdge::SourceType::kGraphInput) {
        max_input_idx = std::max(max_input_idx, edge.source_index);
      }
    }
    return max_input_idx + 1;
  }

  // Returns the number of output arguments from the execution graph.
  size_t GetNumOutputs() const { return outputs_.size(); }

  size_t size() const { return nodes_.size(); }

  GraphNode& operator[](size_t index) { return nodes_[index]; }

  const GraphNode& operator[](size_t index) const { return nodes_[index]; }

  bool empty() const { return nodes_.empty(); }

  std::vector<GraphNode>::iterator begin() { return nodes_.begin(); }

  std::vector<GraphNode>::iterator end() { return nodes_.end(); }

  const GraphNode& back() const { return nodes_.back(); }

  std::vector<GraphNode>::const_iterator begin() const {
    return nodes_.begin();
  }

  std::vector<GraphNode>::const_iterator end() const { return nodes_.end(); }

  const std::vector<GraphNode>& nodes() const { return nodes_; }

  const std::vector<GraphEdge>& outputs() const { return outputs_; }

  const GraphIOStructureProto& io_structure() const { return io_structure_; }

  void clear() {
    nodes_.clear();
    outputs_.clear();
    num_inputs_.reset();
    io_structure_.Clear();
  }

 private:
  static size_t CountLeaves(const PyTreeProto& tree) {
    if (tree.type() == PyTreeProto::LEAF) {
      return 1;
    }
    size_t count = 0;
    for (const auto& child : tree.children()) {
      count += CountLeaves(child);
    }
    return count;
  }

  std::vector<GraphNode> nodes_;
  std::vector<GraphEdge> outputs_;
  // The declared number of graph inputs, if any. See `SetNumInputs`.
  std::optional<size_t> num_inputs_;
  GraphIOStructureProto io_structure_;
};

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_

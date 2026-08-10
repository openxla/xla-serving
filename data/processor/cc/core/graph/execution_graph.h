#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune

namespace neptune::data {
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

  void SetIOStructure(GraphIOStructureProto s) { io_structure_ = std::move(s); }

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

  // Returns the number of input arguments to the execution graph.
  size_t GetNumInputs() const {
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
  }

 private:
  std::vector<GraphNode> nodes_;
  std::vector<GraphEdge> outputs_;
  GraphIOStructureProto io_structure_;
};

}  // namespace core
}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_EXECUTION_GRAPH_H_

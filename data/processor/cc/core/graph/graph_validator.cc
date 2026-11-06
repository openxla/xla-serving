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

#include <cstddef>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

absl::Status GraphValidator::Validate(const ExecutionGraph& graph) {
  if (graph.empty()) {
    return absl::InvalidArgumentError("Unexpected empty graph");
  }
  RETURN_IF_ERROR(ValidateInternalEdges(graph));
  RETURN_IF_ERROR(ValidateGraphOutputs(graph));
  RETURN_IF_ERROR(ValidateGraphInputs(graph));
  return absl::OkStatus();
}

absl::Status GraphValidator::ValidateGraphInputs(const ExecutionGraph& graph) {
  if (!graph.declared_num_inputs().has_value()) {
    // The input count is inferred from the edges, so they are all in range.
    return absl::OkStatus();
  }
  const size_t num_inputs = *graph.declared_num_inputs();
  for (size_t i = 0; i < graph.size(); ++i) {
    const auto& node = graph[i];
    for (size_t j = 0; j < node.input_edges.size(); ++j) {
      const auto& edge = node.input_edges[j];
      if (edge.type == GraphEdge::SourceType::kGraphInput &&
          static_cast<size_t>(edge.source_index) >= num_inputs) {
        return absl::InvalidArgumentError(
            absl::StrCat("Node '", node.name, "' (index ", i, ") input port ",
                         j, " reads graph input ", edge.source_index,
                         ", but the graph has ", num_inputs, " inputs."));
      }
    }
  }
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    const auto& edge = graph.outputs()[i];
    if (edge.type == GraphEdge::SourceType::kGraphInput &&
        static_cast<size_t>(edge.source_index) >= num_inputs) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Graph output ", i, " reads graph input ", edge.source_index,
          ", but the graph has ", num_inputs, " inputs."));
    }
  }
  return absl::OkStatus();
}

absl::Status GraphValidator::ValidateInternalEdges(
    const ExecutionGraph& graph) {
  for (size_t i = 0; i < graph.size(); ++i) {
    const auto& node = graph[i];
    auto input_types = node.transform->InputTypes();
    if (node.input_edges.size() != input_types.size()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Node '", node.name, "' (index ", i, ") expects ", input_types.size(),
          " inputs, but has ", node.input_edges.size(), " input edges."));
    }

    for (size_t j = 0; j < node.input_edges.size(); ++j) {
      const auto& edge = node.input_edges[j];
      if (edge.type == GraphEdge::SourceType::kNodeOutput) {
        if (edge.source_index >= static_cast<int>(i) || edge.source_index < 0) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Graph is not topologically sorted or has invalid edge: node '",
              node.name, "' (index ", i, ") depends on node index ",
              edge.source_index, " which is not a valid previous node."));
        }
        const auto& producer_node = graph[edge.source_index];
        auto producer_outputs = producer_node.transform->OutputTypes();
        if (edge.port_index >= producer_outputs.size() || edge.port_index < 0) {
          return absl::InvalidArgumentError(
              absl::StrCat("Invalid port index: node '", node.name, "' (index ",
                           i, ") depends on output port ", edge.port_index,
                           " of node '", producer_node.name, "' (index ",
                           edge.source_index, "), but producer has only ",
                           producer_outputs.size(), " outputs."));
        }
        // Validate type compatibility
        if (input_types[j] != producer_outputs[edge.port_index]) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Type mismatch: node '", node.name, "' (index ", i,
              ") input port ", j, " expects ", input_types[j].DebugTypeName(),
              ", but connects to output port ", edge.port_index, " of node '",
              producer_node.name, "' (index ", edge.source_index,
              ") which outputs ",
              producer_outputs[edge.port_index].DebugTypeName(), "."));
        }
      }
    }
  }
  return absl::OkStatus();
}

absl::Status GraphValidator::ValidateGraphOutputs(const ExecutionGraph& graph) {
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    const auto& edge = graph.outputs()[i];
    if (edge.type == GraphEdge::SourceType::kNodeOutput) {
      if (edge.source_index >= static_cast<int>(graph.size()) ||
          edge.source_index < 0) {
        return absl::InvalidArgumentError(
            absl::StrCat("Graph output ", i, " references invalid node index ",
                         edge.source_index));
      }
      const auto& producer_node = graph[edge.source_index];
      auto producer_outputs = producer_node.transform->OutputTypes();
      if (edge.port_index >= producer_outputs.size() || edge.port_index < 0) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Graph output ", i, " references invalid port index ",
            edge.port_index, " of node '", producer_node.name, "'"));
      }
    }
  }
  return absl::OkStatus();
}

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

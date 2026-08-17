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

#include "data/processor/cc/core/graph/graph_executor.h"  // from @neptune

#include <cstddef>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune::data {
namespace composite {
namespace core {
namespace {

inline void LogValuesForDebug(absl::string_view prefix,
                              absl::Span<const api::Value> values) {
  if (VLOG_IS_ON(4)) {
    for (int i = 0; i < values.size(); ++i) {
      VLOG(4) << prefix << " " << i << ": " << values[i];
    }
  }
}

}  // namespace

absl::Status SingleThreadedSynchronousGraphExecutor::Execute(
    const ExecutionGraph& graph, api::ExecuteContext& context) {
  LogValuesForDebug("Initial input", context.inputs);

  // Store outputs for each node.
  // node_outputs[node_index][port_index]
  std::vector<absl::InlinedVector<api::Value, 4>> node_outputs(graph.size());

  for (const auto& node : graph.nodes()) {
    // Gather inputs for this node
    std::vector<api::Value> node_inputs;
    node_inputs.reserve(node.input_edges.size());
    {
      tsl::profiler::TraceMe trace(
          absl::string_view(node.collect_node_inputs_trace_name));
      for (const auto& edge : node.input_edges) {
        switch (edge.type) {
          case GraphEdge::SourceType::kGraphInput:
            if (edge.source_index >= context.inputs.size() ||
                edge.source_index < 0) {
              return absl::InternalError(
                  absl::StrCat("Execute error: Node '", node.name,
                               "' references out-of-bound graph input index ",
                               edge.source_index, " (graph has ",
                               context.inputs.size(), " inputs)"));
            }
            node_inputs.push_back(context.inputs[edge.source_index]);
            break;
          case GraphEdge::SourceType::kNodeOutput:
            if (edge.source_index >= node_outputs.size() ||
                edge.source_index < 0) {
              return absl::InternalError(absl::StrCat(
                  "Execute error: Node '", node.name,
                  "' references invalid node index ", edge.source_index));
            }
            if (edge.port_index >= node_outputs[edge.source_index].size() ||
                edge.port_index < 0) {
              return absl::InternalError(absl::StrCat(
                  "Execute error: Node '", node.name,
                  "' references invalid output port ", edge.port_index,
                  " of node index ", edge.source_index));
            }
            node_inputs.push_back(
                node_outputs[edge.source_index][edge.port_index]);
            break;
          case GraphEdge::SourceType::kConstant:
            node_inputs.push_back(edge.constant_value);
            break;
        }
      }
    }

    // Execute the transform
    api::ExecuteContext node_context = {
        .metadata = context.metadata,
        .inputs = absl::MakeConstSpan(node_inputs),
    };
    node_context.outputs.reserve(node.transform->OutputTypes().size());

    VLOG(2) << "Executing node " << node.node_index << ": " << node.name;
    absl::Status status;
    {
      tsl::profiler::TraceMe trace(absl::string_view(node.execute_trace_name));
      status = node.transform->Execute(node_context);
    }
    if (!status.ok()) {
      return absl::InternalError(
          absl::StrCat("Error executing node (index ", node.node_index,
                       "): ", node.name, " with error code ", status.code(),
                       " and error message: ", status.message()));
    }
    LogValuesForDebug(absl::StrCat("Node ", node.name, " output"),
                      node_context.outputs);

    // Save outputs
    node_outputs[node.node_index] = std::move(node_context.outputs);
  }

  // Construct graph outputs
  absl::InlinedVector<api::Value, 4> graph_outputs;
  graph_outputs.reserve(graph.outputs().size());
  for (size_t i = 0; i < graph.outputs().size(); ++i) {
    const auto& edge = graph.outputs()[i];
    switch (edge.type) {
      case GraphEdge::SourceType::kGraphInput:
        if (edge.source_index >= context.inputs.size() ||
            edge.source_index < 0) {
          return absl::InternalError(
              "Graph output references invalid graph input");
        }
        graph_outputs.push_back(context.inputs[edge.source_index]);
        break;
      case GraphEdge::SourceType::kNodeOutput:
        if (edge.source_index >= node_outputs.size() || edge.source_index < 0) {
          return absl::InternalError("Graph output references invalid node");
        }
        if (edge.port_index >= node_outputs[edge.source_index].size() ||
            edge.port_index < 0) {
          return absl::InternalError(
              "Graph output references invalid node output port");
        }
        graph_outputs.push_back(
            node_outputs[edge.source_index][edge.port_index]);
        break;
      case GraphEdge::SourceType::kConstant:
        graph_outputs.push_back(edge.constant_value);
        break;
    }
  }

  context.outputs = std::move(graph_outputs);
  return absl::OkStatus();
}

}  // namespace core
}  // namespace composite
}  // namespace neptune::data

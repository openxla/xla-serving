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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

// Validates the structure and type compatibility of an ExecutionGraph (DAG).
class GraphValidator {
 public:
  // Validates that the graph is non-empty, topologically sorted, all internal
  // edges connect compatible ports and types, and all graph output edges refer
  // to valid nodes/ports.
  static absl::Status Validate(const ExecutionGraph& graph);

 private:
  static absl::Status ValidateInternalEdges(const ExecutionGraph& graph);
  static absl::Status ValidateGraphOutputs(const ExecutionGraph& graph);
};

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_

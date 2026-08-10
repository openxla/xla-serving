#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune

namespace neptune::data {
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
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_VALIDATOR_H_

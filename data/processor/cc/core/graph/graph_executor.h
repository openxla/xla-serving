#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune

namespace neptune::data {
namespace composite {
namespace core {

// Abstract interface for executing an ExecutionGraph.
class GraphExecutor {
 public:
  virtual ~GraphExecutor() = default;

  // Executes the given execution graph with the provided execution context.
  virtual absl::Status Execute(const ExecutionGraph& graph,
                               api::ExecuteContext& context) = 0;
};

// Single-threaded synchronous implementation of GraphExecutor.
class SingleThreadedSynchronousGraphExecutor : public GraphExecutor {
 public:
  absl::Status Execute(const ExecutionGraph& graph,
                       api::ExecuteContext& context) override;
};

}  // namespace core
}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_

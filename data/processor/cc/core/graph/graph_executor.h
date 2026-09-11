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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_

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

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_GRAPH_GRAPH_EXECUTOR_H_

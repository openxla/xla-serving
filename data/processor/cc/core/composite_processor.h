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

#ifndef XLA_SERVING_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_
#define XLA_SERVING_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_

#include <cstddef>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving
#include "data/processor/cc/core/graph/graph_executor.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_manager.h"  // from @xla_serving

namespace xla_serving::data {
namespace composite {
namespace core {

// Processes a batch of data through a composite transform graph.
class CompositeProcessor {
 public:
  // Creates a processor for the given execution graph.
  static absl::StatusOr<CompositeProcessor> Create(
      ExecutionGraph graph,
      std::unique_ptr<GraphExecutor> executor = nullptr);

  // Processes a batch of data through the transform graph.
  //
  // Args:
  //   context: The execute context for the graph.
  //
  // Returns:
  //   OK if the graph was processed successfully.
  absl::Status Process(api::ExecuteContext& context);

  // Propagates shape and dtype signature from inputs to outputs through the
  // graph.
  absl::Status PropagateSignature(api::ShapeInferenceContext& context) const;

  // Returns the number of input arguments to the processor.
  size_t GetNumInputs() const { return graph_.GetNumInputs(); }

  // Returns the number of output arguments from the processor.
  size_t GetNumOutputs() const { return graph_.GetNumOutputs(); }

  // Serializes the composite processor to a configuration proto.
  absl::StatusOr<CompositeProcessorConfig> ToProto() const;

  // Creates a composite processor from a configuration proto.
  static absl::StatusOr<CompositeProcessor> FromProto(
      const CompositeProcessorConfig& config,
      std::unique_ptr<GraphExecutor> executor = nullptr);

  const xla_serving::data::ResourceManager& resource_manager() const;
  GraphExecutor& executor() { return *executor_; }
  const GraphExecutor& executor() const { return *executor_; }

  const ExecutionGraph& graph() const { return graph_; }

  // Exposes mutable ExecutionGraph for internal or test use.
  ExecutionGraph& mutable_graph() { return graph_; }

 private:
  explicit CompositeProcessor(
      ExecutionGraph graph,
      std::unique_ptr<xla_serving::data::ResourceManager> resource_manager =
          nullptr,
      std::unique_ptr<GraphExecutor> executor = nullptr);

  ExecutionGraph graph_;
  std::unique_ptr<xla_serving::data::ResourceManager> resource_manager_;
  std::unique_ptr<GraphExecutor> executor_;
};

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

#endif  // XLA_SERVING_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_

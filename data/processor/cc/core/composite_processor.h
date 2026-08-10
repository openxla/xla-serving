#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_

#include <cstddef>
#include <memory>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune
#include "data/processor/cc/core/graph/graph_executor.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource_manager.h"  // from @neptune

namespace neptune::data {
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

  const neptune::data::ResourceManager& resource_manager() const;
  GraphExecutor& executor() { return *executor_; }
  const GraphExecutor& executor() const { return *executor_; }

  const ExecutionGraph& graph() const { return graph_; }

  // Exposes mutable ExecutionGraph for internal or test use.
  ExecutionGraph& mutable_graph() { return graph_; }

 private:
  explicit CompositeProcessor(
      ExecutionGraph graph,
      std::unique_ptr<neptune::data::ResourceManager> resource_manager =
          nullptr,
      std::unique_ptr<GraphExecutor> executor = nullptr);

  ExecutionGraph graph_;
  std::unique_ptr<neptune::data::ResourceManager> resource_manager_;
  std::unique_ptr<GraphExecutor> executor_;
};

}  // namespace core
}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_CORE_COMPOSITE_PROCESSOR_H_

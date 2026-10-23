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

#include "data/processor/cc/core/composite_processor.h"  // from @xla_serving

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/shape.h"  // from @xla_serving
#include "data/processor/cc/api/transform_context.h"  // from @xla_serving
#include "data/processor/cc/api/value.h"  // from @xla_serving
#include "data/processor/cc/core/graph/execution_graph.h"  // from @xla_serving
#include "data/processor/cc/core/graph/graph_executor.h"  // from @xla_serving
#include "data/processor/cc/core/graph/graph_validator.h"  // from @xla_serving
#include "data/processor/cc/core/transform.h"  // from @xla_serving
#include "data/processor/cc/core/transform_registry.h"  // from @xla_serving
#include "data/processor/proto/transform_config.pb.h"  // from @xla_serving
#include "data/resource_manager/cc/resource.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_manager.h"  // from @xla_serving
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving::data {
namespace composite {
namespace core {
namespace {

using ResourceManagerPtr = const ::xla_serving::data::ResourceManager* absl_nonnull;

constexpr int kCompositeProcessorConfigVersion = 1;

api::ShapeInferenceValue GetSpecFromValue(const api::Value& val) {
  if (val.IsType<xla_serving::HostBuffer>()) {
    const auto& hb = val.Get<xla_serving::HostBuffer>();
    return api::HostBufferSpec{hb.dtype(), hb.shape()};
  } else if (val.IsType<std::vector<xla_serving::HostBuffer>>()) {
    const auto& vec = val.Get<std::vector<xla_serving::HostBuffer>>();
    api::ShapeInferenceValue::SpecList list;
    list.reserve(vec.size());
    for (const auto& hb : vec) {
      list.push_back(api::HostBufferSpec{hb.dtype(), hb.shape()});
    }
    return list;
  } else if (val.IsType<int>()) {
    return api::HostBufferSpec{xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                               xla::ifrt::Shape({})};
  } else if (val.IsType<float>()) {
    return api::HostBufferSpec{xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                               xla::ifrt::Shape({})};
  } else if (val.IsType<std::string>()) {
    return api::HostBufferSpec{
        xla::ifrt::DType(xla::ifrt::DType::Kind::kString),
        xla::ifrt::Shape({})};
  }
  return api::HostBufferSpec{};
}

// Returns true iff the elements of `hb`, each `element_size` bytes, are laid
// out densely in row-major order. REQUIRES: `hb` has few elements, so that no
// stride overflows.
bool IsDenseRowMajor(const xla_serving::HostBuffer& hb, int64_t element_size) {
  const auto& byte_strides = hb.byte_strides();
  if (!byte_strides.has_value()) return true;
  absl::Span<const int64_t> dims = hb.shape().dims();
  if (byte_strides->size() != dims.size()) return false;
  int64_t expected_stride = element_size;
  for (int64_t d = static_cast<int64_t>(dims.size()) - 1; d >= 0; --d) {
    // The stride of a dimension of size 0 or 1 is never used.
    if (dims[d] > 1 && (*byte_strides)[d] != expected_stride) return false;
    expected_stride *= dims[d];
  }
  return true;
}

// Returns the trace-time value of a constant, matching the spec that
// GetSpecFromValue gives it: the elements of a small int32 or int64 HostBuffer,
// or of an int (an int32 scalar). Returns nullopt for anything else.
std::optional<api::PartialValue> GetValueFromConstant(const api::Value& val) {
  if (val.IsType<int>()) {
    return api::PartialValue::Known({val.Get<int>()});
  }
  if (!val.IsType<xla_serving::HostBuffer>()) return std::nullopt;
  const auto& hb = val.Get<xla_serving::HostBuffer>();
  const xla::ifrt::DType::Kind kind = hb.dtype().kind();
  if (kind != xla::ifrt::DType::kS32 && kind != xla::ifrt::DType::kS64) {
    return std::nullopt;
  }
  const int64_t num_elements = hb.shape().num_elements();
  if (num_elements > api::PartialValue::kMaxElements) return std::nullopt;
  if (num_elements == 0) return api::PartialValue::Known({});
  // unaligned_flat() reads a dense row-major buffer only.
  if (!IsDenseRowMajor(hb, kind == xla::ifrt::DType::kS32 ? sizeof(int32_t)
                                                          : sizeof(int64_t))) {
    return std::nullopt;
  }
  std::vector<int64_t> elements;
  elements.reserve(num_elements);
  if (kind == xla::ifrt::DType::kS32) {
    const auto flat = hb.unaligned_flat<int32_t>();
    for (int64_t i = 0; i < num_elements; ++i) elements.push_back(flat(i));
  } else {
    const auto flat = hb.unaligned_flat<int64_t>();
    for (int64_t i = 0; i < num_elements; ++i) elements.push_back(flat(i));
  }
  return api::PartialValue::Known(elements);
}

absl::Status SerializeEdges(const GraphEdge& edge, EdgeProto& edge_proto) {
  switch (edge.type) {
    case GraphEdge::SourceType::kGraphInput:
      edge_proto.set_type(EdgeProto::GRAPH_INPUT);
      edge_proto.set_source_index(edge.source_index);
      break;
    case GraphEdge::SourceType::kNodeOutput:
      edge_proto.set_type(EdgeProto::NODE_OUTPUT);
      edge_proto.set_source_index(edge.source_index);
      edge_proto.set_port_index(edge.port_index);
      break;
    case GraphEdge::SourceType::kConstant:
      edge_proto.set_type(EdgeProto::CONSTANT);
      if (edge.constant_value.IsType<google::protobuf::Any>()) {
        *edge_proto.mutable_constant_value() =
            edge.constant_value.Get<google::protobuf::Any>();
      } else {
        return absl::InvalidArgumentError(
            "Serializing non-Any constants in ExecutionGraph is not "
            "supported yet.");
      }
      break;
  }
  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<CompositeProcessor> CompositeProcessor::Create(
    ExecutionGraph graph, std::unique_ptr<GraphExecutor> executor) {
  RETURN_IF_ERROR(GraphValidator::Validate(graph));

  auto resource_manager = std::make_unique<xla_serving::data::ResourceManager>();
  for (const auto& node : graph.nodes()) {
    for (auto& resource : node.transform->GetAllResources()) {
      RETURN_IF_ERROR(
          resource_manager->AddResourceIgnoringDuplicates(std::move(resource)));
    }
  }

  return CompositeProcessor{std::move(graph), std::move(resource_manager),
                            std::move(executor)};
}

absl::Status CompositeProcessor::Process(api::ExecuteContext& context) {
  return executor_->Execute(graph_, context);
}

absl::StatusOr<CompositeProcessorConfig> CompositeProcessor::ToProto() const {
  CompositeProcessorConfig config;
  config.mutable_metadata()->set_version(kCompositeProcessorConfigVersion);

  if (graph_.io_structure().has_input_structure() ||
      graph_.io_structure().has_output_structure() ||
      graph_.io_structure().has_captured_structure()) {
    *config.mutable_io_structure() = graph_.io_structure();
  }

  for (const auto& node : graph_.nodes()) {
    absl::StatusOr<TransformConfig> transform_config =
        node.transform->ToProto();
    if (!transform_config.ok()) {
      return transform_config.status();
    }
    transform_config->set_name(node.name);

    // Serialize input edges
    for (const auto& edge : node.input_edges) {
      auto* edge_proto = transform_config->add_input_edges();
      RETURN_IF_ERROR(SerializeEdges(edge, *edge_proto));
    }

    for (const auto& resource : node.transform->GetAllResources()) {
      // Validate that the current state of resources match that of the resource
      // manager.
      auto status = resource_manager_->GetResource(resource->id()).status();
      if (!status.ok()) {
        if (absl::IsNotFound(status)) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Resource not found with ID: ", resource->id(),
              " for transform: ", transform_config->name(),
              ". Transforms cannot alter their "
              "resources once a CompositeProcessor has been created."));
        }
        return status;
      }
    }

    *config.add_transforms() = *std::move(transform_config);
  }

  // Serialize graph outputs
  for (const auto& edge : graph_.outputs()) {
    auto* edge_proto = config.add_output_edges();
    RETURN_IF_ERROR(SerializeEdges(edge, *edge_proto));
  }

  for (const auto& resource : resource_manager_->GetAllResources()) {
    *config.add_resources() = resource->ToProto();
  }

  return config;
}

namespace {

absl::StatusOr<std::unique_ptr<TransformBase>> DeserializeTransform(
    const TransformConfig& transform_config,
    const xla_serving::data::ResourceManager& resource_manager) {
  std::vector<std::shared_ptr<Resource>> transform_resources;
  transform_resources.reserve(transform_config.resource_ids_size());
  for (uint64_t id : transform_config.resource_ids()) {
    auto resource = resource_manager.GetResource(id);
    if (!resource.ok()) {
      return absl::Status(
          resource.status().code(),
          absl::StrCat("Resource not found with ID: ", id,
                       " for transform: ", transform_config.name(), ". ",
                       resource.status().message()));
    }
    transform_resources.push_back(std::move(*resource));
  }
  return core::TransformRegistry::Instance().DeserializeTransform(
      transform_config, transform_resources);
}

// Deserialize V0 is used for older graphs that were serialized when
// CompositeProcessor only supported linear execution.
absl::StatusOr<ExecutionGraph> DeserializeV0(
    const CompositeProcessorConfig& config,
    const xla_serving::data::ResourceManager& resource_manager) {
  ExecutionGraph graph;
  for (const auto& transform_config : config.transforms()) {
    ASSIGN_OR_RETURN(auto transform,
                     DeserializeTransform(transform_config, resource_manager));

    // Deserialize input edges.
    std::vector<GraphEdge> input_edges;
    int input_size = transform->InputTypes().size();
    input_edges.reserve(input_size);
    // Because DeserializeV0 is only meant for linear execution the first
    // transform's inputs are always the graph's inputs.
    if (graph.empty()) {
      for (int i = 0; i < input_size; ++i) {
        ASSIGN_OR_RETURN(auto edge, GraphEdge::GraphInput(i));
        input_edges.push_back(std::move(edge));
      }
    } else {
      int prev_node_idx = graph.size() - 1;
      for (int i = 0; i < input_size; ++i) {
        ASSIGN_OR_RETURN(auto edge, GraphEdge::NodeOutput(prev_node_idx, i));
        input_edges.push_back(std::move(edge));
      }
    }

    graph.AddNode(std::string(transform_config.name()), std::move(transform),
                  std::move(input_edges));
  }

  // Deserialize outputs.
  // Because DeserializeV0 is only meant for linear execution the last
  // transform's outputs are always the graph's outputs.
  std::vector<GraphEdge> outputs;
  if (!graph.empty()) {
    int last_node_idx = graph.size() - 1;
    int output_size = graph.nodes().back().transform->OutputTypes().size();
    outputs.reserve(output_size);
    for (int i = 0; i < output_size; ++i) {
      ASSIGN_OR_RETURN(auto edge, GraphEdge::NodeOutput(last_node_idx, i));
      outputs.push_back(std::move(edge));
    }
  }
  graph.SetOutputs(std::move(outputs));

  if (config.has_io_structure()) {
    graph.SetIOStructure(config.io_structure());
  }

  return graph;
}

absl::StatusOr<ExecutionGraph> DeserializeV1(
    const CompositeProcessorConfig& config,
    const xla_serving::data::ResourceManager& resource_manager) {
  ExecutionGraph graph;
  for (const auto& transform_config : config.transforms()) {
    ASSIGN_OR_RETURN(auto transform,
                     DeserializeTransform(transform_config, resource_manager));

    // Deserialize input edges.
    std::vector<GraphEdge> input_edges;
    input_edges.reserve(transform_config.input_edges_size());
    for (const auto& proto_edge : transform_config.input_edges()) {
      switch (proto_edge.type()) {
        case EdgeProto::GRAPH_INPUT: {
          ASSIGN_OR_RETURN(auto edge,
                           GraphEdge::GraphInput(proto_edge.source_index()));
          input_edges.push_back(std::move(edge));
          break;
        }
        case EdgeProto::NODE_OUTPUT: {
          ASSIGN_OR_RETURN(auto edge,
                           GraphEdge::NodeOutput(proto_edge.source_index(),
                                                 proto_edge.port_index()));
          input_edges.push_back(std::move(edge));
          break;
        }
        case EdgeProto::CONSTANT:
          input_edges.push_back(
              GraphEdge::Constant(api::Value(proto_edge.constant_value())));
          break;
        default:
          return absl::InternalError("Unknown input edge type in proto");
      }
    }

    graph.AddNode(std::string(transform_config.name()), std::move(transform),
                  std::move(input_edges));
  }

  // Deserialize outputs.
  std::vector<GraphEdge> outputs;
  outputs.reserve(config.output_edges_size());
  for (const auto& proto_edge : config.output_edges()) {
    switch (proto_edge.type()) {
      case EdgeProto::GRAPH_INPUT: {
        ASSIGN_OR_RETURN(auto edge,
                         GraphEdge::GraphInput(proto_edge.source_index()));
        outputs.push_back(std::move(edge));
        break;
      }
      case EdgeProto::NODE_OUTPUT: {
        ASSIGN_OR_RETURN(auto edge,
                         GraphEdge::NodeOutput(proto_edge.source_index(),
                                               proto_edge.port_index()));
        outputs.push_back(std::move(edge));
        break;
      }
      case EdgeProto::CONSTANT:
        outputs.push_back(
            GraphEdge::Constant(api::Value(proto_edge.constant_value())));
        break;
      default:
        return absl::InternalError("Unknown output edge type in proto");
    }
  }
  graph.SetOutputs(std::move(outputs));

  if (config.has_io_structure()) {
    graph.SetIOStructure(config.io_structure());
  }

  return graph;
}

}  // namespace

absl::StatusOr<CompositeProcessor> CompositeProcessor::FromProto(
    const CompositeProcessorConfig& config,
    std::unique_ptr<GraphExecutor> executor) {
  ResourceManagerProto rm_proto;
  for (const auto& r : config.resources()) {
    *rm_proto.add_resources() = r;
  }
  ASSIGN_OR_RETURN(auto resource_manager,
                   xla_serving::data::ResourceManager::FromProto(rm_proto));
  for (const auto& resource : resource_manager->GetAllResources()) {
    RETURN_IF_ERROR(resource->Load());
  }

  int graph_version = config.metadata().version();
  ASSIGN_OR_RETURN(ExecutionGraph graph,
                   graph_version == 0
                       ? DeserializeV0(config, *resource_manager)
                       : DeserializeV1(config, *resource_manager));

  if (!graph.empty()) {
    RETURN_IF_ERROR(GraphValidator::Validate(graph));
  }
  return CompositeProcessor(std::move(graph), std::move(resource_manager),
                            std::move(executor));
}

absl::Status CompositeProcessor::PropagateSignature(
    api::ShapeInferenceContext& context) const {
  if (!context.input_partial_values.empty() &&
      context.input_partial_values.size() != context.inputs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "PropagateSignature error: input_partial_values has ",
        context.input_partial_values.size(), " entries, but there are ",
        context.inputs.size(), " inputs"));
  }
  // The value of graph input `i`, or nullopt if nothing is known about it.
  auto graph_input_partial_value =
      [&](int64_t i) -> std::optional<api::PartialValue> {
    if (context.input_partial_values.empty()) return std::nullopt;
    return context.input_partial_values[i];
  };

  std::vector<absl::InlinedVector<api::ShapeInferenceValue, 4>> node_outputs(
      graph_.size());
  // Parallel to node_outputs. An inner vector is empty if the node set no
  // output values.
  std::vector<absl::InlinedVector<std::optional<api::PartialValue>, 4>>
      node_output_partial_values(graph_.size());
  // The value of output `port` of node `node`, or nullopt if nothing is known
  // about it. REQUIRES: `node` and `port` index node_outputs.
  auto node_output_partial_value =
      [&](int64_t node, int64_t port) -> std::optional<api::PartialValue> {
    if (node_output_partial_values[node].empty()) return std::nullopt;
    return node_output_partial_values[node][port];
  };

  for (const auto& node : graph_.nodes()) {
    std::vector<api::ShapeInferenceValue> node_inputs;
    std::vector<std::optional<api::PartialValue>> node_input_partial_values;
    node_inputs.reserve(node.input_edges.size());
    node_input_partial_values.reserve(node.input_edges.size());
    for (const auto& edge : node.input_edges) {
      switch (edge.type) {
        case GraphEdge::SourceType::kGraphInput:
          if (edge.source_index >= context.inputs.size() ||
              edge.source_index < 0) {
            return absl::InternalError(
                absl::StrCat("PropagateSignature error: Node '", node.name,
                             "' references out-of-bound graph input index ",
                             edge.source_index, " (graph has ",
                             context.inputs.size(), " inputs)"));
          }
          node_inputs.push_back(context.inputs[edge.source_index]);
          node_input_partial_values.push_back(
              graph_input_partial_value(edge.source_index));
          break;
        case GraphEdge::SourceType::kNodeOutput:
          if (edge.source_index >= node_outputs.size() ||
              edge.source_index < 0) {
            return absl::InternalError(absl::StrCat(
                "PropagateSignature error: Node '", node.name,
                "' references invalid node index ", edge.source_index));
          }
          if (edge.port_index >= node_outputs[edge.source_index].size() ||
              edge.port_index < 0) {
            return absl::InternalError(absl::StrCat(
                "PropagateSignature error: Node '", node.name,
                "' references invalid output port ", edge.port_index,
                " of node index ", edge.source_index));
          }
          node_inputs.push_back(
              node_outputs[edge.source_index][edge.port_index]);
          node_input_partial_values.push_back(
              node_output_partial_value(edge.source_index, edge.port_index));
          break;
        case GraphEdge::SourceType::kConstant:
          node_inputs.push_back(GetSpecFromValue(edge.constant_value));
          node_input_partial_values.push_back(
              GetValueFromConstant(edge.constant_value));
          break;
      }
    }

    api::ShapeInferenceContext node_context = {
        .metadata = context.metadata,
        .inputs = absl::MakeConstSpan(node_inputs),
        .input_partial_values = absl::MakeConstSpan(node_input_partial_values),
    };
    node_context.outputs.reserve(node.transform->OutputTypes().size());

    auto status = node.transform->PropagateSignature(node_context);
    if (!status.ok()) {
      return absl::Status(
          status.code(),
          absl::StrCat("Error propagating signatures at node '", node.name,
                       "' (index ", node.node_index, "): ", status.message()));
    }

    node_outputs[node.node_index] = std::move(node_context.outputs);
    node_output_partial_values[node.node_index] =
        std::move(node_context.output_partial_values);
  }

  absl::InlinedVector<api::ShapeInferenceValue, 4> graph_outputs;
  absl::InlinedVector<std::optional<api::PartialValue>, 4>
      graph_output_partial_values;
  graph_outputs.reserve(graph_.outputs().size());
  graph_output_partial_values.reserve(graph_.outputs().size());
  for (size_t i = 0; i < graph_.outputs().size(); ++i) {
    const auto& edge = graph_.outputs()[i];
    switch (edge.type) {
      case GraphEdge::SourceType::kGraphInput:
        if (edge.source_index >= context.inputs.size() ||
            edge.source_index < 0) {
          return absl::InternalError(
              "Graph output references invalid graph input");
        }
        graph_outputs.push_back(context.inputs[edge.source_index]);
        graph_output_partial_values.push_back(
            graph_input_partial_value(edge.source_index));
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
        graph_output_partial_values.push_back(
            node_output_partial_value(edge.source_index, edge.port_index));
        break;
      case GraphEdge::SourceType::kConstant:
        graph_outputs.push_back(GetSpecFromValue(edge.constant_value));
        graph_output_partial_values.push_back(
            GetValueFromConstant(edge.constant_value));
        break;
    }
  }

  context.outputs = std::move(graph_outputs);
  context.output_partial_values = std::move(graph_output_partial_values);
  return absl::OkStatus();
}

CompositeProcessor::CompositeProcessor(
    ExecutionGraph graph,
    std::unique_ptr<xla_serving::data::ResourceManager> resource_manager,
    std::unique_ptr<GraphExecutor> executor)
    : graph_(std::move(graph)),
      resource_manager_(std::move(resource_manager)),
      executor_(std::move(executor)) {
  if (resource_manager_ == nullptr) {
    resource_manager_ = std::make_unique<xla_serving::data::ResourceManager>();
  }
  if (executor_ == nullptr) {
    executor_ = std::make_unique<SingleThreadedSynchronousGraphExecutor>();
  }
}

const xla_serving::data::ResourceManager& CompositeProcessor::resource_manager()
    const {
  return *resource_manager_;
}

}  // namespace core
}  // namespace composite
}  // namespace xla_serving::data

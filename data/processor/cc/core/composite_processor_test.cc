#include "data/processor/cc/core/composite_processor.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "google/protobuf/wrappers.pb.h"
#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/graph/execution_graph.h"  // from @neptune
#include "data/processor/cc/core/graph/graph_executor.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/cc/core/transform_registry.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/processor/python/examples/add_host_buffer_transform.h"  // from @neptune
#include "data/processor/python/examples/add_int_transform.h"  // from @neptune
#include "data/processor/python/examples/add_two_transform.h"  // from @neptune
#include "data/processor/python/examples/dummy_resource_transform.h"  // from @neptune
#include "data/processor/python/examples/multiply_by_five_transform.h"  // from @neptune
#include "data/processor/python/examples/suffix_transform.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "data/resource_manager/cc/resource_manager.h"  // from @neptune
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @neptune
#include "data/test_utils/proto_matchers.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/host_buffer_matcher.h"  // from @neptune
#include "runtime/core/host_buffer_test_utils.h"  // from @neptune
#include "google/protobuf/text_format.h"
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune::data {
namespace composite {
namespace core {
namespace {

using ::absl_testing::StatusIs;
using ::neptune::data::test_utils::EqualsProto;
using ::google::protobuf::TextFormat;
using ::testing::HasSubstr;

template <uint64_t... ExpectedIds>
class ExpectedResourcesTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<ExpectedResourcesTransform>> Create() {
    return absl::WrapUnique(new ExpectedResourcesTransform());
  }

  absl::Status Execute(api::ExecuteContext& context) override {
    return absl::OkStatus();
  }

  absl::Span<const api::TypeInfo> InputTypes() const override { return {}; }
  absl::Span<const api::TypeInfo> OutputTypes() const override { return {}; }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("ExpectedResourcesTransform");
    std::vector<uint64_t> ids = {ExpectedIds...};
    for (uint64_t id : ids) {
      config.add_resource_ids(id);
    }
    return config;
  }

 private:
  explicit ExpectedResourcesTransform() = default;
};

template <uint64_t... ExpectedIds>
bool RegisterExpectedResourcesTransform(const char* name) {
  auto factory =
      [](absl::Span<const api::Value> args) -> std::unique_ptr<TransformBase> {
    return nullptr;
  };
  auto deserializer =
      [](const TransformConfig& config,
         absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
      -> absl::StatusOr<std::unique_ptr<TransformBase>> {
    std::vector<uint64_t> expected = {ExpectedIds...};
    EXPECT_EQ(resources.size(), expected.size());
    for (size_t i = 0; i < resources.size(); ++i) {
      EXPECT_EQ(resources[i]->id(), expected[i]);
    }
    return ExpectedResourcesTransform<ExpectedIds...>::Create();
  };

  CHECK_OK(TransformRegistry::Instance().RegisterTransformFactory(
      name, std::type_index(typeid(ExpectedResourcesTransform<ExpectedIds...>)),
      std::move(factory)));
  CHECK_OK(TransformRegistry::Instance().RegisterTransformDeserializer(
      name, std::type_index(typeid(ExpectedResourcesTransform<ExpectedIds...>)),
      std::move(deserializer)));
  return true;
}

bool registered_1_3 =
    RegisterExpectedResourcesTransform<1, 3>("ExpectedResourcesTransform_1_3");
bool registered_2 =
    RegisterExpectedResourcesTransform<2>("ExpectedResourcesTransform_2");

class UnpackIntAnyTransform : public TransformBase {
 public:
  UnpackIntAnyTransform() = default;
  static absl::StatusOr<std::unique_ptr<UnpackIntAnyTransform>> Create() {
    return std::make_unique<UnpackIntAnyTransform>();
  }

  int operator()(google::protobuf::Any any) const {
    google::protobuf::Int32Value int_val;
    CHECK(any.UnpackTo(&int_val));
    return int_val.value();
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("UnpackIntAnyTransform");
    return config;
  }
};

ND_REGISTER_TRANSFORM(UnpackIntAnyTransform, "UnpackIntAnyTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return UnpackIntAnyTransform::Create();
        });

class TestResource : public neptune::data::Resource {
 public:
  TestResource(uint64_t id, std::string name)
      : neptune::data::Resource(id, std::move(name), {}, "TestResource") {}

 protected:
  absl::Status LoadImpl() override { return absl::OkStatus(); }
};

class MockDynamicResourcesTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MockDynamicResourcesTransform>>
  Create() {
    return absl::WrapUnique(new MockDynamicResourcesTransform());
  }

  absl::Status Execute(api::ExecuteContext& context) override {
    return absl::OkStatus();
  }

  absl::Span<const api::TypeInfo> InputTypes() const override { return {}; }
  absl::Span<const api::TypeInfo> OutputTypes() const override { return {}; }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("MockDynamicResourcesTransform");
    for (const auto& resource : resources_) {
      config.add_resource_ids(resource->id());
    }
    return config;
  }

  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override {
    return resources_;
  }

  void AddResource(std::shared_ptr<neptune::data::Resource> resource) {
    resources_.push_back(resource);
  }

 private:
  explicit MockDynamicResourcesTransform() = default;
  std::vector<std::shared_ptr<neptune::data::Resource>> resources_;
};

TEST(CompositeProcessorTest,
     SerializeResourceAddedAfterCompositeProcessorFails) {
  ASSERT_OK_AND_ASSIGN(auto dynamic_transform,
                       MockDynamicResourcesTransform::Create());

  auto resource1 = std::make_shared<TestResource>(1, "res1");
  dynamic_transform->AddResource(resource1);

  std::shared_ptr<MockDynamicResourcesTransform> shared_transform =
      std::move(dynamic_transform);

  ExecutionGraph graph;
  ASSERT_OK(graph.push_back("dynamic_node", shared_transform));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  // Add a new resource AFTER the processor creation to simulate dynamic
  // resource accumulation.
  auto resource2 = std::make_shared<TestResource>(2, "res2");
  shared_transform->AddResource(resource2);

  EXPECT_THAT(processor.ToProto(),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeProcessorTest, CreateEmptyGraphFails) {
  EXPECT_THAT(CompositeProcessor::Create({}),
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Unexpected empty graph")));
}

TEST(CompositeProcessorTest, SimpleTransformsSucceed) {
  ExecutionGraph graph;
  int add_two = graph.AddNode("add_two", std::make_unique<AddTwoTransform>(),
                              {GraphEdge::GraphInput(0).value()});
  int mult_five =
      graph.AddNode("mult_five", std::make_unique<MultiplyByFiveTransform>(),
                    {GraphEdge::NodeOutput(add_two, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(mult_five, 0).value()});

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));
  EXPECT_EQ(processor.GetNumInputs(), 1);
  EXPECT_EQ(processor.GetNumOutputs(), 1);

  api::ExecuteContext context;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(10));
  context.inputs = inputs;
  EXPECT_OK(processor.Process(context));

  EXPECT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<int>());
  EXPECT_EQ(context.outputs[0].Get<int>(), (10 + 2) * 5);
}

TEST(CompositeProcessorTest, TypeMismatchFails) {
  ExecutionGraph graph;
  int add_two = graph.AddNode("add_two", std::make_unique<AddTwoTransform>(),
                              {GraphEdge::GraphInput(0).value()});
  // AddTwoTransform outputs int, AddHostBufferTransform expects HostBuffer
  graph.AddNode("add_host_buffer", std::make_unique<AddHostBufferTransform>(),
                {GraphEdge::NodeOutput(add_two, 0).value(),
                 GraphEdge::NodeOutput(add_two, 0).value()});

  auto processor = CompositeProcessor::Create(std::move(graph));
  EXPECT_THAT(processor,
              StatusIs(absl::StatusCode::kInvalidArgument,
                       HasSubstr("Type mismatch: node 'add_host_buffer'")));
}

TEST(CompositeProcessorTest, AddIntTransformSucceeds) {
  ExecutionGraph graph;
  int add = graph.AddNode(
      "add", std::make_unique<AddHostBufferTransform>(),
      {GraphEdge::GraphInput(0).value(), GraphEdge::GraphInput(1).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(add, 0).value()});

  auto processor = CompositeProcessor::Create(std::move(graph));
  ASSERT_OK(processor.status());

  const xla::ifrt::DType dtype(xla::ifrt::DType::Kind::kS32);
  const xla::ifrt::Shape shape({2});

  neptune::HostBuffer hb1 = neptune::AsHostBuffer<int>({1, 2}, shape);
  neptune::HostBuffer hb2 = neptune::AsHostBuffer<int>({10, 20}, shape);

  api::ExecuteContext context;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(hb1));
  inputs.push_back(api::Value(hb2));
  context.inputs = inputs;

  EXPECT_OK(processor->Process(context));

  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<neptune::HostBuffer>());
  neptune::HostBuffer out = context.outputs[0].Get<neptune::HostBuffer>();
  EXPECT_THAT(out, neptune::HostBufferEq(neptune::AsHostBuffer<int>({11, 22}, shape)));
}

TEST(CompositeProcessorTest, Serialize) {
  ExecutionGraph graph;
  int add = graph.AddNode("my_add_node", std::make_unique<AddTwoTransform>(),
                          {GraphEdge::GraphInput(0).value()});
  int mult =
      graph.AddNode("my_mult_node", std::make_unique<MultiplyByFiveTransform>(),
                    {GraphEdge::NodeOutput(add, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(mult, 0).value()});
  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  EXPECT_THAT(config, EqualsProto(R"pb(
                transforms {
                  name: "my_add_node"
                  op_name: "AddTwoTransform"
                  input_edges { type: GRAPH_INPUT }
                }
                transforms {
                  name: "my_mult_node"
                  op_name: "MultiplyByFiveTransform"
                  input_edges { type: NODE_OUTPUT }
                }
                output_edges { type: NODE_OUTPUT source_index: 1 }
                metadata { version: 1 }
              )pb"));
}

TEST(CompositeProcessorTest, Deserialize) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms { name: "my_add_node" op_name: "AddTwoTransform" }
        transforms { name: "my_mult_node" op_name: "MultiplyByFiveTransform" }
      )pb",
      &config));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));

  api::ExecuteContext context;
  std::vector<api::Value> inputs = {api::Value(10)};
  context.inputs = inputs;
  EXPECT_OK(deserialized_processor.Process(context));

  EXPECT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<int>());
  EXPECT_EQ(context.outputs[0].Get<int>(), (10 + 2) * 5);
}

TEST(CompositeProcessorTest, SerializeWithResources) {
  ExecutionGraph graph;
  ASSERT_OK_AND_ASSIGN(auto dummy_resource, DummyResourceTransform::Create());
  graph.AddNode("my_dummy_node", std::move(dummy_resource), {});
  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  auto expected_proto = R"pb(
    transforms {
      name: "my_dummy_node"
      op_name: "DummyResourceTransform"
      resource_ids: 123
    }
    resources { id: 123 name: "dummy_resource" class_name: "DummyResource" }
    metadata { version: 1 }
  )pb";

  EXPECT_THAT(config, EqualsProto(expected_proto));

  // Validate that serializing the same processor again produces the same proto.
  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config2, processor.ToProto());
  EXPECT_THAT(config2, EqualsProto(expected_proto));
}

TEST(CompositeProcessorTest, DeserializeWithResources) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "my_dummy_node"
          op_name: "DummyResourceTransform"
          resource_ids: [ 123 ]
        }
        resources { name: "dummy_resource" class_name: "DummyResource" id: 123 }
      )pb",
      &config));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));
}

TEST(CompositeProcessorTest, FromProtoCallsLoadOnResources) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        resources { name: "dummy_resource1" class_name: "DummyResource" id: 1 }
        resources { name: "dummy_resource2" class_name: "DummyResource" id: 2 }
        resources { name: "dummy_resource3" class_name: "DummyResource" id: 3 }
      )pb",
      &config));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));
  const ResourceManager& rm = deserialized_processor.resource_manager();
  auto resources = rm.GetAllResources();
  ASSERT_EQ(resources.size(), 3);
  for (const auto& resource : resources) {
    ASSERT_TRUE(static_cast<const DummyResource*>(resource.get())->loaded());
  }
}

TEST(CompositeProcessorTest, FromProtoPassesOnlyRequestedResources) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "node1"
          op_name: "ExpectedResourcesTransform_1_3"
          resource_ids: [ 1, 3 ]
        }
        transforms {
          name: "node2"
          op_name: "ExpectedResourcesTransform_2"
          resource_ids: [ 2 ]
        }
        resources { name: "res1" class_name: "DummyResource" id: 1 }
        resources { name: "res2" class_name: "DummyResource" id: 2 }
        resources { name: "res3" class_name: "DummyResource" id: 3 }
      )pb",
      &config));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));
}

TEST(CompositeProcessorTest, FromProtoWithDuplicateResourcesReturnsError) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "node1"
          op_name: "DummyResourceTransform"
          resource_ids: [ 123 ]
        }
        transforms {
          name: "node2"
          op_name: "DummyResourceTransform"
          resource_ids: [ 123 ]
        }
        resources { name: "res1" class_name: "DummyResource" id: 123 }
        resources { name: "res2" class_name: "DummyResource" id: 123 }
      )pb",
      &config));

  EXPECT_THAT(CompositeProcessor::FromProto(config),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(CompositeProcessorTest, FromProtoFailsIfResourceAssetPathHasTraversal) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "node1"
          op_name: "DummyResourceTransform"
          resource_ids: [ 123 ]
        }
        resources {
          name: "res1"
          class_name: "DummyResource"
          id: 123
          assets { name: "asset1" path: "foo/../bar" }
        }
      )pb",
      &config));

  EXPECT_THAT(CompositeProcessor::FromProto(config),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(CompositeProcessorTest, TransformWithDetails) {
  ExecutionGraph graph;
  int node =
      graph.AddNode("my_node", std::make_unique<SuffixTransform>("_configured"),
                    {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node, 0).value()});
  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  CompositeProcessorConfig expected_config;
  ASSERT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "my_node"
          op_name: "SuffixTransform"
          detail {
            [type.googleapis.com/google.protobuf.StringValue] {
              value: "_configured"
            }
          }
          input_edges { type: GRAPH_INPUT }
        }
        output_edges { type: NODE_OUTPUT }
        metadata { version: 1 }
      )pb",
      &expected_config));

  EXPECT_THAT(config, EqualsProto(expected_config));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));

  api::ExecuteContext context;
  std::vector<api::Value> input_values = {api::Value(std::string("hello"))};
  context.inputs = input_values;
  EXPECT_OK(deserialized_processor.Process(context));
  EXPECT_EQ(context.outputs[0].Get<std::string>(), "hello_configured");
}

TEST(CompositeProcessorTest, FewOuputsClearsAndReservesInlineSpace) {
  class FewOutputsTransform : public TransformBase {
   public:
    explicit FewOutputsTransform(int* count, int* capacity)
        : initial_output_vector_count_(count),
          initial_output_vector_capacity_(capacity) {}
    absl::Status Execute(api::ExecuteContext& context) override {
      *initial_output_vector_count_ = context.outputs.size();
      *initial_output_vector_capacity_ = context.outputs.capacity();
      context.outputs.push_back(api::Value(1));
      context.outputs.push_back(api::Value(2));
      return absl::OkStatus();
    }

    absl::Span<const api::TypeInfo> InputTypes() const override {
      static const auto* types =
          new std::vector<api::TypeInfo>{api::CreateTypeInfo<int>()};
      return *types;
    }

    absl::Span<const api::TypeInfo> OutputTypes() const override {
      static const auto* types = new std::vector<api::TypeInfo>{
          api::CreateTypeInfo<int>(), api::CreateTypeInfo<int>()};
      return *types;
    }

    absl::StatusOr<TransformConfig> ToProto() const override {
      TransformConfig config;
      config.set_op_name("few_outputs");
      return config;
    }

    int* initial_output_vector_count_ = nullptr;
    int* initial_output_vector_capacity_ = nullptr;
  };

  int initial_output_vector_count = 0;
  int initial_output_vector_capacity = 0;
  ExecutionGraph graph;
  int add_two = graph.AddNode("add_two", std::make_unique<AddTwoTransform>(),
                              {GraphEdge::GraphInput(0).value()});
  int few_outputs = graph.AddNode(
      "few_outputs",
      std::make_unique<FewOutputsTransform>(&initial_output_vector_count,
                                            &initial_output_vector_capacity),
      {GraphEdge::NodeOutput(add_two, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(few_outputs, 0).value(),
                    GraphEdge::NodeOutput(few_outputs, 1).value()});
  auto processor = CompositeProcessor::Create(std::move(graph));
  ASSERT_OK(processor.status());

  api::ExecuteContext context;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(10));
  context.inputs = inputs;
  EXPECT_OK(processor->Process(context));

  EXPECT_EQ(initial_output_vector_count, 0);
  // Vector remains with inline capacity size of 4.
  EXPECT_EQ(initial_output_vector_capacity, 4);
}

TEST(CompositeProcessorTest, ManyOuputsClearsAndReservesInlineSpace) {
  class ManyOutputsTransform : public TransformBase {
   public:
    explicit ManyOutputsTransform(int* count, int* capacity)
        : initial_output_vector_count_(count),
          initial_output_vector_capacity_(capacity) {}
    absl::Status Execute(api::ExecuteContext& context) override {
      *initial_output_vector_count_ = context.outputs.size();
      *initial_output_vector_capacity_ = context.outputs.capacity();
      for (int i = 0; i < 20; i++) {
        context.outputs.push_back(api::Value(i));
      }
      return absl::OkStatus();
    }

    absl::Span<const api::TypeInfo> InputTypes() const override {
      static const auto* types =
          new std::vector<api::TypeInfo>{api::CreateTypeInfo<int>()};
      return *types;
    }

    absl::Span<const api::TypeInfo> OutputTypes() const override {
      static auto* types = new std::vector<api::TypeInfo>();
      types->reserve(20);
      for (int i = 0; i < 20; i++) {
        types->push_back(api::CreateTypeInfo<int>());
      }
      return *types;
    }

    absl::StatusOr<TransformConfig> ToProto() const override {
      TransformConfig config;
      config.set_op_name("many_outputs");
      return config;
    }

    int* initial_output_vector_count_ = nullptr;
    int* initial_output_vector_capacity_ = nullptr;
  };

  int initial_output_vector_count = 0;
  int initial_output_vector_capacity = 0;
  ExecutionGraph graph;
  int add_two = graph.AddNode("add_two", std::make_unique<AddTwoTransform>(),
                              {GraphEdge::GraphInput(0).value()});
  int many_outputs = graph.AddNode(
      "many_outputs",
      std::make_unique<ManyOutputsTransform>(&initial_output_vector_count,
                                             &initial_output_vector_capacity),
      {GraphEdge::NodeOutput(add_two, 0).value()});
  std::vector<GraphEdge> outputs;
  outputs.reserve(20);
  for (int i = 0; i < 20; ++i) {
    outputs.push_back(GraphEdge::NodeOutput(many_outputs, i).value());
  }
  graph.SetOutputs(std::move(outputs));
  auto processor = CompositeProcessor::Create(std::move(graph));
  ASSERT_OK(processor.status());

  api::ExecuteContext context;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(10));
  context.inputs = inputs;
  EXPECT_OK(processor->Process(context));

  EXPECT_EQ(initial_output_vector_count, 0);
  // Vector allocates memory resulting in capacity of at least 20.
  EXPECT_GE(initial_output_vector_capacity, 20);
}

TEST(CompositeProcessorTest, FromProtoV0IgnoresEdges) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        transforms {
          name: "node1"
          op_name: "AddTwoTransform"
          input_edges { type: INVALID }
        }
      )pb",
      &config));

  EXPECT_OK(CompositeProcessor::FromProto(config));
}

TEST(CompositeProcessorTest, FromProtoV1ValidatesEdges) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        metadata { version: 1 }
        transforms {
          name: "node1"
          op_name: "AddTwoTransform"
          input_edges { type: INVALID }
        }
      )pb",
      &config));

  EXPECT_THAT(CompositeProcessor::FromProto(config),
              StatusIs(absl::StatusCode::kInternal,
                       HasSubstr("Unknown input edge type in proto")));
}

TEST(CompositeProcessorTest, FromProtoV1WithGraphInputSucceeds) {
  CompositeProcessorConfig config;
  EXPECT_TRUE(TextFormat::ParseFromString(
      R"pb(
        metadata { version: 1 }
        transforms {
          name: "node1"
          op_name: "AddTwoTransform"
          input_edges { type: GRAPH_INPUT source_index: 0 }
        }
        output_edges { type: NODE_OUTPUT source_index: 0 port_index: 0 }
      )pb",
      &config));

  EXPECT_OK(CompositeProcessor::FromProto(config));
}

TEST(CompositeProcessorTest, DagExecutionSucceeds) {
  ExecutionGraph graph;

  // Node 0: "add_two_1"
  int node0_idx =
      graph.AddNode("add_two_1", std::make_unique<AddTwoTransform>(),
                    {GraphEdge::GraphInput(0).value()});
  EXPECT_EQ(node0_idx, 0);

  // Node 1: "add_two_2"
  int node1_idx =
      graph.AddNode("add_two_2", std::make_unique<AddTwoTransform>(),
                    {GraphEdge::GraphInput(0).value()});
  EXPECT_EQ(node1_idx, 1);

  // Node 2: "add_int"
  int node2_idx = graph.AddNode("add_int", std::make_unique<AddIntTransform>(),
                                {GraphEdge::NodeOutput(0, 0).value(),
                                 GraphEdge::NodeOutput(1, 0).value()});
  EXPECT_EQ(node2_idx, 2);

  graph.SetOutputs({GraphEdge::NodeOutput(2, 0).value()});

  auto processor = CompositeProcessor::Create(std::move(graph));
  ASSERT_OK(processor.status());

  api::ExecuteContext context;
  std::vector<api::Value> inputs = {api::Value(10)};
  context.inputs = inputs;
  EXPECT_OK(processor->Process(context));

  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<int>());
  EXPECT_EQ(context.outputs[0].Get<int>(), 24);
}

TEST(CompositeProcessorTest, DagSerializeDeserialize) {
  ExecutionGraph graph;
  int node0_idx =
      graph.AddNode("add_two_1", std::make_unique<AddTwoTransform>(),
                    {GraphEdge::GraphInput(0).value()});
  int node1_idx =
      graph.AddNode("add_two_2", std::make_unique<AddTwoTransform>(),
                    {GraphEdge::GraphInput(0).value()});
  int node2_idx = graph.AddNode("add_int", std::make_unique<AddIntTransform>(),
                                {GraphEdge::NodeOutput(node0_idx, 0).value(),
                                 GraphEdge::NodeOutput(node1_idx, 0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node2_idx, 0).value()});

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  EXPECT_THAT(config, EqualsProto(R"pb(
                transforms {
                  name: "add_two_1"
                  op_name: "AddTwoTransform"
                  input_edges { type: GRAPH_INPUT }
                }
                transforms {
                  name: "add_two_2"
                  op_name: "AddTwoTransform"
                  input_edges { type: GRAPH_INPUT }
                }
                transforms {
                  name: "add_int"
                  op_name: "AddIntTransform"
                  input_edges { type: NODE_OUTPUT source_index: 0 }
                  input_edges { type: NODE_OUTPUT source_index: 1 }
                }
                output_edges { type: NODE_OUTPUT source_index: 2 }
                metadata { version: 1 }
              )pb"));

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized,
                       CompositeProcessor::FromProto(config));

  api::ExecuteContext context;
  std::vector<api::Value> inputs2 = {api::Value(10)};
  context.inputs = inputs2;
  EXPECT_OK(deserialized.Process(context));

  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<int>());
  EXPECT_EQ(context.outputs[0].Get<int>(), 24);
}

class CustomCountingGraphExecutor : public GraphExecutor {
 public:
  absl::Status Execute(const ExecutionGraph& graph,
                       api::ExecuteContext& context) override {
    execution_count++;
    return delegate_.Execute(graph, context);
  }

  int execution_count = 0;

 private:
  SingleThreadedSynchronousGraphExecutor delegate_;
};

TEST(CompositeProcessorTest, CustomGraphExecutorInjection) {
  ExecutionGraph graph;
  int add_two = graph.AddNode("add_two", std::make_unique<AddTwoTransform>(),
                              {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(add_two, 0).value()});

  auto custom_executor = std::make_unique<CustomCountingGraphExecutor>();
  auto* custom_executor_ptr = custom_executor.get();

  ASSERT_OK_AND_ASSIGN(
      CompositeProcessor processor,
      CompositeProcessor::Create(std::move(graph), std::move(custom_executor)));

  api::ExecuteContext context;
  std::vector<api::Value> inputs = {api::Value(10)};
  context.inputs = inputs;
  EXPECT_OK(processor.Process(context));
  EXPECT_EQ(custom_executor_ptr->execution_count, 1);
  EXPECT_EQ(context.outputs[0].Get<int>(), 12);
}

TEST(CompositeProcessorTest, ConstantEdgeSerializeDeserialize) {
  ExecutionGraph graph;

  google::protobuf::Int32Value int_val;
  int_val.set_value(42);
  google::protobuf::Any any_val;
  any_val.PackFrom(int_val);

  int unpack_node =
      graph.AddNode("unpack", std::make_unique<UnpackIntAnyTransform>(),
                    {GraphEdge::Constant(api::Value(any_val))});

  graph.SetOutputs({GraphEdge::NodeOutput(unpack_node, 0).value()});

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  // Run the processor
  api::ExecuteContext context;
  EXPECT_OK(processor.Process(context));
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_TRUE(context.outputs[0].IsType<int>());
  EXPECT_EQ(context.outputs[0].Get<int>(), 42);

  // Serialize
  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  // Deserialize
  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized,
                       CompositeProcessor::FromProto(config));

  // Run deserialized processor
  api::ExecuteContext context2;
  EXPECT_OK(deserialized.Process(context2));
  ASSERT_EQ(context2.outputs.size(), 1);
  EXPECT_TRUE(context2.outputs[0].IsType<int>());
  EXPECT_EQ(context2.outputs[0].Get<int>(), 42);
}

TEST(CompositeProcessorTest, IOStructureSerializeDeserialize) {
  ExecutionGraph graph;
  int add = graph.AddNode("my_add_node", std::make_unique<AddTwoTransform>(),
                          {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(add, 0).value()});

  GraphIOStructureProto io_structure;
  io_structure.mutable_input_structure()->mutable_leaf_value()->add_shape(42);
  io_structure.mutable_output_structure()->mutable_leaf_value()->add_shape(43);
  io_structure.mutable_captured_structure()->mutable_leaf_value()->add_shape(
      44);
  graph.SetIOStructure(io_structure);

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_processor,
                       CompositeProcessor::FromProto(config));

  GraphIOStructureProto deserialized_io =
      deserialized_processor.graph().io_structure();
  EXPECT_EQ(deserialized_io.input_structure().leaf_value().shape(0), 42);
  EXPECT_EQ(deserialized_io.output_structure().leaf_value().shape(0), 43);
  EXPECT_EQ(deserialized_io.captured_structure().leaf_value().shape(0), 44);
}

template <typename T>
class MultiKernelMathTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MultiKernelMathTransform<T>>> Create() {
    return std::make_unique<MultiKernelMathTransform<T>>();
  }

  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer in) {
    neptune::HostBuffer out(in.dtype(), in.shape());
    const T* in_data = static_cast<const T*>(in.base());
    T* out_data = static_cast<T*>(out.base());
    for (size_t i = 0; i < out.shape().num_elements(); ++i) {
      out_data[i] = in_data[i] + static_cast<T>(10);
    }
    return out;
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("MultiKernelMathTransform");
    return config;
  }

 protected:
  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override {
    if (context.inputs.size() != 1 || !context.inputs[0].IsSpec()) {
      return absl::InvalidArgumentError("Expected 1 tensor spec input");
    }
    context.outputs.push_back(context.inputs[0]);
    return absl::OkStatus();
  }
};

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<int32_t>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<neptune::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<int32_t>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kS32}}});

ND_REGISTER_TRANSFORM(MultiKernelMathTransform<float>,
                      "MultiKernelMathTransform")
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<neptune::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathTransform<float>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kF32}}});

TEST(CompositeProcessorTest, DynamicDispatchDeserializationAndExecution) {
  // Build a graph containing the multi-kernel transform.
  ExecutionGraph graph;
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      DynamicDispatchTransform::Create("MultiKernelMathTransform"));
  int node_idx = graph.AddNode("multi_math_node", std::move(transform),
                               {GraphEdge::GraphInput(0).value()});
  graph.SetOutputs({GraphEdge::NodeOutput(node_idx, 0).value()});

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  // Verify we can run the graph before serialization.
  {
    neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                       xla::ifrt::Shape({2}));
    int32_t* data = static_cast<int32_t*>(hb.base());
    data[0] = 1;
    data[1] = 2;

    std::vector<api::ShapeInferenceValue> sig_inputs = {
        api::HostBufferSpec{hb.dtype(), hb.shape()}};
    api::ShapeInferenceContext sig_ctx;
    sig_ctx.inputs = sig_inputs;
    ASSERT_OK(processor.PropagateSignature(sig_ctx));
    ASSERT_EQ(sig_ctx.outputs.size(), 1);
    EXPECT_EQ(sig_ctx.outputs[0].spec().dtype, hb.dtype());

    std::vector<api::Value> exec_inputs = {api::Value(hb)};
    api::ExecuteContext exec_ctx;
    exec_ctx.inputs = exec_inputs;
    ASSERT_OK(processor.Process(exec_ctx));
    ASSERT_EQ(exec_ctx.outputs.size(), 1);
    const auto& out_hb = exec_ctx.outputs[0].Get<neptune::HostBuffer>();
    const int32_t* out_data = static_cast<const int32_t*>(out_hb.base());
    EXPECT_EQ(out_data[0], 11);
    EXPECT_EQ(out_data[1], 12);
  }

  // Serialize the graph to a proto.
  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig config, processor.ToProto());

  // Deserialize from proto - should instantiate a DynamicDispatchTransform
  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_s32,
                       CompositeProcessor::FromProto(config));

  // Verify dispatch to int32 kernel.
  {
    neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                       xla::ifrt::Shape({2}));
    int32_t* data = static_cast<int32_t*>(hb.base());
    data[0] = 1;
    data[1] = 2;

    std::vector<api::ShapeInferenceValue> sig_inputs = {
        api::HostBufferSpec{hb.dtype(), hb.shape()}};
    api::ShapeInferenceContext sig_ctx;
    sig_ctx.inputs = sig_inputs;
    ASSERT_OK(deserialized_s32.PropagateSignature(sig_ctx));
    ASSERT_EQ(sig_ctx.outputs.size(), 1);
    EXPECT_EQ(sig_ctx.outputs[0].spec().dtype, hb.dtype());

    std::vector<api::Value> exec_inputs = {api::Value(hb)};
    api::ExecuteContext exec_ctx;
    exec_ctx.inputs = exec_inputs;
    ASSERT_OK(deserialized_s32.Process(exec_ctx));
    ASSERT_EQ(exec_ctx.outputs.size(), 1);
    const auto& out_hb = exec_ctx.outputs[0].Get<neptune::HostBuffer>();
    const int32_t* out_data = static_cast<const int32_t*>(out_hb.base());
    EXPECT_EQ(out_data[0], 11);
    EXPECT_EQ(out_data[1], 12);
  }

  // Verify dispatch to float32 kernel.
  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_f32,
                       CompositeProcessor::FromProto(config));
  {
    neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                       xla::ifrt::Shape({2}));
    float* data = static_cast<float*>(hb.base());
    data[0] = 1.5f;
    data[1] = 2.5f;

    std::vector<api::ShapeInferenceValue> sig_inputs = {
        api::HostBufferSpec{hb.dtype(), hb.shape()}};
    api::ShapeInferenceContext sig_ctx;
    sig_ctx.inputs = sig_inputs;
    ASSERT_OK(deserialized_f32.PropagateSignature(sig_ctx));
    ASSERT_EQ(sig_ctx.outputs.size(), 1);
    EXPECT_EQ(sig_ctx.outputs[0].spec().dtype, hb.dtype());

    std::vector<api::Value> exec_inputs = {api::Value(hb)};
    api::ExecuteContext exec_ctx;
    exec_ctx.inputs = exec_inputs;
    ASSERT_OK(deserialized_f32.Process(exec_ctx));
    ASSERT_EQ(exec_ctx.outputs.size(), 1);
    const auto& out_hb = exec_ctx.outputs[0].Get<neptune::HostBuffer>();
    const float* out_data = static_cast<const float*>(out_hb.base());
    EXPECT_EQ(out_data[0], 11.5f);
    EXPECT_EQ(out_data[1], 12.5f);
  }
}

template <typename T>
class MultiKernelMathGpuTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MultiKernelMathGpuTransform<T>>>
  Create() {
    return std::make_unique<MultiKernelMathGpuTransform<T>>();
  }

  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer in) {
    neptune::HostBuffer out(in.dtype(), in.shape());
    const T* in_data = static_cast<const T*>(in.base());
    T* out_data = static_cast<T*>(out.base());
    for (size_t i = 0; i < out.shape().num_elements(); ++i) {
      // GPU kernel adds 100 to differentiate from CPU kernel which adds 10.
      out_data[i] = in_data[i] + static_cast<T>(100);
    }
    return out;
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("MultiKernelMathTransform");
    return config;
  }

 protected:
  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override {
    if (context.inputs.size() != 1 || !context.inputs[0].IsSpec()) {
      return absl::InvalidArgumentError("Expected 1 tensor spec input");
    }
    context.outputs.push_back(context.inputs[0]);
    return absl::OkStatus();
  }
};

ND_REGISTER_TRANSFORM(MultiKernelMathGpuTransform<float>,
                      "MultiKernelMathTransform")
    .SetDevice(Device::kGpu)
    .SetDeserializer(
        [](const TransformConfig&,
           absl::Span<const std::shared_ptr<neptune::data::Resource>>)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return MultiKernelMathGpuTransform<float>::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kF32}}});

TEST(CompositeProcessorTest, GpuDynamicDispatchExecution) {
  ExecutionGraph graph;
  // Node 0: GPU dynamic dispatch transform
  ASSERT_OK_AND_ASSIGN(
      auto gpu_transform,
      DynamicDispatchTransform::Create("MultiKernelMathTransform",
                                       /*args=*/{}, Device::kGpu));
  int gpu_node = graph.AddNode("gpu_math_node", std::move(gpu_transform),
                               {GraphEdge::GraphInput(0).value()});

  // Node 1: CPU dynamic dispatch transform
  ASSERT_OK_AND_ASSIGN(
      auto cpu_transform,
      DynamicDispatchTransform::Create("MultiKernelMathTransform",
                                       /*args=*/{}, Device::kCpu));
  int cpu_node = graph.AddNode("cpu_math_node", std::move(cpu_transform),
                               {GraphEdge::GraphInput(0).value()});

  graph.SetOutputs({GraphEdge::NodeOutput(gpu_node, 0).value(),
                    GraphEdge::NodeOutput(cpu_node, 0).value()});

  ASSERT_OK_AND_ASSIGN(CompositeProcessor processor,
                       CompositeProcessor::Create(std::move(graph)));

  neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                     xla::ifrt::Shape({2}));
  float* data = static_cast<float*>(hb.base());
  data[0] = 1.5f;
  data[1] = 2.5f;

  // Propagate signature across both GPU and CPU nodes
  std::vector<api::ShapeInferenceValue> sig_inputs = {
      api::HostBufferSpec{hb.dtype(), hb.shape()}};
  api::ShapeInferenceContext sig_ctx;
  sig_ctx.inputs = sig_inputs;
  ASSERT_OK(processor.PropagateSignature(sig_ctx));
  ASSERT_EQ(sig_ctx.outputs.size(), 2);
  EXPECT_EQ(sig_ctx.outputs[0].spec().dtype, hb.dtype());
  EXPECT_EQ(sig_ctx.outputs[1].spec().dtype, hb.dtype());

  // Execute graph
  std::vector<api::Value> exec_inputs = {api::Value(hb)};
  api::ExecuteContext exec_ctx;
  exec_ctx.inputs = exec_inputs;
  ASSERT_OK(processor.Process(exec_ctx));
  ASSERT_EQ(exec_ctx.outputs.size(), 2);

  // GPU node output: input + 100
  const auto& gpu_out_hb = exec_ctx.outputs[0].Get<neptune::HostBuffer>();
  const float* gpu_out_data = static_cast<const float*>(gpu_out_hb.base());
  EXPECT_EQ(gpu_out_data[0], 101.5f);
  EXPECT_EQ(gpu_out_data[1], 102.5f);

  // CPU node output: input + 10
  const auto& cpu_out_hb = exec_ctx.outputs[1].Get<neptune::HostBuffer>();
  const float* cpu_out_data = static_cast<const float*>(cpu_out_hb.base());
  EXPECT_EQ(cpu_out_data[0], 11.5f);
  EXPECT_EQ(cpu_out_data[1], 12.5f);

  // Verify proto serialization and deserialization preserves GPU & CPU device
  // assignment automatically across serialization boundaries without
  // programmatic assignment.
  ASSERT_OK_AND_ASSIGN(CompositeProcessorConfig proto_config,
                       processor.ToProto());
  ASSERT_EQ(proto_config.transforms_size(), 2);
  EXPECT_EQ(proto_config.transforms(0).device(), DeviceProto::DEVICE_GPU);
  EXPECT_EQ(proto_config.transforms(1).device(), DeviceProto::DEVICE_CPU);

  // Deserialize without any programmatic device assignment
  ASSERT_OK_AND_ASSIGN(CompositeProcessor deserialized_proc,
                       CompositeProcessor::FromProto(proto_config));

  // Propagate signature on deserialized processor
  std::vector<api::ShapeInferenceValue> des_sig_inputs = {
      api::HostBufferSpec{hb.dtype(), hb.shape()}};
  api::ShapeInferenceContext des_sig_ctx;
  des_sig_ctx.inputs = des_sig_inputs;
  ASSERT_OK(deserialized_proc.PropagateSignature(des_sig_ctx));
  ASSERT_EQ(des_sig_ctx.outputs.size(), 2);

  // Execute deserialized processor
  std::vector<api::Value> des_exec_inputs = {api::Value(hb)};
  api::ExecuteContext des_exec_ctx;
  des_exec_ctx.inputs = des_exec_inputs;
  ASSERT_OK(deserialized_proc.Process(des_exec_ctx));
  ASSERT_EQ(des_exec_ctx.outputs.size(), 2);

  // Verify GPU node output from deserialized processor: input + 100
  const auto& des_gpu_out_hb = des_exec_ctx.outputs[0].Get<neptune::HostBuffer>();
  const float* des_gpu_out_data =
      static_cast<const float*>(des_gpu_out_hb.base());
  EXPECT_EQ(des_gpu_out_data[0], 101.5f);
  EXPECT_EQ(des_gpu_out_data[1], 102.5f);

  // Verify CPU node output from deserialized processor: input + 10
  const auto& des_cpu_out_hb = des_exec_ctx.outputs[1].Get<neptune::HostBuffer>();
  const float* des_cpu_out_data =
      static_cast<const float*>(des_cpu_out_hb.base());
  EXPECT_EQ(des_cpu_out_data[0], 11.5f);
  EXPECT_EQ(des_cpu_out_data[1], 12.5f);
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace neptune::data

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

#include "data/processor/cc/core/transform_registry.h"  // from @neptune

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/base/no_destructor.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/processor/cc/api/registry.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/api/value.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla

namespace neptune::data {
namespace composite {
namespace core {
namespace {

class TransformRegistryTest : public ::testing::Test {
 protected:
  TransformRegistryTest() : registry_() {}

  TransformRegistry registry_;
};

class AppendFooTransform : public TransformBase {
 public:
  absl::Status Execute(api::ExecuteContext& context) override {
    context.outputs.push_back(
        api::Value(context.inputs[0].Get<std::string>() + "foo"));
    return absl::OkStatus();
  }

  absl::Span<const api::TypeInfo> InputTypes() const override {
    static const absl::NoDestructor<absl::InlinedVector<api::TypeInfo, 4>>
        kTypes({api::CreateTypeInfo<std::string>()});
    return absl::MakeConstSpan(*kTypes);
  }

  absl::Span<const api::TypeInfo> OutputTypes() const override {
    static const absl::NoDestructor<absl::InlinedVector<api::TypeInfo, 4>>
        kTypes({api::CreateTypeInfo<std::string>()});
    return absl::MakeConstSpan(*kTypes);
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    return absl::UnimplementedError("ToProto not implemented");
  }
};

class MacroRegisteredTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<MacroRegisteredTransform>> Create() {
    return std::make_unique<MacroRegisteredTransform>();
  }

  absl::Status Execute(api::ExecuteContext& context) override {
    return absl::OkStatus();
  }

  absl::Span<const api::TypeInfo> InputTypes() const override { return {}; }
  absl::Span<const api::TypeInfo> OutputTypes() const override { return {}; }

  absl::StatusOr<TransformConfig> ToProto() const override {
    return TransformConfig();
  }
};

TEST_F(TransformRegistryTest, RegisterAndCreate) {
  const std::string name = "successful_registration_and_creation_test";

  ASSERT_TRUE(registry_
                  .RegisterTransformFactory(
                      name,
                      [](const std::vector<api::Value>&) {
                        return std::make_unique<AppendFooTransform>();
                      })
                  .ok());

  ASSERT_OK_AND_ASSIGN(auto transform, registry_.CreateTransform(name));

  api::ExecuteContext context;
  std::vector<api::Value> inputs;
  inputs.push_back(api::Value(std::string("hello")));
  context.inputs = inputs;
  auto result = transform->Execute(context);
  ASSERT_OK(result) << result;
  EXPECT_EQ(context.outputs[0].Get<std::string>(), "hellofoo");
}

TEST_F(TransformRegistryTest, DuplicateRegistrationCallsFail) {
  const std::string name = "duplicate_transform_calls_fail";

  ASSERT_TRUE(registry_
                  .RegisterTransformFactory(
                      name,
                      [](const std::vector<api::Value>&) {
                        return std::make_unique<AppendFooTransform>();
                      })
                  .ok());

  auto status2 = registry_.RegisterTransformFactory(
      name, [](absl::Span<const api::Value>) {
        return std::make_unique<AppendFooTransform>();
      });
  ASSERT_FALSE(status2.ok());
  EXPECT_THAT(status2,
              absl_testing::StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST_F(TransformRegistryTest, NonExistentTransformReturnsError) {
  auto transform = registry_.CreateTransform("non_existent_transform");
  ASSERT_FALSE(transform.ok());
  EXPECT_THAT(transform.status(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

TEST_F(TransformRegistryTest, DeserializeTransformNotFound) {
  TransformConfig config;
  config.set_op_name("non_existent_transform");
  auto transform = registry_.DeserializeTransform(config);
  ASSERT_FALSE(transform.ok());
  EXPECT_THAT(transform.status(),
              absl_testing::StatusIs(absl::StatusCode::kNotFound));
}

ND_REGISTER_TRANSFORM(MacroRegisteredTransform, "MacroRegisteredTransform");
TEST(TransformRegistryMacroTest, RegisterAndCreate) {
  ASSERT_OK_AND_ASSIGN(auto transform,
                       TransformRegistry::Instance().CreateTransform(
                           "MacroRegisteredTransform"));
  EXPECT_NE(transform, nullptr);
}

ND_REGISTER_TRANSFORM(MacroRegisteredTransform,
                      "MacroRegisteredWithDeserializerTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return std::make_unique<MacroRegisteredTransform>();
        });
TEST(TransformRegistryMacroTest, RegisterAndCreateWithDeserializer) {
  TransformConfig config;
  config.set_op_name("MacroRegisteredWithDeserializerTransform");
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      TransformRegistry::Instance().DeserializeTransform(config));
  EXPECT_NE(transform, nullptr);
}

class CustomExecutorTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<CustomExecutorTransform>> Create() {
    return std::make_unique<CustomExecutorTransform>();
  }

  float operator()(float a, int b) { return a + b; }

  absl::StatusOr<TransformConfig> ToProto() const override {
    return TransformConfig();
  }
};

ND_REGISTER_TRANSFORM(CustomExecutorTransform, "CustomExecutorTransform");
TEST(TransformRegistryMacroTest, RegisterAndCreateWithExecutor) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      TransformRegistry::Instance().CreateTransform("CustomExecutorTransform"));
  ASSERT_NE(transform, nullptr);

  // Verify types were extracted correctly
  ASSERT_EQ(transform->InputTypes().size(), 2);
  EXPECT_TRUE(transform->InputTypes()[0].IsType<float>());
  EXPECT_TRUE(transform->InputTypes()[1].IsType<int>());

  ASSERT_EQ(transform->OutputTypes().size(), 1);
  EXPECT_TRUE(transform->OutputTypes()[0].IsType<float>());

  // Verify execution via the registry
  api::ExecuteContext context;
  api::Value arg1(1.5f);
  api::Value arg2(2);
  std::vector<api::Value> inputs;
  inputs.push_back(std::move(arg1));
  inputs.push_back(std::move(arg2));
  context.inputs = inputs;
  absl::Status status = transform->Execute(context);
  EXPECT_OK(status) << status;
  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_EQ(context.outputs[0].Get<float>(), 3.5f);
}

TEST(TransformRegistryMacroTest,
     RegisterAndCreateWithExecutorIncorrectArgsError) {
  ASSERT_OK_AND_ASSIGN(
      auto transform,
      TransformRegistry::Instance().CreateTransform("CustomExecutorTransform"));

  api::ExecuteContext context;
  api::Value arg1(1.5f);
  std::vector<api::Value> inputs;
  inputs.push_back(std::move(arg1));
  context.inputs = inputs;

  absl::Status status = transform->Execute(context);
  EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
  EXPECT_EQ(status.message(),
            "Expected 2 input(s) when calling Execute, but got 1 instead.");
}

class DefaultDebugStringTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<TransformBase>> Create() {
    return std::make_unique<DefaultDebugStringTransform>();
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_name("DefaultDebugStringTransform");
    config.set_op_name("DefaultDebugStringTransform");
    return config;
  }
};

class CustomDebugStringTransform : public TransformBase {
 public:
  absl::string_view DebugString() const override {
    return "some custom DebugString";
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_name("custom_debug_string");
    config.set_op_name("CustomDebugStringTransform");
    return config;
  }
};

TEST(TransformBaseTest, DebugStringPreference) {
  // Expect transforms with no implementation of DebugString
  // to return unknown.
  DefaultDebugStringTransform t1;
  EXPECT_EQ(t1.DebugString(), "TransformBase(unknown)");

  // Expect transforms registered with the transform registry to
  // return the registered name.
  TransformRegistrar<DefaultDebugStringTransform> registrar(
      "DefaultDebugStringTransform");
  auto transform = TransformRegistry::Instance().CreateTransform(
      "DefaultDebugStringTransform");
  ASSERT_OK(transform);
  EXPECT_EQ((*transform)->DebugString(), "DefaultDebugStringTransform");

  // Expect transforms with user provided DebugString to return that
  // string.
  CustomDebugStringTransform t3;
  EXPECT_EQ(t3.DebugString(), "some custom DebugString");
}

class ConstrainedDummyTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<TransformBase>> Create() {
    return std::make_unique<ConstrainedDummyTransform>();
  }
  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer a) { return a; }
  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("ConstrainedDummyTransform");
    return config;
  }
  absl::string_view DebugString() const override {
    return "ConstrainedDummyTransform";
  }

 protected:
  absl::Status PropagateSignatureImpl(
      api::ShapeInferenceContext& context) const override {
    if (context.inputs.size() != 1) {
      return absl::InvalidArgumentError(
          "ConstrainedDummyTransform expects 1 input");
    }
    context.outputs.push_back(context.inputs[0]);
    return absl::OkStatus();
  }
};

ND_REGISTER_TRANSFORM(ConstrainedDummyTransform, "ConstrainedDummyTransform")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return ConstrainedDummyTransform::Create();
        })
    .SetInputConstraints({{.allowed_dtypes = {xla::ifrt::DType::kS32}}});

TEST(TransformBaseTest, InputConstraintsValidation) {
  auto transform = TransformBase::FromProto(
      []() {
        TransformConfig config;
        config.set_op_name("ConstrainedDummyTransform");
        return config;
      }(),
      {});

  ASSERT_OK(transform) << transform.status().message();

  // Test with valid input (S32)
  {
    api::ExecuteContext context;
    neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                       xla::ifrt::Shape({1}));
    std::vector<api::Value> inputs = {api::Value(hb)};
    context.inputs = inputs;
    absl::Status status = (*transform)->Execute(context);
    EXPECT_OK(status) << status.message();
  }

  // Test with invalid input (F32)
  {
    api::ExecuteContext context;
    neptune::HostBuffer hb(xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                       xla::ifrt::Shape({1}));
    std::vector<api::Value> inputs = {api::Value(hb)};
    context.inputs = inputs;
    absl::Status status = (*transform)->Execute(context);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(status.message(), testing::HasSubstr("unsupported dtype: F32"));
  }
}

class DummyTransformWithoutShapeInference : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<TransformBase>> Create() {
    return std::make_unique<DummyTransformWithoutShapeInference>();
  }

  absl::StatusOr<neptune::HostBuffer> operator()(neptune::HostBuffer) {
    return absl::UnimplementedError("Execution not supported by dummy");
  }

  absl::StatusOr<TransformConfig> ToProto() const override {
    TransformConfig config;
    config.set_op_name("DummyTransformWithoutShapeInference");
    return config;
  }

  absl::string_view DebugString() const override {
    return "DummyTransformWithoutShapeInference";
  }
};

ND_REGISTER_TRANSFORM(DummyTransformWithoutShapeInference,
                      "DummyTransformWithoutShapeInference")
    .SetDeserializer(
        [](const TransformConfig& config,
           absl::Span<const std::shared_ptr<neptune::data::Resource>> resources)
            -> absl::StatusOr<std::unique_ptr<TransformBase>> {
          return DummyTransformWithoutShapeInference::Create();
        });

TEST(TransformBaseTest, DefaultPropagateSignature) {
  auto transform = TransformBase::FromProto(
      []() {
        TransformConfig config;
        config.set_op_name("DummyTransformWithoutShapeInference");
        return config;
      }(),
      {});

  ASSERT_OK(transform) << transform.status().message();

  api::ShapeInferenceContext context;
  api::HostBufferSpec spec{xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                           xla::ifrt::Shape({5, 5})};
  std::vector<api::ShapeInferenceValue> inputs = {
      api::ShapeInferenceValue(spec)};
  context.inputs = inputs;

  absl::Status status = (*transform)->PropagateSignature(context);
  EXPECT_OK(status) << status.message();

  ASSERT_EQ(context.outputs.size(), 1);
  EXPECT_FALSE(context.outputs[0].IsSpec());
  EXPECT_FALSE(context.outputs[0].IsSpecList());
}

TEST(TransformBaseTest, PropagateSignature) {
  auto transform = TransformBase::FromProto(
      []() {
        TransformConfig config;
        config.set_op_name("ConstrainedDummyTransform");
        return config;
      }(),
      {});

  ASSERT_OK(transform) << transform.status().message();

  // Test with valid input (S32)
  {
    api::ShapeInferenceContext context;
    api::HostBufferSpec spec{xla::ifrt::DType(xla::ifrt::DType::Kind::kS32),
                             xla::ifrt::Shape({5, 5})};
    std::vector<api::ShapeInferenceValue> inputs = {
        api::ShapeInferenceValue(spec)};
    context.inputs = inputs;
    absl::Status status = (*transform)->PropagateSignature(context);
    EXPECT_OK(status) << status.message();
    ASSERT_EQ(context.outputs.size(), 1);
    ASSERT_TRUE(context.outputs[0].IsSpec());
    const auto& out_spec = context.outputs[0].spec();
    EXPECT_EQ(out_spec.dtype, spec.dtype);
    EXPECT_EQ(out_spec.shape, spec.shape);
  }

  // Test with invalid input (F32)
  {
    api::ShapeInferenceContext context;
    api::HostBufferSpec spec{xla::ifrt::DType(xla::ifrt::DType::Kind::kF32),
                             xla::ifrt::Shape({5, 5})};
    std::vector<api::ShapeInferenceValue> inputs = {
        api::ShapeInferenceValue(spec)};
    context.inputs = inputs;
    absl::Status status = (*transform)->PropagateSignature(context);
    EXPECT_FALSE(status.ok());
    EXPECT_EQ(status.code(), absl::StatusCode::kInvalidArgument);
    EXPECT_THAT(status.message(), testing::HasSubstr("unsupported dtype: F32"));
  }
}

}  // namespace
}  // namespace core
}  // namespace composite
}  // namespace neptune::data

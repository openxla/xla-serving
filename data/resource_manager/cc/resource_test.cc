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

#include "data/resource_manager/cc/resource.h"  // from @xla_serving

#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @xla_serving
#include "data/resource_manager/cc/resource_internal.h"  // from @xla_serving
#include "data/test_utils/status_macros.h"  // from @xla_serving

namespace test_namespace {
namespace {

class FakeResource : public xla_serving::data::Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("TestNamespaceFakeResource");
  }
};

}  // namespace
}  // namespace test_namespace

namespace xla_serving::data {
namespace {

using ::absl_testing::StatusIs;
using ::testing::IsEmpty;
using ::testing::Not;
using ::testing::Return;

class FakeResource : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override { return absl::OkStatus(); }
};

class MacroRegisteredFakeResource : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("MacroRegisteredFakeResource");
  }
};

class UnregisteredResource : public Resource {
  absl::Status LoadImpl() override { return absl::OkStatus(); }
};

class AutoResolveTestResource : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("AutoResolveTestResource");
  }
};

class MockResource : public Resource {
 public:
  using Resource::Resource;
  MOCK_METHOD(absl::Status, LoadImpl, (), (override));
};

// Internal logic **************************************************************
TEST(ResourceTest, ResourceTypeNotFound) {
  internal::ResourceTypeRegistry& registry =
      internal::ResourceTypeRegistry::GetInstance();
  EXPECT_THAT(
      registry.GetClassName(std::type_index(typeid(UnregisteredResource))),
      StatusIs(absl::StatusCode::kNotFound));
}

TEST(ResourceTest, ResourceFactoryRegisterAndCreate) {
  internal::ResourceFactory& factory = internal::ResourceFactory::GetInstance();
  std::string class_name = "FactoryTestResource";

  auto factory_fn = [](uint64_t id, std::string name, std::vector<Asset> assets,
                       std::string class_name) {
    return std::make_shared<FakeResource>(
        id, std::move(name), std::move(assets), std::move(class_name));
  };

  EXPECT_OK(factory.Register(class_name, std::move(factory_fn)));

  // Test Create
  std::vector<Asset> assets = {{.name = "asset1", .path = "path1"}};
  ASSERT_OK_AND_ASSIGN(auto resource,
                       factory.Create(class_name, 123, "res_name", assets));
  const std::vector<Asset>& retrieved_assets = resource->assets();
  EXPECT_EQ(resource->id(), 123);
  EXPECT_EQ(resource->name(), "res_name");
  EXPECT_EQ(resource->class_name(), class_name);
  EXPECT_EQ(retrieved_assets.size(), 1);
  EXPECT_EQ(retrieved_assets[0].name, "asset1");
  EXPECT_EQ(retrieved_assets[0].path, "path1");

  // Test Duplicate Register
  EXPECT_THAT(factory.Register(class_name,
                               [](auto, auto, auto, auto) { return nullptr; }),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(ResourceTest, ResourceFactoryCreateNotFound) {
  auto& factory = internal::ResourceFactory::GetInstance();
  EXPECT_THAT(factory.Create("NonExistentClass", 123, "name", {}),
              StatusIs(absl::StatusCode::kNotFound));
}

// Public Apis *****************************************************************
TEST(ResourceTest, ResourceConstructors) {
  // Default constructor
  FakeResource r1;
  EXPECT_THAT(r1.class_name(), IsEmpty());  // Empty by default.
  EXPECT_THAT(r1.name(), Not(IsEmpty()));   // Should have random ID
  EXPECT_THAT(r1.assets(), IsEmpty());

  // Constructor with assets
  std::vector<Asset> assets = {{"asset1", "/path/1"}};
  FakeResource r2(assets);
  EXPECT_THAT(r1.class_name(), IsEmpty());  // Empty by default.
  EXPECT_THAT(r2.name(), Not(IsEmpty()));
  EXPECT_EQ(r2.assets().size(), 1);
  EXPECT_EQ(r2.assets()[0].name, "asset1");

  // Constructor with name and assets
  FakeResource r3("custom_name", assets, "CustomClass");
  EXPECT_EQ(r3.name(), "custom_name");
  EXPECT_EQ(r3.class_name(), "CustomClass");
  EXPECT_EQ(r3.assets().size(), 1);

  // Verify all resource IDs are unique and non-zero.
  EXPECT_NE(r1.id(), 0);
  EXPECT_NE(r1.id(), r2.id());
  EXPECT_NE(r1.id(), r3.id());
  EXPECT_NE(r2.id(), r3.id());

  // Constructor with id, name and assets
  FakeResource r4(123, "custom_name2", assets, "CustomClass");
  EXPECT_EQ(r4.id(), 123);
  EXPECT_EQ(r4.name(), "custom_name2");
  EXPECT_EQ(r4.class_name(), "CustomClass");
  EXPECT_EQ(r4.assets().size(), 1);
}

ND_REGISTER_RESOURCE(MacroRegisteredFakeResource,
                     "MacroRegisteredFakeResource");
ND_REGISTER_RESOURCE(::test_namespace::FakeResource,
                     "TestNamespaceFakeResource");
TEST(ResourceTest, RegisteredResourceClassNameAutoResolution) {
  EXPECT_OK(
      RegisterResource<AutoResolveTestResource>("AutoResolveTestResource"));

  AutoResolveTestResource r;
  MacroRegisteredFakeResource r2;
  test_namespace::FakeResource r3;
  EXPECT_EQ(r.class_name(), "AutoResolveTestResource");
  EXPECT_EQ(r2.class_name(), "MacroRegisteredFakeResource");
  EXPECT_EQ(r3.class_name(), "TestNamespaceFakeResource");

  // Verify we can load user registered resources internally.
  internal::ResourceFactory& factory = internal::ResourceFactory::GetInstance();
  ASSERT_OK_AND_ASSIGN(auto resource1,
                       factory.Create("AutoResolveTestResource", 1, "r1", {}));
  EXPECT_EQ(resource1->class_name(), "AutoResolveTestResource");
  EXPECT_THAT(resource1->Load(),
              StatusIs(absl::StatusCode::kInternal, "AutoResolveTestResource"));

  ASSERT_OK_AND_ASSIGN(
      auto resource2,
      factory.Create("MacroRegisteredFakeResource", 2, "r2", {}));
  EXPECT_EQ(resource2->class_name(), "MacroRegisteredFakeResource");
  EXPECT_THAT(resource2->Load(), StatusIs(absl::StatusCode::kInternal,
                                          "MacroRegisteredFakeResource"));

  ASSERT_OK_AND_ASSIGN(
      auto resource3, factory.Create("TestNamespaceFakeResource", 3, "r3", {}));
  EXPECT_EQ(resource3->class_name(), "TestNamespaceFakeResource");
  EXPECT_THAT(resource3->Load(), StatusIs(absl::StatusCode::kInternal,
                                          "TestNamespaceFakeResource"));
}

TEST(ResourceTest, ResourceLoad) {
  FakeResource r;
  EXPECT_OK(r.Load());
}

TEST(ResourceTest, LoadImplCalledOnlyOnce) {
  MockResource resource;
  EXPECT_CALL(resource, LoadImpl).Times(1).WillOnce(Return(absl::OkStatus()));
  EXPECT_OK(resource.Load());
  EXPECT_OK(resource.Load());
}

}  // namespace
}  // namespace xla_serving::data

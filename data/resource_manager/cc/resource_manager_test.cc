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

#include "data/resource_manager/cc/resource_manager.h"  // from @neptune

#include <algorithm>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "gmock/gmock.h"
#include "gtest/gtest.h"
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_matchers.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @neptune
#include "data/test_utils/proto_matchers.h"  // from @neptune
#include "data/test_utils/status_macros.h"  // from @neptune
#include "google/protobuf/text_format.h"

namespace neptune::data {
namespace {

using ::absl_testing::StatusIs;
using ::neptune::data::test_utils::EqualsProtoIgnoringRepeatedFieldOrdering;
using ::testing::UnorderedElementsAre;
class FakeResource : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("FakeResource");
  }
};

class FakeResource2 : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("FakeResource2");
  }
};

class FakeResource3 : public Resource {
 public:
  using Resource::Resource;

 protected:
  absl::Status LoadImpl() override {
    return absl::InternalError("FakeResource3");
  }
};

ND_REGISTER_RESOURCE(FakeResource, "FakeResource");
ND_REGISTER_RESOURCE(FakeResource2, "FakeResource2");
ND_REGISTER_RESOURCE(FakeResource3, "FakeResource3");

TEST(ResourceManagerTest, CreateReturnsErrorOnDuplicateResource) {
  auto resource = std::make_shared<FakeResource>();
  EXPECT_THAT(ResourceManager::Create({resource, resource}),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(ResourceManagerTest, CreateReturnsErrorOnDuplicateResourceId) {
  auto resource1 =
      std::make_shared<FakeResource>(1, "resource_1", std::vector<Asset>{}, "");
  auto resource1_dup = std::make_shared<FakeResource>(
      1, "resource_1_dup", std::vector<Asset>{}, "FakeResource");
  EXPECT_THAT(ResourceManager::Create({resource1, resource1_dup}),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(ResourceManagerTest, AddResource) {
  ResourceManager manager;
  auto resource = std::make_shared<FakeResource>();
  EXPECT_OK(manager.AddResource(resource));
  ASSERT_OK_AND_ASSIGN(auto resource_get, manager.GetResource(resource->id()));
  EXPECT_EQ(resource_get, resource);
}

TEST(ResourceManagerTest, AddResourceIgnoringDuplicates) {
  ResourceManager manager;
  auto resource = std::make_shared<FakeResource>();
  EXPECT_OK(manager.AddResourceIgnoringDuplicates(resource));
  ASSERT_OK_AND_ASSIGN(auto resource_get, manager.GetResource(resource->id()));
  EXPECT_EQ(resource_get, resource);
}

TEST(ResourceManagerTest, AddResourceDuplicateIsAnError) {
  ResourceManager manager;
  auto resource = std::make_shared<FakeResource>();
  EXPECT_OK(manager.AddResource(resource));
  EXPECT_THAT(manager.AddResource(resource),
              StatusIs(absl::StatusCode::kAlreadyExists));
}

TEST(ResourceManagerTest, AddResourceIgnoringDuplicatesDuplicateIsIgnored) {
  ResourceManager manager;
  auto resource = std::make_shared<FakeResource>();
  EXPECT_OK(manager.AddResourceIgnoringDuplicates(resource));
  EXPECT_OK(manager.AddResourceIgnoringDuplicates(resource));
}

TEST(ResourceManagerTest, GetResourceNotFound) {
  ResourceManager manager;
  EXPECT_THAT(manager.GetResource(123), StatusIs(absl::StatusCode::kNotFound));
}

TEST(ResourceManagerTest, AddAndGetDifferentResources) {
  ResourceManager manager;
  auto resourcex = std::make_shared<FakeResource>();
  auto resourcey = std::make_shared<FakeResource>();
  EXPECT_OK(manager.AddResourceIgnoringDuplicates(resourcex));
  EXPECT_OK(manager.AddResourceIgnoringDuplicates(resourcey));
  ASSERT_OK_AND_ASSIGN(auto resourcex_get,
                       manager.GetResource(resourcex->id()));
  ASSERT_OK_AND_ASSIGN(auto resourcey_get,
                       manager.GetResource(resourcey->id()));
  EXPECT_EQ(resourcex_get, resourcex);
  EXPECT_EQ(resourcey_get, resourcey);
  EXPECT_NE(resourcex_get, resourcey_get);
  EXPECT_NE(resourcex_get->id(), resourcey_get->id());
}

constexpr absl::string_view kSerializedResourceManagerProto = R"pb(
  resources { id: 1 name: "resource_1" class_name: "FakeResource" }
  resources {
    id: 2
    name: "resource_2"
    assets { name: "asset2" path: "path2" }
    class_name: "FakeResource2"
  }
)pb";

TEST(ResourceManagerTest, ToProto) {
  std::vector<Asset> assets = {};
  auto resource = std::make_shared<FakeResource>(1, "resource_1", assets, "");
  std::vector<Asset> assets2 = {{.name = "asset2", .path = "path2"}};
  auto resource2 =
      std::make_shared<FakeResource2>(2, "resource_2", assets2, "");
  ASSERT_OK_AND_ASSIGN(auto manager,
                       ResourceManager::Create({resource, resource2}));
  ResourceManagerProto proto = manager->ToProto();
  EXPECT_THAT(proto, EqualsProtoIgnoringRepeatedFieldOrdering(
                         kSerializedResourceManagerProto));
}

TEST(ResourceManagerTest, ToProtoSomeResourcesAddedAfterCtor) {
  std::vector<Asset> assets = {};
  auto resource = std::make_shared<FakeResource>(1, "resource_1", assets, "");
  std::vector<Asset> assets2 = {{.name = "asset2", .path = "path2"}};
  auto resource2 =
      std::make_shared<FakeResource2>(2, "resource_2", assets2, "");
  ASSERT_OK_AND_ASSIGN(auto manager, ResourceManager::Create({resource}));
  EXPECT_OK(manager->AddResourceIgnoringDuplicates(resource2));
  ResourceManagerProto proto = manager->ToProto();
  EXPECT_THAT(proto, EqualsProtoIgnoringRepeatedFieldOrdering(
                         kSerializedResourceManagerProto));
}

TEST(ResourceManagerTest, FromProto) {
  std::vector<Asset> assets = {};
  auto extra_res = std::make_shared<FakeResource3>("resource_3", assets, "");
  ResourceManagerProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      kSerializedResourceManagerProto, &proto));

  ASSERT_OK_AND_ASSIGN(auto manager, ResourceManager::FromProto(proto));
  EXPECT_OK(manager->AddResourceIgnoringDuplicates(extra_res));
  auto resources = manager->GetAllResources();
  auto it = std::find_if(resources.begin(), resources.end(),
                         [](std::shared_ptr<Resource> resource) {
                           return resource->name() == "resource_1";
                         });
  Resource* resource1 = it->get();
  it = std::find_if(resources.begin(), resources.end(),
                    [](std::shared_ptr<Resource> resource) {
                      return resource->name() == "resource_2";
                    });
  Resource* resource2 = it->get();
  it = std::find_if(resources.begin(), resources.end(),
                    [](std::shared_ptr<Resource> resource) {
                      return resource->name() == "resource_3";
                    });
  Resource* resource3 = it->get();

  ASSERT_EQ(resources.size(), 3);

  EXPECT_EQ(resource1->name(), "resource_1");
  ASSERT_EQ(resource1->assets().size(), 0);
  EXPECT_THAT(resource1->Load(),
              StatusIs(absl::StatusCode::kInternal, "FakeResource"));

  EXPECT_EQ(resource2->name(), "resource_2");
  ASSERT_EQ(resource2->assets().size(), 1);
  EXPECT_EQ(resource2->assets()[0].name, "asset2");
  EXPECT_EQ(resource2->assets()[0].path, "path2");
  EXPECT_THAT(resource2->Load(),
              StatusIs(absl::StatusCode::kInternal, "FakeResource2"));

  EXPECT_EQ(resource3->name(), "resource_3");
  ASSERT_EQ(resource3->assets().size(), 0);
  EXPECT_THAT(resource3->Load(),
              StatusIs(absl::StatusCode::kInternal, "FakeResource3"));
}

TEST(ResourceManagerTest, FromProtoLoadsAllAssetFields) {
  constexpr absl::string_view kProtoWithAllAssetFields = R"pb(
    resources {
      id: 1
      name: "resource_1"
      class_name: "FakeResource"
      assets { name: "asset_with_path" path: "path/to/asset" }
      assets { name: "asset_with_content" content: "some content" }
    }
  )pb";

  ResourceManagerProto proto;
  ASSERT_TRUE(
      google::protobuf::TextFormat::ParseFromString(kProtoWithAllAssetFields, &proto));
  ASSERT_OK_AND_ASSIGN(auto manager, ResourceManager::FromProto(proto));

  auto resources = manager->GetAllResources();
  ASSERT_EQ(resources.size(), 1);
  Resource* resource = resources[0].get();

  ASSERT_EQ(resource->assets().size(), 2);

  EXPECT_EQ(resource->assets()[0].name, "asset_with_path");
  EXPECT_EQ(resource->assets()[0].path, "path/to/asset");
  EXPECT_EQ(resource->assets()[0].content, "");

  EXPECT_EQ(resource->assets()[1].name, "asset_with_content");
  EXPECT_EQ(resource->assets()[1].path, "");
  EXPECT_EQ(resource->assets()[1].content, "some content");
}

TEST(ResourceManagerTest, FromProtoFailsIfIdIsMissing) {
  ResourceManagerProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        resources { name: "resource_1" class_name: "FakeResource" }
      )pb",
      &proto));
  EXPECT_THAT(ResourceManager::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResourceManagerTest, FromProtoFailsIfPathHasTraversal) {
  ResourceManagerProto proto;
  ASSERT_TRUE(google::protobuf::TextFormat::ParseFromString(
      R"pb(
        resources {
          id: 1
          name: "resource_1"
          class_name: "FakeResource"
          assets { name: "asset1" path: "foo/../bar" }
        }
      )pb",
      &proto));
  EXPECT_THAT(ResourceManager::FromProto(proto),
              StatusIs(absl::StatusCode::kInvalidArgument));
}

TEST(ResourceManagersViewTest, GetAllResources) {
  auto resource1 =
      std::make_shared<FakeResource>("res1", std::vector<Asset>{}, "");
  ASSERT_OK_AND_ASSIGN(auto manager1, ResourceManager::Create({resource1}));

  auto resource2 =
      std::make_shared<FakeResource2>("res2", std::vector<Asset>{}, "");
  auto resource3 =
      std::make_shared<FakeResource3>("res3", std::vector<Asset>{}, "");
  ASSERT_OK_AND_ASSIGN(auto manager2,
                       ResourceManager::Create({resource2, resource3}));

  std::vector<std::unique_ptr<ResourceManager>> managers;
  managers.push_back(std::move(manager1));
  managers.push_back(std::move(manager2));

  ResourceManagersView view(managers);
  std::vector<const Resource*> resources = view.GetAllResources();
  EXPECT_THAT(resources, UnorderedElementsAre(resource1.get(), resource2.get(),
                                              resource3.get()));
}

TEST(ResourceManagersViewTest, GetResource) {
  auto resource1 =
      std::make_shared<FakeResource>(1, "res1", std::vector<Asset>{}, "");
  ASSERT_OK_AND_ASSIGN(auto manager1, ResourceManager::Create({resource1}));

  auto resource2 =
      std::make_shared<FakeResource2>(2, "res2", std::vector<Asset>{}, "");
  auto resource3 =
      std::make_shared<FakeResource3>(3, "res3", std::vector<Asset>{}, "");
  ASSERT_OK_AND_ASSIGN(auto manager2,
                       ResourceManager::Create({resource2, resource3}));

  std::vector<std::unique_ptr<ResourceManager>> managers;
  managers.push_back(std::move(manager1));
  managers.push_back(std::move(manager2));

  ResourceManagersView view(managers);

  ASSERT_OK_AND_ASSIGN(auto get1, view.GetResource(1));
  ASSERT_OK_AND_ASSIGN(auto get2, view.GetResource(2));
  ASSERT_OK_AND_ASSIGN(auto get3, view.GetResource(3));
  EXPECT_EQ(get1, resource1.get());
  EXPECT_EQ(get2, resource2.get());
  EXPECT_EQ(get3, resource3.get());
  EXPECT_THAT(view.GetResource(4), StatusIs(absl::StatusCode::kNotFound));
}

}  // namespace
}  // namespace neptune::data

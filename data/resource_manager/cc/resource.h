#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <typeindex>
#include <utility>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "data/resource_manager/cc/asset.h"  // from @neptune
#include "data/resource_manager/cc/resource_internal.h"  // from @neptune
#include "data/resource_manager/proto/resource_manager.pb.h"  // from @neptune

namespace neptune::data {

class Resource {
 public:
  explicit Resource();
  explicit Resource(std::vector<Asset> assets);
  Resource(std::string name, std::vector<Asset> assets, std::string class_name);
  // Constructor used only for tests that need to specify the id.
  // By default, Resource::id should be generated to prevent collisions between
  // different instances of resources automatically.
  Resource(uint64_t id, std::string name, std::vector<Asset> assets,
           std::string class_name);

  virtual ~Resource() = default;

  uint64_t id() const;
  const std::string& name() const;
  const std::vector<Asset>& assets() const;
  const std::string& class_name() const;

  absl::Status Load();
  ResourceProto ToProto() const;

  template <typename Sink>
  friend void AbslStringify(Sink& sink, const Resource& resource) {
    std::string assets_str = "[ ";
    for (const auto& asset : resource.assets()) {
      absl::StrAppend(&assets_str, asset);
      if (&asset != &resource.assets().back()) {
        absl::StrAppend(&assets_str, ", ");
      }
    }
    absl::StrAppend(&assets_str, "]");
    absl::Format(&sink, "Resource(id=%d, name=%s, class_name=%s, assets=%s)",
                 resource.id_, resource.name_, resource.class_name_,
                 assets_str);
  }

 protected:
  explicit Resource(uint64_t id);
  Resource(uint64_t id, std::vector<Asset> assets);

  virtual absl::Status LoadImpl() = 0;
  mutable absl::Mutex mu_;
  bool loaded_ ABSL_GUARDED_BY(mu_) = false;
  uint64_t id_;
  std::string name_;
  mutable std::string class_name_;
  std::vector<Asset> assets_;
};

template <typename T>
absl::Status RegisterResource(const std::string& class_name) {
  RETURN_IF_ERROR(internal::ResourceFactory::GetInstance().Register(
      class_name,
      [](uint64_t id, std::string name,
         std::vector<neptune::data::Asset> assets, std::string class_name) {
        return std::make_shared<T>(id, std::move(name), std::move(assets),
                                   std::move(class_name));
      }));
  return internal::ResourceTypeRegistry::GetInstance().Register(
      std::type_index(typeid(T)), class_name);
}

}  // namespace neptune::data

#ifndef ND_REGISTER_RESOURCE_HOOK
#define ND_REGISTER_RESOURCE_HOOK(Type, Name)
#endif

#define ND_REGISTER_RESOURCE_INTERNAL_IMPL(Type, Name, Id)   \
  static bool Resource_Registered_##Id = []() {              \
    CHECK_OK(::neptune::data::RegisterResource<Type>(Name)); \
    ND_REGISTER_RESOURCE_HOOK(Type, Name);                   \
    return true;                                             \
  }();

#define ND_REGISTER_RESOURCE_INTERNAL(Type, Name, Id) \
  ND_REGISTER_RESOURCE_INTERNAL_IMPL(Type, Name, Id)

#define ND_REGISTER_RESOURCE(Type, Name) \
  ND_REGISTER_RESOURCE_INTERNAL(Type, Name, __COUNTER__)

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_CC_RESOURCE_H_

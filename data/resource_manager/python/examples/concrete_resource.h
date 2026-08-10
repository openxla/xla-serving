#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_CONCRETE_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_CONCRETE_RESOURCE_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {

// Concrete C++ resource for factory registration.
class CppConcreteResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override {
    return absl::InternalError("CppConcreteResource");
  }
};

ND_REGISTER_RESOURCE(CppConcreteResource, "CppConcreteResource");

// Another Concrete C++ resource for factory registration.
class OkConcreteResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override { return absl::OkStatus(); }
};

ND_REGISTER_RESOURCE(OkConcreteResource, "OkConcreteResource");

}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_CONCRETE_RESOURCE_H_

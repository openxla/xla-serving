#ifndef THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_ANOTHER_CONCRETE_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_ANOTHER_CONCRETE_RESOURCE_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace testing_namespace {

// Concrete C++ resource for factory registration.
class AnotherConcreteResource : public neptune::data::Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
};

}  // namespace testing_namespace

#endif  // THIRD_PARTY_NEPTUNE_DATA_RESOURCE_MANAGER_PYTHON_EXAMPLES_ANOTHER_CONCRETE_RESOURCE_H_

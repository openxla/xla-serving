#include "data/resource_manager/python/examples/another_concrete_resource.h"  // from @neptune

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace testing_namespace {

absl::Status AnotherConcreteResource::LoadImpl() {
  return absl::InternalError("AnotherConcreteResource");
}

ND_REGISTER_RESOURCE(AnotherConcreteResource, "AnotherConcreteResource");

}  // namespace testing_namespace

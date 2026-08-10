#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

class IntResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
  int value() const { return value_; }

 private:
  int value_ = 0;
};

class IntContentResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
  int value() const { return value_; }

 private:
  int value_ = 0;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_INT_RESOURCE_H_

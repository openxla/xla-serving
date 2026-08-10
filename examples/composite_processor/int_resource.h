#ifndef THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_INT_RESOURCE_H_
#define THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_INT_RESOURCE_H_

#include "absl/status/status.h"  // from @abseil-cpp
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune_examples {
namespace composite {

class IntResource : public neptune::data::Resource {
 public:
  using neptune::data::Resource::Resource;
  absl::Status LoadImpl() override;
  int value() const { return value_; }

 private:
  int value_ = 0;
};

}  // namespace composite
}  // namespace neptune_examples

#endif  // THIRD_PARTY_NEPTUNE_EXAMPLES_COMPOSITE_PROCESSOR_INT_RESOURCE_H_

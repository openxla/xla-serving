#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_DUMMY_RESOURCE_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_DUMMY_RESOURCE_TRANSFORM_H_

#include <memory>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune

namespace neptune::data {
namespace composite {

class DummyResource : public Resource {
 public:
  using Resource::Resource;
  absl::Status LoadImpl() override;
  bool loaded() const ABSL_LOCKS_EXCLUDED(mu_) {
    absl::MutexLock lock(mu_);
    return loaded_;
  }
};

class DummyResourceTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<DummyResourceTransform>> Create();
  explicit DummyResourceTransform(
      std::shared_ptr<neptune::data::Resource> resource);
  void operator()();
  absl::StatusOr<TransformConfig> ToProto() const override;
  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;
  absl::string_view DebugString() const override {
    return "DummyResourceTransform";
  }

 private:
  std::vector<std::shared_ptr<neptune::data::Resource>> resources_;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_PYTHON_EXAMPLES_DUMMY_RESOURCE_TRANSFORM_H_

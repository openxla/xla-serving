#ifndef THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_
#define THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding_resource.h"  // from @neptune
#include "data/processor/cc/api/transform_context.h"  // from @neptune
#include "data/processor/cc/api/type_info.h"  // from @neptune
#include "data/processor/cc/core/transform.h"  // from @neptune
#include "data/processor/proto/transform_config.pb.h"  // from @neptune
#include "data/resource_manager/cc/resource.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune

namespace neptune::data {
namespace composite {

class CpuEmbeddingLookupTransform : public TransformBase {
 public:
  static absl::StatusOr<std::unique_ptr<CpuEmbeddingLookupTransform>> Create(
      std::shared_ptr<CpuEmbeddingEngineResource> resource,
      std::vector<std::string> feature_names, bool dense_lookup = false,
      std::vector<std::string> output_feature_names = {});

  std::vector<std::shared_ptr<neptune::data::Resource>> GetAllResources()
      const override;

  absl::StatusOr<std::vector<neptune::HostBuffer>> operator()(
      std::vector<neptune::HostBuffer> ids,
      std::vector<neptune::HostBuffer> row_pointers,
      std::vector<neptune::HostBuffer> weights);

  absl::StatusOr<TransformConfig> ToProto() const override;

  absl::string_view DebugString() const override {
    return "CpuEmbeddingLookupTransform";
  }

 private:
  CpuEmbeddingLookupTransform(
      std::shared_ptr<CpuEmbeddingEngineResource> resource,
      std::vector<std::string> feature_names,
      std::vector<std::string> output_feature_names, bool dense_lookup);

  std::shared_ptr<CpuEmbeddingEngineResource> resource_;
  std::vector<std::string> feature_names_;
  std::vector<std::string> output_feature_names_;
  bool dense_lookup_ = false;
};

}  // namespace composite
}  // namespace neptune::data

#endif  // THIRD_PARTY_NEPTUNE_DATA_KERNELS_CPU_EMBEDDING_CPU_EMBEDDING_LOOKUP_TRANSFORM_H_

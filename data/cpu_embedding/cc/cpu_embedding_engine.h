#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_spec.h"  // from @neptune

namespace neptune::data::cpu_embedding {

class CpuEmbeddingEngine final {
 public:
  struct Options {
    int max_cpu_slots = 8;
  };

  CpuEmbeddingEngine();
  explicit CpuEmbeddingEngine(Options options);
  ~CpuEmbeddingEngine();

  absl::Status RegisterFeatureSpec(FeatureSpec feature_spec);

  absl::Status AddLocalTable(std::string name,
                             std::shared_ptr<const EmbeddingTable> table);

  absl::StatusOr<std::vector<Array>> Execute(
      absl::Span<const FeatureBatchCsrInput> feature_batches) const;

  template <typename T>
  absl::StatusOr<absl::Span<T>> Execute(
      absl::string_view feature_name, absl::Span<const int64_t> embedding_ids,
      absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
      absl::AnyInvocable<absl::Span<T>(size_t)> allocator) const;

  void Clear();

  absl::StatusOr<FeatureSpec> GetFeatureSpec(absl::string_view feature_name)
      const ABSL_LOCKS_EXCLUDED(feature_specs_mu_);

  absl::StatusOr<std::shared_ptr<const EmbeddingTable>> GetLocalTable(
      absl::string_view table_name) const ABSL_LOCKS_EXCLUDED(local_tables_mu_);

 private:
  mutable absl::Mutex feature_specs_mu_;
  mutable absl::Mutex local_tables_mu_;

  absl::flat_hash_map<std::string, FeatureSpec> feature_specs_
      ABSL_GUARDED_BY(feature_specs_mu_);
  absl::flat_hash_map<std::string, std::shared_ptr<const EmbeddingTable>>
      local_tables_ ABSL_GUARDED_BY(local_tables_mu_);

  Options options_;
  mutable std::unique_ptr<Eigen::ThreadPoolInterface> thread_pool_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_CPU_EMBEDDING_ENGINE_H_

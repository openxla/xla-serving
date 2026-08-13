#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "unsupported/Eigen/CXX11/Tensor"

namespace neptune::data::cpu_embedding {

// Define the CustomCombinerFn signature matching C-ABI
typedef void (*CustomCombinerFn)(const int64_t* ids, int64_t ids_size,
                                 const int32_t* row_pointers,
                                 int64_t row_pointers_size,
                                 const float* weights, int64_t weights_size,
                                 const float* table_data, int64_t vocab_size,
                                 int64_t dim, float* output_data);

// Read-only embedding table.
class EmbeddingTable {
 public:
  enum class CombineMode {
    kSum,
    kMean,
    kSqrtN,
  };

  static absl::StatusOr<std::unique_ptr<EmbeddingTable>> Create(
      absl::string_view name, Array data,
      CombineMode combine_mode = CombineMode::kSum,
      CustomCombinerFn custom_combiner = nullptr);

  static absl::StatusOr<std::unique_ptr<EmbeddingTable>> CreateQuantized(
      absl::string_view name, Array data, Array scales, Array offsets,
      int64_t block_size, CombineMode combine_mode = CombineMode::kSum);

  EmbeddingTable() = default;

  // Not copyable.
  EmbeddingTable(const EmbeddingTable&) = delete;
  EmbeddingTable& operator=(const EmbeddingTable&) = delete;

  // Move only.
  EmbeddingTable(EmbeddingTable&& other) = default;
  EmbeddingTable& operator=(EmbeddingTable&& other) noexcept = default;

  absl::string_view Name() const { return name_; }

  // Self-explanatory getters.
  DType GetDType() const { return data_.GetDType(); }
  enum CombineMode GetCombineMode() const { return combine_mode_; }
  int64_t VocabSize() const { return vocab_size_; }
  int64_t EmbeddingDim() const { return embedding_dim_; }

  // Performs a lookup. Result is an Array with shape (ids.size(),
  // embedding_dim).
  absl::StatusOr<Array> Lookup(absl::Span<const int64_t> ids) const;

  // Get the table as a read-only Eigen Tensor.
  template <typename T>
  Array::TensorMap<const T, 2> AsTensor() const {
    return data_.AsTensor<const T, 2>();
  }

  CustomCombinerFn GetCustomCombiner() const { return custom_combiner_; }

  bool IsQuantized() const { return is_quantized_; }
  int64_t BlockSize() const { return block_size_; }
  const Array& Scales() const { return scales_; }
  const Array& Offsets() const { return offsets_; }

 private:
  // Private constructor. Use Create() instead.
  EmbeddingTable(absl::string_view name, Array data,
                 enum CombineMode combine_mode,
                 CustomCombinerFn custom_combiner, bool is_quantized,
                 int64_t block_size, Array scales, Array offsets);

  std::string name_;
  Array data_;
  enum CombineMode combine_mode_ = CombineMode::kSum;
  CustomCombinerFn custom_combiner_ = nullptr;

  bool is_quantized_ = false;
  int64_t block_size_ = 32;
  Array scales_;
  Array offsets_;

  // Cached for convenience.
  int64_t vocab_size_;
  int64_t embedding_dim_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_

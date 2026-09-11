// Copyright 2026 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     https://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_

#include <cstdint>
#include <memory>
#include <string>

#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
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

 private:
  // Private constructor. Use Create() instead.
  EmbeddingTable(absl::string_view name, Array data,
                 enum CombineMode combine_mode,
                 CustomCombinerFn custom_combiner);

  std::string name_;
  Array data_;
  enum CombineMode combine_mode_ = CombineMode::kSum;
  CustomCombinerFn custom_combiner_ = nullptr;

  // Cached for convenience.
  int64_t vocab_size_;
  int64_t embedding_dim_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_TABLE_H_

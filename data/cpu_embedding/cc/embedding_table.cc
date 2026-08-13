#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <optional>
#include <utility>

#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/cpu_embedding/cc/array.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

template <typename T>
absl::StatusOr<Array> LookupImpl(absl::Span<const int64_t> ids,
                                 const Array& data) {
  const int64_t vocab_size = data.GetShape().dims[0];
  const int64_t embedding_dim = data.GetShape().dims[1];
  Array results(T(0), Shape({static_cast<int64_t>(ids.size()), embedding_dim}));
  auto results_tensor = results.AsTensor<T, 2>();
  auto table_tensor = data.AsTensor<const T, 2>();

  for (int64_t i = 0; i < ids.size(); ++i) {
    int64_t id = ids[i];
    if (id < 0 || id >= vocab_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Embedding table has vocab size %d. Got id %d.", vocab_size, id));
    }
    results_tensor.chip(i, 0) = table_tensor.chip(id, 0);
  }
  return results;
}

template <typename T>
absl::StatusOr<Array> LookupQuantizedImpl(absl::Span<const int64_t> ids,
                                          const Array& data,
                                          const Array& scales,
                                          const Array& offsets,
                                          int64_t block_size) {
  const int64_t vocab_size = data.GetShape().dims[0];
  const int64_t embedding_dim = data.GetShape().dims[1];
  Array results(T(0), Shape({static_cast<int64_t>(ids.size()), embedding_dim}));
  auto results_tensor = results.AsTensor<T, 2>();

  auto q_data_tensor = data.AsTensor<const int8_t, 2>();
  auto scales_tensor = scales.AsTensor<const float, 2>();
  const bool has_offsets = offsets.GetShape().NumElements() != 0;
  std::optional<Array::TensorMap<const float, 2>> offsets_tensor;
  if (has_offsets) {
    offsets_tensor = offsets.AsTensor<const float, 2>();
  }

  for (int64_t i = 0; i < ids.size(); ++i) {
    int64_t id = ids[i];
    if (id < 0 || id >= vocab_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Embedding table has vocab size %d. Got id %d.", vocab_size, id));
    }
    for (int64_t j = 0; j < embedding_dim; ++j) {
      int64_t block_idx = j / block_size;
      float scale = scales_tensor(id, block_idx);
      float offset = has_offsets ? (*offsets_tensor)(id, block_idx) : 0.0f;
      int8_t q_val = q_data_tensor(id, j);
      results_tensor(i, j) = T(static_cast<float>(q_val) * scale + offset);
    }
  }
  return results;
}

}  // namespace

absl::StatusOr<std::unique_ptr<EmbeddingTable>> EmbeddingTable::Create(
    absl::string_view name, Array data, enum CombineMode combine_mode,
    CustomCombinerFn custom_combiner) {
  if (data.GetShape().dims.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Embedding table data must be a 2D array. Got shape: (%s)",
        absl::StrJoin(data.GetShape().dims, ",")));
  }
  if (name.empty()) {
    return absl::InvalidArgumentError("Embedding table name cannot be empty.");
  }

  return absl::WrapUnique(new EmbeddingTable(
      name, std::move(data), combine_mode, custom_combiner,
      /*is_quantized=*/false, /*block_size=*/32, Array(), Array()));
}

absl::StatusOr<std::unique_ptr<EmbeddingTable>> EmbeddingTable::CreateQuantized(
    absl::string_view name, Array data, Array scales, Array offsets,
    int64_t block_size, CombineMode combine_mode) {
  if (data.GetShape().dims.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Embedding table data must be a 2D array. Got shape: (%s)",
        absl::StrJoin(data.GetShape().dims, ",")));
  }
  if (data.GetDType() != DType::kInt8) {
    return absl::InvalidArgumentError(
        "Quantized table data must be of Int8 type.");
  }
  if (scales.GetShape().dims.size() != 2) {
    return absl::InvalidArgumentError("Scales must be 2D arrays.");
  }
  const bool has_offsets = offsets.GetShape().NumElements() != 0;
  if (has_offsets && offsets.GetShape().dims.size() != 2) {
    return absl::InvalidArgumentError("Offsets must be 2D arrays if provided.");
  }
  if (name.empty()) {
    return absl::InvalidArgumentError("Embedding table name cannot be empty.");
  }
  if (block_size <= 0) {
    return absl::InvalidArgumentError("Block size must be positive.");
  }

  return absl::WrapUnique(new EmbeddingTable(
      name, std::move(data), combine_mode, /*custom_combiner=*/nullptr,
      /*is_quantized=*/true, block_size, std::move(scales),
      std::move(offsets)));
}

absl::StatusOr<Array> EmbeddingTable::Lookup(
    absl::Span<const int64_t> ids) const {
  if (is_quantized_) {
    return LookupQuantizedImpl<float>(ids, data_, scales_, offsets_,
                                      block_size_);
  }
  switch (data_.GetDType()) {
    case DType::kBfloat16:
      return LookupImpl<Eigen::bfloat16>(ids, data_);
    case DType::kFloat32:
      return LookupImpl<float>(ids, data_);
    default:
      return absl::InvalidArgumentError(
          absl::StrFormat("Unsupported data type: %d", data_.GetDType()));
  }
}

EmbeddingTable::EmbeddingTable(absl::string_view name, Array data,
                               enum CombineMode combine_mode,
                               CustomCombinerFn custom_combiner,
                               bool is_quantized, int64_t block_size,
                               Array scales, Array offsets)
    : name_(name),
      data_(std::move(data)),
      combine_mode_(combine_mode),
      custom_combiner_(custom_combiner),
      is_quantized_(is_quantized),
      block_size_(block_size),
      scales_(std::move(scales)),
      offsets_(std::move(offsets)) {
  vocab_size_ = data_.GetShape().dims[0];
  embedding_dim_ = data_.GetShape().dims[1];
}

}  // namespace neptune::data::cpu_embedding

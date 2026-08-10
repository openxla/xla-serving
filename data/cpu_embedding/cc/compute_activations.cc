#include "data/cpu_embedding/cc/compute_activations.h"  // from @neptune

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>

#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

template <typename T>
absl::Status AssembleCombinedEmbeddings(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table,
    Array::TensorMap<T, 2> activations_tensor) {
  const T* table_data = embedding_table.AsTensor<T>().data();
  const int64_t dim = embedding_table.EmbeddingDim();
  const int64_t vocab_size = embedding_table.VocabSize();
  const size_t num_samples = row_pointers.size() - 1;

  for (size_t sample_num = 0; sample_num < num_samples; ++sample_num) {
    const size_t sample_start = row_pointers[sample_num];
    const size_t sample_size =
        row_pointers[sample_num + 1] - row_pointers[sample_num];

    if (sample_size == 0) {
      continue;
    }

    absl::Span<const int64_t> sample_ids =
        embedding_ids.subspan(sample_start, sample_size);
    absl::Span<const float> sample_weights =
        weights.empty() ? weights : weights.subspan(sample_start, sample_size);

    T* act_ptr = activations_tensor.data() + sample_num * dim;

    // 1. Bounds check scan
    for (size_t emb_idx = 0; emb_idx < sample_size; ++emb_idx) {
      int64_t id = sample_ids[emb_idx];
      if (id < 0 || id >= vocab_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Embedding ID (%d) at index %d is out of bounds [0, %d) for table "
            "%s.",
            id, sample_start + emb_idx, vocab_size, embedding_table.Name()));
      }
    }

    // 2. Hot lookup execution
    if (sample_weights.empty()) {
      for (size_t emb_idx = 0; emb_idx < sample_size; ++emb_idx) {
        const T* row_ptr = table_data + sample_ids[emb_idx] * dim;
        Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> row_vec(row_ptr,
                                                                      dim);
        Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> act_vec(act_ptr, dim);
        act_vec += row_vec;
      }

      float scale = 1.0f;
      switch (embedding_table.GetCombineMode()) {
        case EmbeddingTable::CombineMode::kSum:
          break;
        case EmbeddingTable::CombineMode::kMean:
          scale = 1.0f / sample_size;
          break;
        case EmbeddingTable::CombineMode::kSqrtN:
          scale = 1.0f / std::sqrt(static_cast<float>(sample_size));
          break;
      }
      if (scale != 1.0f) {
        Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> act_vec(act_ptr, dim);
        act_vec *= T(scale);
      }
    } else {
      float sum_weights = 0.0f;
      for (size_t emb_idx = 0; emb_idx < sample_size; ++emb_idx) {
        const T* row_ptr = table_data + sample_ids[emb_idx] * dim;
        float w = sample_weights[emb_idx];
        Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>> row_vec(row_ptr,
                                                                      dim);
        Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> act_vec(act_ptr, dim);
        act_vec += T(w) * row_vec;

        switch (embedding_table.GetCombineMode()) {
          case EmbeddingTable::CombineMode::kSum:
            break;
          case EmbeddingTable::CombineMode::kMean:
            sum_weights += w;
            break;
          case EmbeddingTable::CombineMode::kSqrtN:
            sum_weights += w * w;
            break;
        }
      }

      float scale = 1.0f;
      switch (embedding_table.GetCombineMode()) {
        case EmbeddingTable::CombineMode::kSum:
          break;
        case EmbeddingTable::CombineMode::kMean:
          scale = 1.0f / sum_weights;
          break;
        case EmbeddingTable::CombineMode::kSqrtN:
          scale = 1.0f / std::sqrt(sum_weights);
          break;
      }
      if (scale != 1.0f) {
        Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>> act_vec(act_ptr, dim);
        act_vec *= T(scale);
      }
    }
  }
  return absl::OkStatus();
}

template <typename T>
absl::StatusOr<Array> AssembleSequenceEmbeddings(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table, const int64_t max_sequence_length) {
  const int64_t vocab_size = embedding_table.VocabSize();
  Array activations(
      T(0), Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                   max_sequence_length, embedding_table.EmbeddingDim()}));

  auto embedding_table_tensor = embedding_table.AsTensor<T>();
  auto activations_tensor = activations.AsTensor<T, 3>();

  for (size_t sample_num = 0; sample_num < feature_batch.NumSamples();
       ++sample_num) {
    absl::Span<const int64_t> emb_ids = feature_batch.GetSampleIds(sample_num);
    size_t num_sample_ids =
        std::min(emb_ids.size(), static_cast<size_t>(max_sequence_length));

    // 1. Scan bounds
    for (size_t emb_idx = 0; emb_idx < num_sample_ids; ++emb_idx) {
      int64_t id = emb_ids[emb_idx];
      if (id < 0 || id >= vocab_size) {
        return absl::InvalidArgumentError(
            absl::StrFormat("Embedding ID (%d) at index %d is out of bounds "
                            "[0, %d) for table %s.",
                            id, emb_idx, vocab_size, embedding_table.Name()));
      }
    }

    // 2. Hot copy loop
    auto sample_sequence = activations_tensor.chip(sample_num, 0);
    for (size_t emb_idx = 0; emb_idx < num_sample_ids; ++emb_idx) {
      sample_sequence.chip(emb_idx, 0) =
          embedding_table_tensor.chip(emb_ids[emb_idx], 0);
    }
  }
  return activations;
}

}  // namespace

absl::StatusOr<Array> ComputeEmbeddingActivations(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table) {
  if (embedding_table.GetCustomCombiner() != nullptr) {
    // TODO: Support flexible / generic datatypes in
    // CustomCombiner.
    if (embedding_table.GetDType() != DType::kFloat32) {
      return absl::InvalidArgumentError(
          "Custom combiners are only supported for Float32 embedding tables.");
    }
    Array activations(0.f,
                      Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                             embedding_table.EmbeddingDim()}));
    embedding_table.GetCustomCombiner()(
        feature_batch.EmbeddingIds().data(),
        feature_batch.EmbeddingIds().size(), feature_batch.RowPointers().data(),
        feature_batch.RowPointers().size(), feature_batch.Weights().data(),
        feature_batch.Weights().size(),
        embedding_table.AsTensor<float>().data(), embedding_table.VocabSize(),
        embedding_table.EmbeddingDim(),
        activations.AsTensor<float, 2>().data());
    return activations;
  }

  switch (embedding_table.GetDType()) {
    case DType::kFloat32: {
      Array activations(0.f,
                        Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                               embedding_table.EmbeddingDim()}));
      RETURN_IF_ERROR(AssembleCombinedEmbeddings<float>(
          feature_batch.EmbeddingIds(), feature_batch.RowPointers(),
          feature_batch.Weights(), embedding_table,
          activations.AsTensor<float, 2>()));
      return activations;
    } break;
    case DType::kBfloat16: {
      Array activations(Eigen::bfloat16(0),
                        Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                               embedding_table.EmbeddingDim()}));
      RETURN_IF_ERROR(AssembleCombinedEmbeddings<Eigen::bfloat16>(
          feature_batch.EmbeddingIds(), feature_batch.RowPointers(),
          feature_batch.Weights(), embedding_table,
          activations.AsTensor<Eigen::bfloat16, 2>()));
      return activations;
    } break;
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported data type: %d", embedding_table.GetDType()));
  }
}

template <typename T>
absl::Status ComputeEmbeddingActivations(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<T> buffer) {
  const size_t num_samples = row_pointers.size() - 1;
  if (buffer.size() != num_samples * embedding_table.EmbeddingDim()) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Buffer size (%d) does not match the expected size (%d * %d)",
        buffer.size(), num_samples, embedding_table.EmbeddingDim()));
  }

  if (embedding_table.GetCustomCombiner() != nullptr) {
    if constexpr (std::is_same_v<T, float>) {
      embedding_table.GetCustomCombiner()(
          embedding_ids.data(), embedding_ids.size(), row_pointers.data(),
          row_pointers.size(), weights.data(), weights.size(),
          embedding_table.AsTensor<float>().data(), embedding_table.VocabSize(),
          embedding_table.EmbeddingDim(), buffer.data());
      return absl::OkStatus();
    } else {
      return absl::InvalidArgumentError(
          "Custom combiners are only supported for Float32 embedding tables.");
    }
  }

  Array::TensorMap<T, 2> activations_tensor(buffer.data(), num_samples,
                                            embedding_table.EmbeddingDim());
  return AssembleCombinedEmbeddings<T>(embedding_ids, row_pointers, weights,
                                       embedding_table, activations_tensor);
}

template absl::Status ComputeEmbeddingActivations<float>(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<float> buffer);

template absl::Status ComputeEmbeddingActivations<Eigen::bfloat16>(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<Eigen::bfloat16> buffer);

absl::StatusOr<Array> LookupEmbeddingSequence(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table, int64_t max_sequence_length) {
  if (max_sequence_length < 0) {
    return absl::InvalidArgumentError(
        absl::StrFormat("max_sequence_length must be non-negative, but got %d",
                        max_sequence_length));
  }

  if (!feature_batch.Weights().empty()) {
    return absl::InvalidArgumentError(
        "Weights should be empty for sequence assembly.");
  }

  switch (embedding_table.GetDType()) {
    case DType::kFloat32:
      return AssembleSequenceEmbeddings<float>(feature_batch, embedding_table,
                                               max_sequence_length);
    case DType::kBfloat16:
      return AssembleSequenceEmbeddings<Eigen::bfloat16>(
          feature_batch, embedding_table, max_sequence_length);
    default:
      return absl::InvalidArgumentError(absl::StrFormat(
          "Unsupported data type: %d", embedding_table.GetDType()));
  }
}

}  // namespace neptune::data::cpu_embedding

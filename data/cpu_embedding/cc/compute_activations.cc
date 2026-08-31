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

#include "data/cpu_embedding/cc/compute_activations.h"  // from @neptune

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include "absl/status/status.h"  // from @abseil-cpp
#include "xla/tsl/platform/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/compiled_custom_combiner.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune

namespace neptune::data::cpu_embedding {
namespace {

template <typename T, int Dim = Eigen::Dynamic>
absl::Status AssembleCombinedEmbeddings(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table,
    Array::TensorMap<T, 2> activations_tensor) {
  const T* table_data = embedding_table.AsTensor<T>().data();
  const int64_t dim = embedding_table.EmbeddingDim();
  const int64_t vocab_size = embedding_table.VocabSize();
  const size_t num_samples = row_pointers.size() - 1;

  auto get_row_vec = [&](const T* ptr) {
    if constexpr (Dim == Eigen::Dynamic) {
      return Eigen::Map<const Eigen::Matrix<T, 1, Eigen::Dynamic>>(ptr, dim);
    } else {
      return Eigen::Map<const Eigen::Matrix<T, 1, Dim>>(ptr);
    }
  };

  auto get_act_vec = [&](T* ptr) {
    if constexpr (Dim == Eigen::Dynamic) {
      return Eigen::Map<Eigen::Matrix<T, 1, Eigen::Dynamic>>(ptr, dim);
    } else {
      return Eigen::Map<Eigen::Matrix<T, 1, Dim>>(ptr);
    }
  };

  for (size_t sample_num = 0; sample_num < num_samples; ++sample_num) {
    const size_t sample_start = row_pointers[sample_num];
    const size_t sample_size =
        row_pointers[sample_num + 1] - row_pointers[sample_num];
    T* act_ptr = activations_tensor.data() + sample_num * dim;

    if (sample_size == 0) {
      get_act_vec(act_ptr).setZero();
      continue;
    }

    absl::Span<const int64_t> sample_ids =
        embedding_ids.subspan(sample_start, sample_size);
    absl::Span<const float> sample_weights =
        weights.empty() ? weights : weights.subspan(sample_start, sample_size);

    Eigen::Matrix<T, 1, Dim> accumulator;
    if constexpr (Dim == Eigen::Dynamic) {
      accumulator.resize(dim);
    }
    accumulator.setZero();

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
        accumulator += get_row_vec(row_ptr);
      }

      float scale = 1.0f;
      switch (embedding_table.GetCombineMode()) {
        case EmbeddingTable::CombineMode::kSum:
        case EmbeddingTable::CombineMode::kCustom:
          break;
        case EmbeddingTable::CombineMode::kMean:
          scale = 1.0f / sample_size;
          break;
        case EmbeddingTable::CombineMode::kSqrtN:
          scale = 1.0f / std::sqrt(static_cast<float>(sample_size));
          break;
      }
      if (scale != 1.0f) {
        accumulator *= T(scale);
      }
    } else {
      float sum_weights = 0.0f;
      for (size_t emb_idx = 0; emb_idx < sample_size; ++emb_idx) {
        const T* row_ptr = table_data + sample_ids[emb_idx] * dim;
        float w = sample_weights[emb_idx];
        accumulator += T(w) * get_row_vec(row_ptr);

        switch (embedding_table.GetCombineMode()) {
          case EmbeddingTable::CombineMode::kSum:
          case EmbeddingTable::CombineMode::kCustom:
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
        case EmbeddingTable::CombineMode::kCustom:
          break;
        case EmbeddingTable::CombineMode::kMean:
          scale = 1.0f / sum_weights;
          break;
        case EmbeddingTable::CombineMode::kSqrtN:
          scale = 1.0f / std::sqrt(sum_weights);
          break;
      }
      if (scale != 1.0f) {
        accumulator *= T(scale);
      }
    }
    get_act_vec(act_ptr) = accumulator;
  }
  return absl::OkStatus();
}

template <typename T>
absl::Status DispatchAssemble(absl::Span<const int64_t> embedding_ids,
                              absl::Span<const int32_t> row_pointers,
                              absl::Span<const float> weights,
                              const EmbeddingTable& embedding_table,
                              Array::TensorMap<T, 2> activations_tensor) {
  const int64_t dim = embedding_table.EmbeddingDim();
  switch (dim) {
    case 16:
      return AssembleCombinedEmbeddings<T, 16>(embedding_ids, row_pointers,
                                               weights, embedding_table,
                                               activations_tensor);
    case 32:
      return AssembleCombinedEmbeddings<T, 32>(embedding_ids, row_pointers,
                                               weights, embedding_table,
                                               activations_tensor);
    case 64:
      return AssembleCombinedEmbeddings<T, 64>(embedding_ids, row_pointers,
                                               weights, embedding_table,
                                               activations_tensor);
    case 128:
      return AssembleCombinedEmbeddings<T, 128>(embedding_ids, row_pointers,
                                                weights, embedding_table,
                                                activations_tensor);
    default:
      return AssembleCombinedEmbeddings<T, Eigen::Dynamic>(
          embedding_ids, row_pointers, weights, embedding_table,
          activations_tensor);
  }
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

absl::Status ExecuteCustomCombinerTileBlocked(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<float> buffer) {
  const auto* custom_combiner =
      embedding_table.GetCompiledCustomCombiner().get();
  if (custom_combiner == nullptr) {
    return absl::InternalError("Compiled custom combiner is null.");
  }
  const auto& options = custom_combiner->GetOptions();
  const int32_t tile_size = options.tile_size;
  const int32_t max_valency = options.max_valency;
  const int32_t dim = options.embedding_dim;
  const int32_t num_weights = options.num_weights;
  const int32_t output_dim = options.output_dim;

  const float* table_data = embedding_table.AsTensor<float>().data();
  const int64_t vocab_size = embedding_table.VocabSize();
  const size_t num_samples = row_pointers.size() - 1;

  if (buffer.size() != num_samples * output_dim) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Buffer size (%d) does not match expected (%d * %d)", buffer.size(),
        num_samples, output_dim));
  }

  thread_local std::vector<int32_t> tl_valencies;
  thread_local std::vector<float> tl_vectors;
  thread_local std::vector<float> tl_weights;
  thread_local std::vector<float> tl_output;

  tl_valencies.resize(tile_size);
  tl_vectors.resize(tile_size * max_valency * dim);
  if (num_weights > 0) {
    tl_weights.resize(tile_size * num_weights);
  }
  tl_output.resize(tile_size * output_dim);

  const size_t num_tiles = (num_samples + tile_size - 1) / tile_size;

  for (size_t tile_idx = 0; tile_idx < num_tiles; ++tile_idx) {
    const size_t tile_start = tile_idx * tile_size;
    const size_t tile_end = std::min(tile_start + tile_size, num_samples);
    const size_t active_samples = tile_end - tile_start;

    for (size_t i = 0; i < active_samples; ++i) {
      const size_t sample_num = tile_start + i;
      const size_t sample_start = row_pointers[sample_num];
      const size_t sample_size =
          row_pointers[sample_num + 1] - row_pointers[sample_num];
      const size_t valency =
          std::min(sample_size, static_cast<size_t>(max_valency));

      tl_valencies[i] = static_cast<int32_t>(valency);

      float* vec_dest = tl_vectors.data() + i * (max_valency * dim);
      for (size_t v = 0; v < valency; ++v) {
        int64_t id = embedding_ids[sample_start + v];
        if (id < 0 || id >= vocab_size) {
          return absl::InvalidArgumentError(absl::StrFormat(
              "Embedding ID (%d) at index %d is out of bounds [0, %d) for "
              "table %s.",
              id, sample_start + v, vocab_size, embedding_table.Name()));
        }
        const float* src = table_data + id * dim;
        std::copy_n(src, dim, vec_dest + v * dim);
      }
      if (valency < static_cast<size_t>(max_valency)) {
        std::fill_n(vec_dest + valency * dim, (max_valency - valency) * dim,
                    0.0f);
      }

      if (num_weights > 0 && !weights.empty()) {
        float* w_dest = tl_weights.data() + i * num_weights;
        const size_t w_count =
            std::min(sample_size, static_cast<size_t>(num_weights));
        for (size_t w = 0; w < w_count; ++w) {
          w_dest[w] = weights[sample_start + w];
        }
        if (w_count < static_cast<size_t>(num_weights)) {
          std::fill_n(w_dest + w_count, num_weights - w_count, 0.0f);
        }
      }
    }

    // Zero pad remaining inactive samples in the remainder tile
    if (active_samples < static_cast<size_t>(tile_size)) {
      std::fill(tl_valencies.begin() + active_samples, tl_valencies.end(), 0);
      std::fill(
          tl_vectors.begin() + active_samples * (max_valency * dim),
          tl_vectors.end(), 0.0f);
      if (num_weights > 0) {
        std::fill(tl_weights.begin() + active_samples * num_weights,
                  tl_weights.end(), 0.0f);
      }
    }

    RETURN_IF_ERROR(custom_combiner->ExecuteTile(
        absl::MakeSpan(tl_valencies), absl::MakeSpan(tl_vectors),
        num_weights > 0 ? absl::MakeSpan(tl_weights)
                        : absl::Span<const float>(),
        absl::MakeSpan(tl_output)));

    for (size_t i = 0; i < active_samples; ++i) {
      const size_t sample_num = tile_start + i;
      const float* src = tl_output.data() + i * output_dim;
      float* dst = buffer.data() + sample_num * output_dim;
      std::copy_n(src, output_dim, dst);
    }
  }

  return absl::OkStatus();
}

}  // namespace

absl::StatusOr<Array> ComputeEmbeddingActivations(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table) {
  if (embedding_table.GetCompiledCustomCombiner() != nullptr) {
    if (embedding_table.GetDType() != DType::kFloat32) {
      return absl::InvalidArgumentError(
          "Compiled custom combiners are only supported for Float32 embedding "
          "tables.");
    }
    const int64_t output_dim =
        embedding_table.GetCompiledCustomCombiner()->GetOptions().output_dim;
    Array activations(0.f,
                      Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                             output_dim}));
    RETURN_IF_ERROR(ExecuteCustomCombinerTileBlocked(
        feature_batch.EmbeddingIds(), feature_batch.RowPointers(),
        feature_batch.Weights(), embedding_table,
        absl::MakeSpan(activations.AsTensor<float, 2>().data(),
                       feature_batch.NumSamples() * output_dim)));
    return activations;
  }

  switch (embedding_table.GetDType()) {
    case DType::kFloat32: {
      Array activations(0.f,
                        Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                               embedding_table.EmbeddingDim()}));
      RETURN_IF_ERROR(DispatchAssemble<float>(
          feature_batch.EmbeddingIds(), feature_batch.RowPointers(),
          feature_batch.Weights(), embedding_table,
          activations.AsTensor<float, 2>()));
      return activations;
    } break;
    case DType::kBfloat16: {
      Array activations(Eigen::bfloat16(0),
                        Shape({static_cast<int64_t>(feature_batch.NumSamples()),
                               embedding_table.EmbeddingDim()}));
      RETURN_IF_ERROR(DispatchAssemble<Eigen::bfloat16>(
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
  const int64_t output_dim =
      embedding_table.GetCompiledCustomCombiner() != nullptr
          ? embedding_table.GetCompiledCustomCombiner()->GetOptions().output_dim
          : embedding_table.EmbeddingDim();
  if (buffer.size() != num_samples * output_dim) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Buffer size (%d) does not match the expected size (%d * %d)",
        buffer.size(), num_samples, output_dim));
  }

  if (embedding_table.GetCompiledCustomCombiner() != nullptr) {
    if constexpr (std::is_same_v<T, float>) {
      if (embedding_table.GetDType() != DType::kFloat32) {
        return absl::InvalidArgumentError(
            "Compiled custom combiners are only supported for Float32 "
            "embedding tables.");
      }
      return ExecuteCustomCombinerTileBlocked(
          embedding_ids, row_pointers, weights, embedding_table, buffer);
    } else {
      return absl::InvalidArgumentError(
          "Compiled custom combiners are only supported for Float32 embedding "
          "tables.");
    }
  }

  Array::TensorMap<T, 2> activations_tensor(buffer.data(), num_samples,
                                            embedding_table.EmbeddingDim());
  return DispatchAssemble<T>(embedding_ids, row_pointers, weights,
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

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

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_

#include <cstdint>

#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @xla_serving
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp

namespace neptune::data::cpu_embedding {

// Compute the embedding activations, return them as an NDArray. The shape of
// the output activations will be (batch_size, embedding_dim). The DType of the
// output activations will be the same as the embedding table.
absl::StatusOr<Array> ComputeEmbeddingActivations(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table);

// Compute the embedding activations, and store them in the given buffer. The
// buffer must be large enough to hold the activations, and the DType of the
// buffer must match the DType of the embedding table.
template <typename T>
absl::Status ComputeEmbeddingActivations(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<T> buffer);

// Lookup the embedding sequence for each sample in the feature batch. Returns
// an NDArray with shape (batch_size, max_sequence_length, embedding_dim).
//
// max_sequence_length specifies the maximum sequence length to use for each
// sample. The actual sequence length for each sample will be less than or equal
// to max_sequence_length.
absl::StatusOr<Array> LookupEmbeddingSequence(
    const FeatureBatchCsrInput& feature_batch,
    const EmbeddingTable& embedding_table, int64_t max_sequence_length);

}  // namespace neptune::data::cpu_embedding

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_

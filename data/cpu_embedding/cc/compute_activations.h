#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_

#include <cstdint>

#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune
#include "data/cpu_embedding/cc/feature_batch_csr_input.h"  // from @neptune
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

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPUTE_ACTIVATIONS_H_

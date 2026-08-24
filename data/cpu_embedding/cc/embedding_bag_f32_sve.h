// Copyright 2026 Arm Ltd. and affiliates
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

#ifndef XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_F32_SVE_H_
#define XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_F32_SVE_H_

#include <cstddef>
#include <cstdint>

namespace xla_serving::data::cpu_embedding::internal::sve {

// Returns the calling thread's SVE vector length in bytes.
int VectorLengthBytes();

// Accumulates validated FP32 bag numerators into preinitialized output. Row
// pointers are absolute offsets into IDs and optional weights; normalization
// remains in portable orchestration.
void AccumulateEmbeddingBagsF32(const int64_t* embedding_ids,
                                const int32_t* row_pointers, size_t num_bags,
                                const float* weights,
                                const float* table_data,
                                size_t embedding_dim, float* output);

}  // namespace xla_serving::data::cpu_embedding::internal::sve

#endif  // XLA_SERVING_DATA_CPU_EMBEDDING_CC_EMBEDDING_BAG_F32_SVE_H_

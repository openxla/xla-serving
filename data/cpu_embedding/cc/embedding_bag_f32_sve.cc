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

#include "data/cpu_embedding/cc/embedding_bag_f32_sve.h"  // from @xla_serving

#if defined(XLA_SERVING_HAS_SVE_EMBEDDING_BAG)
#include <arm_sve.h>
#else
#include <cstdlib>
#endif

#include <cstddef>
#include <cstdint>

#if defined(XLA_SERVING_HAS_SVE_EMBEDDING_BAG) && !defined(__ARM_FEATURE_SVE)
#error "embedding_bag_f32_sve.cc must be compiled with SVE enabled"
#endif

namespace xla_serving::data::cpu_embedding::internal::sve {

#if defined(XLA_SERVING_HAS_SVE_EMBEDDING_BAG)
namespace {

// Accumulate selected rows in registers without changing FP32 addition order.
template <bool kWeighted, size_t kNumIds>
inline void AccumulateIdBlock(const int64_t* embedding_ids,
                              const float* weights, const float* table_data,
                              size_t embedding_dim, float* output) {
  for (size_t dim = 0; dim < embedding_dim; dim += svcntw()) {
    const svbool_t predicate = svwhilelt_b32(dim, embedding_dim);
    svfloat32_t accumulator = svld1_f32(predicate, output + dim);

    for (size_t id_index = 0; id_index < kNumIds; ++id_index) {
      const size_t row = static_cast<size_t>(embedding_ids[id_index]);
      const float* row_data = table_data + row * embedding_dim;
      svfloat32_t values = svld1_f32(predicate, row_data + dim);
      if constexpr (kWeighted) {
        values = svmul_n_f32_x(predicate, values, weights[id_index]);
      }
      accumulator = svadd_f32_m(predicate, accumulator, values);
    }

    svst1_f32(predicate, output + dim, accumulator);
  }
}

template <bool kWeighted, size_t kNumIds>
inline void ConsumeIdBlocks(const int64_t* embedding_ids, size_t* position,
                            size_t end, const float* weights,
                            const float* table_data, size_t embedding_dim,
                            float* output) {
  while (end - *position >= kNumIds) {
    AccumulateIdBlock<kWeighted, kNumIds>(
        embedding_ids + *position, kWeighted ? weights + *position : nullptr,
        table_data, embedding_dim, output);
    *position += kNumIds;
  }
}

template <bool kWeighted>
void AccumulateEmbeddingBagsF32Impl(const int64_t* embedding_ids,
                                    const int32_t* row_pointers,
                                    size_t num_bags, const float* weights,
                                    const float* table_data,
                                    size_t embedding_dim, float* output) {
  for (size_t bag = 0; bag < num_bags; ++bag) {
    size_t position = static_cast<size_t>(row_pointers[bag]);
    const size_t end = static_cast<size_t>(row_pointers[bag + 1]);
    float* bag_output = output + bag * embedding_dim;

    // Reuse each ID block across dimensions to reduce random-row TLB pressure.
    ConsumeIdBlocks<kWeighted, 16>(embedding_ids, &position, end, weights,
                                   table_data, embedding_dim, bag_output);
    ConsumeIdBlocks<kWeighted, 8>(embedding_ids, &position, end, weights,
                                  table_data, embedding_dim, bag_output);
    ConsumeIdBlocks<kWeighted, 4>(embedding_ids, &position, end, weights,
                                  table_data, embedding_dim, bag_output);
    ConsumeIdBlocks<kWeighted, 2>(embedding_ids, &position, end, weights,
                                  table_data, embedding_dim, bag_output);
    ConsumeIdBlocks<kWeighted, 1>(embedding_ids, &position, end, weights,
                                  table_data, embedding_dim, bag_output);
  }
}

}  // namespace

int VectorLengthBytes() { return static_cast<int>(svcntb()); }

void AccumulateEmbeddingBagsF32(const int64_t* embedding_ids,
                                const int32_t* row_pointers, size_t num_bags,
                                const float* weights,
                                const float* table_data,
                                size_t embedding_dim, float* output) {
  if (weights == nullptr) {
    AccumulateEmbeddingBagsF32Impl<false>(embedding_ids, row_pointers,
                                          num_bags, nullptr, table_data,
                                          embedding_dim, output);
  } else {
    AccumulateEmbeddingBagsF32Impl<true>(embedding_ids, row_pointers,
                                         num_bags, weights, table_data,
                                         embedding_dim, output);
  }
}

#else

int VectorLengthBytes() { return 0; }

void AccumulateEmbeddingBagsF32(const int64_t*, const int32_t*, size_t,
                                const float*, const float*, size_t, float*) {
  std::abort();
}

#endif

}  // namespace xla_serving::data::cpu_embedding::internal::sve

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

#include "data/cpu_embedding/cc/embedding_bag.h"  // from @xla_serving

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <utility>
#include <vector>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#endif

#include "absl/base/optimization.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/status_macros.h"
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/synchronization/blocking_counter.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"
#include "unsupported/Eigen/CXX11/ThreadPool"
#include "data/cpu_embedding/cc/array.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_bag_f32_sve.h"  // from @xla_serving
#include "data/cpu_embedding/cc/embedding_table.h"  // from @xla_serving

namespace xla_serving::data::cpu_embedding::internal {

namespace {

bool UsesInt32Ids(const EmbeddingBagInput& request) {
  return !request.embedding_ids_i32.empty() ||
         (request.embedding_ids.empty() &&
          request.embedding_ids_i32.data() != nullptr);
}

absl::StatusOr<EmbeddingBagBackend> ResolveBackend(
    const EmbeddingBagInput& request, EmbeddingBagBackend requested) {
  switch (requested) {
    case EmbeddingBagBackend::kAuto: {
      const EmbeddingTable& table = *request.embedding_table;
      if (table.EmbeddingDim() == 128 && request.weights.empty() &&
          table.GetCombineMode() == EmbeddingTable::CombineMode::kSum &&
          IsEmbeddingBagSveAvailable()) {
        // Defer kAuto because SVE vector length is thread-local.
        return EmbeddingBagBackend::kAuto;
      }
      return EmbeddingBagBackend::kPortable;
    }
    case EmbeddingBagBackend::kPortable:
      return EmbeddingBagBackend::kPortable;
    case EmbeddingBagBackend::kSve:
      if (!IsEmbeddingBagSveAvailable()) {
        return absl::FailedPreconditionError(
            "The SVE embedding-bag backend is not compiled or unavailable "
            "at runtime.");
      }
      return EmbeddingBagBackend::kSve;
  }
  return absl::InvalidArgumentError("Unknown embedding-bag backend.");
}

bool IsStandardF32Bag(const EmbeddingTable& table) {
  if (table.GetDType() != DType::kFloat32 ||
      table.GetCustomCombiner() != nullptr || table.EmbeddingDim() <= 0 ||
      table.VocabSize() < 0) {
    return false;
  }
  switch (table.GetCombineMode()) {
    case EmbeddingTable::CombineMode::kSum:
    case EmbeddingTable::CombineMode::kMean:
    case EmbeddingTable::CombineMode::kSqrtN:
      return true;
  }
  return false;
}

template <typename IdType>
absl::Status ValidateIdsBounds(const IdType* ids, size_t first, size_t last,
                               int64_t vocab_size) {
  const uint64_t vocab_u64 = static_cast<uint64_t>(vocab_size);
  uint64_t any_oob = 0;
  for (size_t index = first; index < last; ++index) {
    any_oob |=
        (static_cast<uint64_t>(static_cast<int64_t>(ids[index])) >= vocab_u64);
  }
  if (ABSL_PREDICT_FALSE(any_oob != 0)) {
    for (size_t index = first; index < last; ++index) {
      const int64_t id = static_cast<int64_t>(ids[index]);
      if (id < 0 || id >= vocab_size) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "Embedding ID (%d) at index %d is out of bounds [0, %d).", id,
            index, vocab_size));
      }
    }
  }
  return absl::OkStatus();
}

absl::StatusOr<size_t> ValidateEmbeddingBagInput(
    const EmbeddingBagInput& request) {
  if (request.embedding_table == nullptr) {
    return absl::InvalidArgumentError("Embedding table must be non-null.");
  }
  if (!IsStandardF32Bag(*request.embedding_table)) {
    return absl::FailedPreconditionError(
        "Grouped execution requires a non-sequence FP32 SUM, MEAN, or SQRT_N "
        "table without a custom combiner.");
  }
  if (request.row_pointers.size() < 2 ||
      request.row_pointers.data() == nullptr) {
    return absl::InvalidArgumentError(
        "Row pointers must contain at least two elements.");
  }
  const int32_t* rp = request.row_pointers.data();
  const size_t rp_size = request.row_pointers.size();
  if (rp[0] < 0) {
    return absl::InvalidArgumentError(
        "Row pointers must be non-negative and monotonic.");
  }
  int32_t non_monotonic = 0;
  for (size_t i = 0; i + 1 < rp_size; ++i) {
    non_monotonic |= (rp[i] > rp[i + 1]);
  }
  if (ABSL_PREDICT_FALSE(non_monotonic != 0)) {
    return absl::InvalidArgumentError(
        "Row pointers must be non-negative and monotonic.");
  }
  const size_t first = static_cast<size_t>(rp[0]);
  const size_t last = static_cast<size_t>(rp[rp_size - 1]);
  if (!request.embedding_ids.empty() && !request.embedding_ids_i32.empty()) {
    return absl::InvalidArgumentError(
        "EmbeddingBagInput cannot specify both int64 and int32 embedding IDs.");
  }
  const bool use_i32 = UsesInt32Ids(request);
  const size_t ids_size =
      use_i32 ? request.embedding_ids_i32.size() : request.embedding_ids.size();
  const void* ids_ptr =
      use_i32 ? static_cast<const void*>(request.embedding_ids_i32.data())
              : static_cast<const void*>(request.embedding_ids.data());
  if (last > ids_size) {
    return absl::InvalidArgumentError(
        "Final row pointer exceeds embedding IDs size.");
  }
  if (!request.weights.empty() && last > request.weights.size()) {
    return absl::InvalidArgumentError(
        "Final row pointer exceeds weights size.");
  }
  if (first != last && ids_ptr == nullptr) {
    return absl::InvalidArgumentError("Embedding-ID storage must be non-null.");
  }
  if (!request.weights.empty() && request.weights.data() == nullptr) {
    return absl::InvalidArgumentError("Weight storage must be non-null.");
  }
  const int64_t vocab_size = request.embedding_table->VocabSize();
  if (use_i32) {
    RETURN_IF_ERROR(ValidateIdsBounds(request.embedding_ids_i32.data(), first,
                                      last, vocab_size));
  } else {
    RETURN_IF_ERROR(ValidateIdsBounds(request.embedding_ids.data(), first, last,
                                      vocab_size));
  }
  const size_t dim = request.embedding_table->EmbeddingDim();
  const size_t num_bags = rp_size - 1;
  if (num_bags > std::numeric_limits<size_t>::max() / dim) {
    return absl::InvalidArgumentError("Embedding output size overflows.");
  }
  const size_t rows = static_cast<size_t>(vocab_size);
  if (rows > std::numeric_limits<size_t>::max() / dim) {
    return absl::InvalidArgumentError("Embedding table size overflows.");
  }
  return num_bags * dim;
}

float BagScale(const EmbeddingBagInput& request, size_t bag) {
  const size_t begin = request.row_pointers[bag];
  const size_t end = request.row_pointers[bag + 1];
  const size_t size = end - begin;
  const auto mode = request.embedding_table->GetCombineMode();
  if (size == 0 || mode == EmbeddingTable::CombineMode::kSum) return 1.0f;
  if (request.weights.empty()) {
    return mode == EmbeddingTable::CombineMode::kMean
               ? 1.0f / static_cast<float>(size)
               : 1.0f / std::sqrt(static_cast<float>(size));
  }

  float denominator = 0.0f;
  for (size_t position = begin; position < end; ++position) {
    const float weight = request.weights[position];
    denominator +=
        mode == EmbeddingTable::CombineMode::kMean ? weight : weight * weight;
  }
  return mode == EmbeddingTable::CombineMode::kMean
             ? 1.0f / denominator
             : 1.0f / std::sqrt(denominator);
}

#if defined(__x86_64__) || defined(_M_X64)

bool HasAvx2() {
  static const bool has_avx2 = __builtin_cpu_supports("avx2");
  return has_avx2;
}

__attribute__((target("avx2"))) inline void PrefetchRow128(const float* row) {
  _mm_prefetch(reinterpret_cast<const char*>(row), _MM_HINT_T0);
  _mm_prefetch(reinterpret_cast<const char*>(row + 64), _MM_HINT_T0);
}

__attribute__((target("avx2"))) inline void PrefetchRow64(const float* row) {
  _mm_prefetch(reinterpret_cast<const char*>(row), _MM_HINT_T0);
}

#define NEPTUNE_AVX2_ADD_ROW_64(r)                          \
  do {                                                      \
    acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps((r) + 0));   \
    acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps((r) + 8));   \
    acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps((r) + 16));  \
    acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps((r) + 24));  \
    acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps((r) + 32));  \
    acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps((r) + 40));  \
    acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps((r) + 48));  \
    acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps((r) + 56));  \
  } while (0)

template <typename IdType>
__attribute__((target("avx2"))) void ReducePortableAvx2Dim64Unweighted(
    const EmbeddingBagInput& request, const IdType* ids,
    absl::Span<float> output_buffer, size_t bag_begin, size_t bag_end) {
  constexpr size_t kDim = 64;
  const float* table = request.embedding_table->AsTensor<float>().data();
  const int32_t* rp = request.row_pointers.data();
  const auto mode = request.embedding_table->GetCombineMode();
  float* out_base = output_buffer.data();
  const size_t total_ids_end = static_cast<size_t>(rp[bag_end]);
  const bool do_prefetch =
      request.embedding_table->VocabSize() > 1024;  // > 256 KB (L2 cache)

  for (size_t bag = bag_begin; bag < bag_end; ++bag) {
    float* out_ptr = out_base + bag * kDim;
    const size_t begin = static_cast<size_t>(rp[bag]);
    const size_t end = static_cast<size_t>(rp[bag + 1]);
    const size_t count = end - begin;

    if (ABSL_PREDICT_FALSE(count == 0)) {
      std::memset(out_ptr, 0, kDim * sizeof(float));
      continue;
    }

    if (count == 1) {
      if (do_prefetch && begin + 3 < total_ids_end) {
        PrefetchRow64(table + static_cast<size_t>(ids[begin + 3]) * kDim);
      }
      const float* r0 = table + static_cast<size_t>(ids[begin]) * kDim;
      std::memcpy(out_ptr, r0, kDim * sizeof(float));
      continue;
    }

    if (do_prefetch && count <= 6 && end < total_ids_end) {
      const size_t prefetch_end = std::min(total_ids_end, end + 3);
      for (size_t p = end; p < prefetch_end; ++p) {
        PrefetchRow64(table + static_cast<size_t>(ids[p]) * kDim);
      }
    }

    const float* r0 = table + static_cast<size_t>(ids[begin]) * kDim;
    const float* r1 = table + static_cast<size_t>(ids[begin + 1]) * kDim;
    __m256 acc0 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 0), _mm256_loadu_ps(r1 + 0));
    __m256 acc1 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 8), _mm256_loadu_ps(r1 + 8));
    __m256 acc2 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 16), _mm256_loadu_ps(r1 + 16));
    __m256 acc3 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 24), _mm256_loadu_ps(r1 + 24));
    __m256 acc4 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 32), _mm256_loadu_ps(r1 + 32));
    __m256 acc5 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 40), _mm256_loadu_ps(r1 + 40));
    __m256 acc6 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 48), _mm256_loadu_ps(r1 + 48));
    __m256 acc7 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 56), _mm256_loadu_ps(r1 + 56));

    size_t pos = begin + 2;
    for (; pos + 8 <= end; pos += 8) {
      const float* r_0 = table + static_cast<size_t>(ids[pos + 0]) * kDim;
      const float* r_1 = table + static_cast<size_t>(ids[pos + 1]) * kDim;
      const float* r_2 = table + static_cast<size_t>(ids[pos + 2]) * kDim;
      const float* r_3 = table + static_cast<size_t>(ids[pos + 3]) * kDim;
      const float* r_4 = table + static_cast<size_t>(ids[pos + 4]) * kDim;
      const float* r_5 = table + static_cast<size_t>(ids[pos + 5]) * kDim;
      const float* r_6 = table + static_cast<size_t>(ids[pos + 6]) * kDim;
      const float* r_7 = table + static_cast<size_t>(ids[pos + 7]) * kDim;
      if (do_prefetch) {
        PrefetchRow64(r_4);
        PrefetchRow64(r_5);
        PrefetchRow64(r_6);
        PrefetchRow64(r_7);
        if (pos + 12 <= total_ids_end) {
          PrefetchRow64(table + static_cast<size_t>(ids[pos + 8]) * kDim);
          PrefetchRow64(table + static_cast<size_t>(ids[pos + 9]) * kDim);
          PrefetchRow64(table + static_cast<size_t>(ids[pos + 10]) * kDim);
          PrefetchRow64(table + static_cast<size_t>(ids[pos + 11]) * kDim);
        }
      }
      NEPTUNE_AVX2_ADD_ROW_64(r_0);
      NEPTUNE_AVX2_ADD_ROW_64(r_1);
      NEPTUNE_AVX2_ADD_ROW_64(r_2);
      NEPTUNE_AVX2_ADD_ROW_64(r_3);
      NEPTUNE_AVX2_ADD_ROW_64(r_4);
      NEPTUNE_AVX2_ADD_ROW_64(r_5);
      NEPTUNE_AVX2_ADD_ROW_64(r_6);
      NEPTUNE_AVX2_ADD_ROW_64(r_7);
    }
    for (; pos + 4 <= end; pos += 4) {
      const float* r_0 = table + static_cast<size_t>(ids[pos + 0]) * kDim;
      const float* r_1 = table + static_cast<size_t>(ids[pos + 1]) * kDim;
      const float* r_2 = table + static_cast<size_t>(ids[pos + 2]) * kDim;
      const float* r_3 = table + static_cast<size_t>(ids[pos + 3]) * kDim;
      NEPTUNE_AVX2_ADD_ROW_64(r_0);
      NEPTUNE_AVX2_ADD_ROW_64(r_1);
      NEPTUNE_AVX2_ADD_ROW_64(r_2);
      NEPTUNE_AVX2_ADD_ROW_64(r_3);
    }
    for (; pos < end; ++pos) {
      const float* r_0 = table + static_cast<size_t>(ids[pos]) * kDim;
      NEPTUNE_AVX2_ADD_ROW_64(r_0);
    }

    if (mode != EmbeddingTable::CombineMode::kSum) {
      const __m256 vscale = _mm256_set1_ps(BagScale(request, bag));
      acc0 = _mm256_mul_ps(acc0, vscale);
      acc1 = _mm256_mul_ps(acc1, vscale);
      acc2 = _mm256_mul_ps(acc2, vscale);
      acc3 = _mm256_mul_ps(acc3, vscale);
      acc4 = _mm256_mul_ps(acc4, vscale);
      acc5 = _mm256_mul_ps(acc5, vscale);
      acc6 = _mm256_mul_ps(acc6, vscale);
      acc7 = _mm256_mul_ps(acc7, vscale);
    }

    _mm256_storeu_ps(out_ptr + 0, acc0);
    _mm256_storeu_ps(out_ptr + 8, acc1);
    _mm256_storeu_ps(out_ptr + 16, acc2);
    _mm256_storeu_ps(out_ptr + 24, acc3);
    _mm256_storeu_ps(out_ptr + 32, acc4);
    _mm256_storeu_ps(out_ptr + 40, acc5);
    _mm256_storeu_ps(out_ptr + 48, acc6);
    _mm256_storeu_ps(out_ptr + 56, acc7);
  }
}

#undef NEPTUNE_AVX2_ADD_ROW_64

#define NEPTUNE_AVX2_ADD_ROW_128(r)                           \
  do {                                                        \
    acc0 = _mm256_add_ps(acc0, _mm256_loadu_ps((r) + 0));     \
    acc1 = _mm256_add_ps(acc1, _mm256_loadu_ps((r) + 8));     \
    acc2 = _mm256_add_ps(acc2, _mm256_loadu_ps((r) + 16));    \
    acc3 = _mm256_add_ps(acc3, _mm256_loadu_ps((r) + 24));    \
    acc4 = _mm256_add_ps(acc4, _mm256_loadu_ps((r) + 32));    \
    acc5 = _mm256_add_ps(acc5, _mm256_loadu_ps((r) + 40));    \
    acc6 = _mm256_add_ps(acc6, _mm256_loadu_ps((r) + 48));    \
    acc7 = _mm256_add_ps(acc7, _mm256_loadu_ps((r) + 56));    \
    acc8 = _mm256_add_ps(acc8, _mm256_loadu_ps((r) + 64));    \
    acc9 = _mm256_add_ps(acc9, _mm256_loadu_ps((r) + 72));    \
    acc10 = _mm256_add_ps(acc10, _mm256_loadu_ps((r) + 80));  \
    acc11 = _mm256_add_ps(acc11, _mm256_loadu_ps((r) + 88));  \
    acc12 = _mm256_add_ps(acc12, _mm256_loadu_ps((r) + 96));  \
    acc13 = _mm256_add_ps(acc13, _mm256_loadu_ps((r) + 104)); \
    acc14 = _mm256_add_ps(acc14, _mm256_loadu_ps((r) + 112)); \
    acc15 = _mm256_add_ps(acc15, _mm256_loadu_ps((r) + 120)); \
  } while (0)

template <typename IdType>
__attribute__((target("avx2"))) void ReducePortableAvx2Dim128Unweighted(
    const EmbeddingBagInput& request, const IdType* ids,
    absl::Span<float> output_buffer, size_t bag_begin, size_t bag_end) {
  constexpr size_t kDim = 128;
  const float* table = request.embedding_table->AsTensor<float>().data();
  const int32_t* rp = request.row_pointers.data();
  const auto mode = request.embedding_table->GetCombineMode();
  float* out_base = output_buffer.data();
  const size_t total_ids_end = static_cast<size_t>(rp[bag_end]);
  const bool do_prefetch =
      request.embedding_table->VocabSize() > 512;  // > 256 KB (L2 cache)

  for (size_t bag = bag_begin; bag < bag_end; ++bag) {
    float* out_ptr = out_base + bag * kDim;
    const size_t begin = static_cast<size_t>(rp[bag]);
    const size_t end = static_cast<size_t>(rp[bag + 1]);
    const size_t count = end - begin;

    if (ABSL_PREDICT_FALSE(count == 0)) {
      std::memset(out_ptr, 0, kDim * sizeof(float));
      continue;
    }

    if (count == 1) {
      if (do_prefetch && begin + 3 < total_ids_end) {
        PrefetchRow128(table + static_cast<size_t>(ids[begin + 3]) * kDim);
      }
      const float* r0 = table + static_cast<size_t>(ids[begin]) * kDim;
      std::memcpy(out_ptr, r0, kDim * sizeof(float));
      continue;
    }

    if (do_prefetch && count <= 6 && end < total_ids_end) {
      const size_t prefetch_end = std::min(total_ids_end, end + 3);
      for (size_t p = end; p < prefetch_end; ++p) {
        PrefetchRow128(table + static_cast<size_t>(ids[p]) * kDim);
      }
    }

    const float* r0 = table + static_cast<size_t>(ids[begin]) * kDim;
    const float* r1 = table + static_cast<size_t>(ids[begin + 1]) * kDim;
    __m256 acc0 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 0), _mm256_loadu_ps(r1 + 0));
    __m256 acc1 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 8), _mm256_loadu_ps(r1 + 8));
    __m256 acc2 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 16), _mm256_loadu_ps(r1 + 16));
    __m256 acc3 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 24), _mm256_loadu_ps(r1 + 24));
    __m256 acc4 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 32), _mm256_loadu_ps(r1 + 32));
    __m256 acc5 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 40), _mm256_loadu_ps(r1 + 40));
    __m256 acc6 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 48), _mm256_loadu_ps(r1 + 48));
    __m256 acc7 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 56), _mm256_loadu_ps(r1 + 56));
    __m256 acc8 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 64), _mm256_loadu_ps(r1 + 64));
    __m256 acc9 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 72), _mm256_loadu_ps(r1 + 72));
    __m256 acc10 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 80), _mm256_loadu_ps(r1 + 80));
    __m256 acc11 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 88), _mm256_loadu_ps(r1 + 88));
    __m256 acc12 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 96), _mm256_loadu_ps(r1 + 96));
    __m256 acc13 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 104), _mm256_loadu_ps(r1 + 104));
    __m256 acc14 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 112), _mm256_loadu_ps(r1 + 112));
    __m256 acc15 =
        _mm256_add_ps(_mm256_loadu_ps(r0 + 120), _mm256_loadu_ps(r1 + 120));

    size_t pos = begin + 2;
    for (; pos + 8 <= end; pos += 8) {
      const float* r_0 = table + static_cast<size_t>(ids[pos + 0]) * kDim;
      const float* r_1 = table + static_cast<size_t>(ids[pos + 1]) * kDim;
      const float* r_2 = table + static_cast<size_t>(ids[pos + 2]) * kDim;
      const float* r_3 = table + static_cast<size_t>(ids[pos + 3]) * kDim;
      const float* r_4 = table + static_cast<size_t>(ids[pos + 4]) * kDim;
      const float* r_5 = table + static_cast<size_t>(ids[pos + 5]) * kDim;
      const float* r_6 = table + static_cast<size_t>(ids[pos + 6]) * kDim;
      const float* r_7 = table + static_cast<size_t>(ids[pos + 7]) * kDim;
      if (do_prefetch) {
        PrefetchRow128(r_4);
        PrefetchRow128(r_5);
        PrefetchRow128(r_6);
        PrefetchRow128(r_7);
        if (pos + 12 <= total_ids_end) {
          PrefetchRow128(table + static_cast<size_t>(ids[pos + 8]) * kDim);
          PrefetchRow128(table + static_cast<size_t>(ids[pos + 9]) * kDim);
          PrefetchRow128(table + static_cast<size_t>(ids[pos + 10]) * kDim);
          PrefetchRow128(table + static_cast<size_t>(ids[pos + 11]) * kDim);
        }
      }
      NEPTUNE_AVX2_ADD_ROW_128(r_0);
      NEPTUNE_AVX2_ADD_ROW_128(r_1);
      NEPTUNE_AVX2_ADD_ROW_128(r_2);
      NEPTUNE_AVX2_ADD_ROW_128(r_3);
      NEPTUNE_AVX2_ADD_ROW_128(r_4);
      NEPTUNE_AVX2_ADD_ROW_128(r_5);
      NEPTUNE_AVX2_ADD_ROW_128(r_6);
      NEPTUNE_AVX2_ADD_ROW_128(r_7);
    }
    for (; pos + 4 <= end; pos += 4) {
      const float* r_0 = table + static_cast<size_t>(ids[pos + 0]) * kDim;
      const float* r_1 = table + static_cast<size_t>(ids[pos + 1]) * kDim;
      const float* r_2 = table + static_cast<size_t>(ids[pos + 2]) * kDim;
      const float* r_3 = table + static_cast<size_t>(ids[pos + 3]) * kDim;
      NEPTUNE_AVX2_ADD_ROW_128(r_0);
      NEPTUNE_AVX2_ADD_ROW_128(r_1);
      NEPTUNE_AVX2_ADD_ROW_128(r_2);
      NEPTUNE_AVX2_ADD_ROW_128(r_3);
    }
    for (; pos < end; ++pos) {
      const float* r_0 = table + static_cast<size_t>(ids[pos]) * kDim;
      NEPTUNE_AVX2_ADD_ROW_128(r_0);
    }

    if (mode != EmbeddingTable::CombineMode::kSum) {
      const __m256 vscale = _mm256_set1_ps(BagScale(request, bag));
      acc0 = _mm256_mul_ps(acc0, vscale);
      acc1 = _mm256_mul_ps(acc1, vscale);
      acc2 = _mm256_mul_ps(acc2, vscale);
      acc3 = _mm256_mul_ps(acc3, vscale);
      acc4 = _mm256_mul_ps(acc4, vscale);
      acc5 = _mm256_mul_ps(acc5, vscale);
      acc6 = _mm256_mul_ps(acc6, vscale);
      acc7 = _mm256_mul_ps(acc7, vscale);
      acc8 = _mm256_mul_ps(acc8, vscale);
      acc9 = _mm256_mul_ps(acc9, vscale);
      acc10 = _mm256_mul_ps(acc10, vscale);
      acc11 = _mm256_mul_ps(acc11, vscale);
      acc12 = _mm256_mul_ps(acc12, vscale);
      acc13 = _mm256_mul_ps(acc13, vscale);
      acc14 = _mm256_mul_ps(acc14, vscale);
      acc15 = _mm256_mul_ps(acc15, vscale);
    }

    _mm256_storeu_ps(out_ptr + 0, acc0);
    _mm256_storeu_ps(out_ptr + 8, acc1);
    _mm256_storeu_ps(out_ptr + 16, acc2);
    _mm256_storeu_ps(out_ptr + 24, acc3);
    _mm256_storeu_ps(out_ptr + 32, acc4);
    _mm256_storeu_ps(out_ptr + 40, acc5);
    _mm256_storeu_ps(out_ptr + 48, acc6);
    _mm256_storeu_ps(out_ptr + 56, acc7);
    _mm256_storeu_ps(out_ptr + 64, acc8);
    _mm256_storeu_ps(out_ptr + 72, acc9);
    _mm256_storeu_ps(out_ptr + 80, acc10);
    _mm256_storeu_ps(out_ptr + 88, acc11);
    _mm256_storeu_ps(out_ptr + 96, acc12);
    _mm256_storeu_ps(out_ptr + 104, acc13);
    _mm256_storeu_ps(out_ptr + 112, acc14);
    _mm256_storeu_ps(out_ptr + 120, acc15);
  }
}

#undef NEPTUNE_AVX2_ADD_ROW_128

#endif  // defined(__x86_64__) || defined(_M_X64)

template <typename IdType>
void ReducePortableFallback(const EmbeddingBagInput& request, const IdType* ids,
                            absl::Span<float> output_buffer, size_t bag_begin,
                            size_t bag_end) {
  const size_t dim = request.embedding_table->EmbeddingDim();
  const float* table = request.embedding_table->AsTensor<float>().data();
  const int32_t* rp = request.row_pointers.data();
  const auto mode = request.embedding_table->GetCombineMode();
  const bool unweighted = request.weights.empty();
  const float* weights = unweighted ? nullptr : request.weights.data();

  for (size_t bag = bag_begin; bag < bag_end; ++bag) {
    float* out_ptr = output_buffer.data() + bag * dim;
    const size_t begin = static_cast<size_t>(rp[bag]);
    const size_t end = static_cast<size_t>(rp[bag + 1]);
    const size_t count = end - begin;

    if (count == 0) {
      std::memset(out_ptr, 0, dim * sizeof(float));
      continue;
    }

    Eigen::Map<Eigen::VectorXf> output(out_ptr, dim);
    if (unweighted) {
      if (count == 1) {
        const float* r0 = table + static_cast<size_t>(ids[begin]) * dim;
        std::memcpy(out_ptr, r0, dim * sizeof(float));
        continue;
      }
      const float* r0 = table + static_cast<size_t>(ids[begin]) * dim;
      const float* r1 = table + static_cast<size_t>(ids[begin + 1]) * dim;
      output.noalias() = Eigen::Map<const Eigen::VectorXf>(r0, dim) +
                         Eigen::Map<const Eigen::VectorXf>(r1, dim);
      size_t pos = begin + 2;
      for (; pos + 4 <= end; pos += 4) {
        const float* r_0 = table + static_cast<size_t>(ids[pos + 0]) * dim;
        const float* r_1 = table + static_cast<size_t>(ids[pos + 1]) * dim;
        const float* r_2 = table + static_cast<size_t>(ids[pos + 2]) * dim;
        const float* r_3 = table + static_cast<size_t>(ids[pos + 3]) * dim;
        output += Eigen::Map<const Eigen::VectorXf>(r_0, dim);
        output += Eigen::Map<const Eigen::VectorXf>(r_1, dim);
        output += Eigen::Map<const Eigen::VectorXf>(r_2, dim);
        output += Eigen::Map<const Eigen::VectorXf>(r_3, dim);
      }
      for (; pos < end; ++pos) {
        const float* r_0 = table + static_cast<size_t>(ids[pos]) * dim;
        output += Eigen::Map<const Eigen::VectorXf>(r_0, dim);
      }
    } else {
      output.setZero();
      for (size_t position = begin; position < end; ++position) {
        Eigen::Map<const Eigen::VectorXf> row(
            table + static_cast<size_t>(ids[position]) * dim, dim);
        output += weights[position] * row;
      }
    }

    if (mode != EmbeddingTable::CombineMode::kSum) {
      output *= BagScale(request, bag);
    }
  }
}

template <typename IdType>
void ReducePortableDispatch(const EmbeddingBagInput& request, const IdType* ids,
                            absl::Span<float> output_buffer, size_t bag_begin,
                            size_t bag_end) {
#if defined(__x86_64__) || defined(_M_X64)
  if (request.weights.empty() && HasAvx2()) {
    const size_t dim = request.embedding_table->EmbeddingDim();
    if (dim == 128) {
      ReducePortableAvx2Dim128Unweighted(request, ids, output_buffer, bag_begin,
                                         bag_end);
      return;
    }
    if (dim == 64) {
      ReducePortableAvx2Dim64Unweighted(request, ids, output_buffer, bag_begin,
                                        bag_end);
      return;
    }
  }
#endif
  ReducePortableFallback(request, ids, output_buffer, bag_begin, bag_end);
}

// Inputs are validated before worker tasks run.
void ReducePortable(const EmbeddingBagInput& request,
                    absl::Span<float> output_buffer, size_t bag_begin,
                    size_t bag_end) {
  if (UsesInt32Ids(request)) {
    ReducePortableDispatch(request, request.embedding_ids_i32.data(),
                           output_buffer, bag_begin, bag_end);
  } else {
    ReducePortableDispatch(request, request.embedding_ids.data(), output_buffer,
                           bag_begin, bag_end);
  }
}

void ReduceSve(const EmbeddingBagInput& request,
               absl::Span<float> output_buffer, size_t bag_begin,
               size_t bag_end) {
  const size_t dim = request.embedding_table->EmbeddingDim();
  absl::Span<float> output =
      output_buffer.subspan(bag_begin * dim, (bag_end - bag_begin) * dim);
  // CSR offsets are absolute; shift only the row-pointer base.
  std::fill(output.begin(), output.end(), 0.0f);
  if (UsesInt32Ids(request)) {
    sve::AccumulateEmbeddingBagsF32(
        request.embedding_ids_i32.data(),
        request.row_pointers.data() + bag_begin, bag_end - bag_begin,
        request.weights.empty() ? nullptr : request.weights.data(),
        request.embedding_table->AsTensor<float>().data(), dim, output.data());
  } else {
    sve::AccumulateEmbeddingBagsF32(
        request.embedding_ids.data(), request.row_pointers.data() + bag_begin,
        bag_end - bag_begin,
        request.weights.empty() ? nullptr : request.weights.data(),
        request.embedding_table->AsTensor<float>().data(), dim, output.data());
  }
  if (request.embedding_table->GetCombineMode() ==
      EmbeddingTable::CombineMode::kSum) {
    return;
  }
  for (size_t bag = bag_begin; bag < bag_end; ++bag) {
    if (request.row_pointers[bag] == request.row_pointers[bag + 1]) continue;
    Eigen::Map<Eigen::VectorXf> bag_output(output_buffer.data() + bag * dim,
                                           dim);
    bag_output *= BagScale(request, bag);
  }
}

struct PreparedRequest {
  const EmbeddingBagInput* request;
  EmbeddingBagBackend backend;
  size_t table_elements;
  size_t output_size;
  absl::Span<float> output;
};

absl::Status RequestError(size_t index, const absl::Status& status) {
  return absl::Status(
      status.code(), absl::StrFormat("FP32 embedding-bag request %d: %s", index,
                                     status.message()));
}

absl::StatusOr<PreparedRequest> PrepareRequest(const EmbeddingBagInput& request,
                                               size_t index,
                                               EmbeddingBagBackend requested) {
  if (request.output_allocator == nullptr || !*request.output_allocator) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FP32 embedding-bag request %d requires an output allocator.", index));
  }
  absl::StatusOr<size_t> output_size = ValidateEmbeddingBagInput(request);
  if (!output_size.ok()) return RequestError(index, output_size.status());
  absl::StatusOr<EmbeddingBagBackend> backend =
      ResolveBackend(request, requested);
  if (!backend.ok()) return RequestError(index, backend.status());
  const size_t table_elements =
      static_cast<size_t>(request.embedding_table->VocabSize()) *
      static_cast<size_t>(request.embedding_table->EmbeddingDim());
  return PreparedRequest{&request, *backend, table_elements, *output_size, {}};
}

absl::Status AllocateOutputs(absl::Span<PreparedRequest> requests) {
  for (size_t index = 0; index < requests.size(); ++index) {
    PreparedRequest& prepared = requests[index];
    prepared.output =
        (*prepared.request->output_allocator)(prepared.output_size);
    if (prepared.output.size() != prepared.output_size) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "FP32 embedding-bag request %d allocator returned %d elements; "
          "expected %d.",
          index, prepared.output.size(), prepared.output_size));
    }
  }
  return absl::OkStatus();
}

struct AddressRange {
  uintptr_t begin;
  uintptr_t end;
  size_t request_index;
  const char* kind;
};

template <typename VectorType>
absl::Status AppendRange(const void* data, size_t elements, size_t element_size,
                         size_t request_index, const char* kind,
                         VectorType* ranges) {
  if (elements == 0) return absl::OkStatus();
  if (data == nullptr) {
    return absl::InvalidArgumentError(
        absl::StrFormat("FP32 embedding-bag request %d has null %s storage.",
                         request_index, kind));
  }
  if (elements > std::numeric_limits<uintptr_t>::max() / element_size) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FP32 embedding-bag request %d %s size overflows uintptr_t.",
        request_index, kind));
  }
  const uintptr_t begin = reinterpret_cast<uintptr_t>(data);
  const uintptr_t bytes = elements * element_size;
  if (begin > std::numeric_limits<uintptr_t>::max() - bytes) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "FP32 embedding-bag request %d %s range overflows uintptr_t.",
        request_index, kind));
  }
  ranges->push_back({begin, begin + bytes, request_index, kind});
  return absl::OkStatus();
}

bool Overlaps(const AddressRange& lhs, const AddressRange& rhs) {
  return lhs.begin < rhs.end && rhs.begin < lhs.end;
}

absl::Status ValidateOutputIsolation(
    absl::Span<const PreparedRequest> requests) {
  absl::InlinedVector<AddressRange, 32> outputs;
  absl::InlinedVector<AddressRange, 128> inputs;
  outputs.reserve(requests.size());
  inputs.reserve(requests.size() * 4);
  for (size_t index = 0; index < requests.size(); ++index) {
    const PreparedRequest& prepared = requests[index];
    const EmbeddingBagInput& request = *prepared.request;
    RETURN_IF_ERROR(AppendRange(prepared.output.data(), prepared.output.size(),
                                sizeof(float), index, "output", &outputs));
    if (UsesInt32Ids(request)) {
      RETURN_IF_ERROR(AppendRange(request.embedding_ids_i32.data(),
                                  request.embedding_ids_i32.size(),
                                  sizeof(int32_t), index, "embedding-ID",
                                  &inputs));
    } else {
      RETURN_IF_ERROR(AppendRange(request.embedding_ids.data(),
                                  request.embedding_ids.size(), sizeof(int64_t),
                                  index, "embedding-ID", &inputs));
    }
    RETURN_IF_ERROR(AppendRange(request.row_pointers.data(),
                                request.row_pointers.size(), sizeof(int32_t),
                                index, "row-pointer", &inputs));
    RETURN_IF_ERROR(AppendRange(request.weights.data(), request.weights.size(),
                                sizeof(float), index, "weight", &inputs));
    RETURN_IF_ERROR(
        AppendRange(request.embedding_table->AsTensor<float>().data(),
                    prepared.table_elements, sizeof(float), index,
                    "embedding-table", &inputs));
  }

  std::sort(outputs.begin(), outputs.end(),
            [](const AddressRange& lhs, const AddressRange& rhs) {
              return lhs.begin < rhs.begin;
            });
  for (size_t index = 1; index < outputs.size(); ++index) {
    if (Overlaps(outputs[index - 1], outputs[index])) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "FP32 embedding-bag request outputs %d and %d overlap.",
          outputs[index - 1].request_index, outputs[index].request_index));
    }
  }
  for (const AddressRange& output : outputs) {
    for (const AddressRange& input : inputs) {
      if (Overlaps(output, input)) {
        return absl::InvalidArgumentError(absl::StrFormat(
            "FP32 embedding-bag request %d output overlaps request %d %s "
            "storage.",
            output.request_index, input.request_index, input.kind));
      }
    }
  }
  return absl::OkStatus();
}

struct Task {
  size_t request_index;
  size_t bag_begin;
  size_t bag_end;
};

absl::StatusOr<uint64_t> TaskCost(const EmbeddingBagInput& request,
                                  size_t bag_begin, size_t bag_end) {
  const uint64_t ids = static_cast<uint64_t>(request.row_pointers[bag_end]) -
                       request.row_pointers[bag_begin];
  const uint64_t bags = bag_end - bag_begin;
  if (ids > std::numeric_limits<uint64_t>::max() - bags) {
    return absl::InvalidArgumentError("Embedding-bag task cost overflows.");
  }
  const uint64_t base = ids + bags;
  const uint64_t dim = request.embedding_table->EmbeddingDim();
  if (base > std::numeric_limits<uint64_t>::max() / dim) {
    return absl::InvalidArgumentError("Embedding-bag task cost overflows.");
  }
  return base * dim;
}

using TaskList = absl::InlinedVector<Task, 128>;

absl::StatusOr<TaskList> BuildTasks(absl::Span<const PreparedRequest> requests,
                                    int pool_threads) {
  // Aim for four bag-range tasks per worker without splitting individual bags.
  constexpr uint64_t kTasksPerThread = 4;
  uint64_t total_cost = 0;
  absl::InlinedVector<uint64_t, 32> request_costs(requests.size());
  for (size_t i = 0; i < requests.size(); ++i) {
    const EmbeddingBagInput& request = *requests[i].request;
    ASSIGN_OR_RETURN(uint64_t cost,
                     TaskCost(request, 0, request.row_pointers.size() - 1));
    if (cost > std::numeric_limits<uint64_t>::max() - total_cost) {
      return absl::InvalidArgumentError("Embedding-bag batch cost overflows.");
    }
    request_costs[i] = cost;
    total_cost += cost;
  }
  const uint64_t target_count =
      static_cast<uint64_t>(pool_threads) * kTasksPerThread;
  const uint64_t target_cost = std::max<uint64_t>(
      1, total_cost / target_count + (total_cost % target_count != 0));

  constexpr size_t kLargeTableElements = 65536;  // 256 KB (L2 cache threshold)
  TaskList tasks;
  TaskList small_table_tasks;

  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const PreparedRequest& prepared = requests[request_index];
    const EmbeddingBagInput& request = *prepared.request;
    const size_t num_bags = request.row_pointers.size() - 1;
    const bool is_large_table = prepared.table_elements > kLargeTableElements;
    TaskList& dest = is_large_table ? tasks : small_table_tasks;

    const size_t max_bags_for_large =
        (is_large_table && num_bags >= 32)
            ? std::max<size_t>(16, num_bags / 4)
            : num_bags;

    if (num_bags <= 1 ||
        (request_costs[request_index] <= target_cost &&
         num_bags <= max_bags_for_large)) {
      dest.push_back({request_index, 0, num_bags});
      continue;
    }
    const int32_t* rp = request.row_pointers.data();
    const uint64_t dim = request.embedding_table->EmbeddingDim();
    const uint64_t target_base =
        target_cost / dim + (target_cost % dim != 0 ? 1 : 0);

    size_t tile_begin = 0;
    while (tile_begin + 1 < num_bags) {
      const size_t last_candidate =
          std::min(num_bags - 1, tile_begin + max_bags_for_large);
      const uint64_t max_base =
          static_cast<uint64_t>(rp[last_candidate] - rp[tile_begin]) +
          (last_candidate - tile_begin);
      if (max_base < target_base) {
        if (last_candidate < num_bags - 1) {
          dest.push_back({request_index, tile_begin, last_candidate});
          tile_begin = last_candidate;
          continue;
        }
        break;
      }
      // Binary search for the first bag in [tile_begin + 1, last_candidate]
      // where base(tile_begin, bag) >= target_base.
      size_t lo = tile_begin + 1;
      size_t hi = last_candidate;
      while (lo < hi) {
        const size_t mid = lo + (hi - lo) / 2;
        const uint64_t mid_base =
            static_cast<uint64_t>(rp[mid] - rp[tile_begin]) +
            (mid - tile_begin);
        if (mid_base >= target_base) {
          hi = mid;
        } else {
          lo = mid + 1;
        }
      }
      dest.push_back({request_index, tile_begin, lo});
      tile_begin = lo;
    }
    dest.push_back({request_index, tile_begin, num_bags});
  }
  tasks.insert(tasks.end(), small_table_tasks.begin(), small_table_tasks.end());
  return tasks;
}

EmbeddingBagBackend BackendForCurrentThread(const PreparedRequest& prepared) {
  if (prepared.backend != EmbeddingBagBackend::kAuto) return prepared.backend;
  return EmbeddingBagSveVectorLengthBytes() == kAutoSveVectorLengthBytes
             ? EmbeddingBagBackend::kSve
             : EmbeddingBagBackend::kPortable;
}

void ExecuteTask(absl::Span<const PreparedRequest> requests, const Task& task) {
  const PreparedRequest& prepared = requests[task.request_index];
  switch (BackendForCurrentThread(prepared)) {
    case EmbeddingBagBackend::kAuto:
    case EmbeddingBagBackend::kPortable:
      ReducePortable(*prepared.request, prepared.output, task.bag_begin,
                     task.bag_end);
      return;
    case EmbeddingBagBackend::kSve:
      ReduceSve(*prepared.request, prepared.output, task.bag_begin,
                task.bag_end);
      return;
  }
}

struct PoolConcurrencyState {
  std::atomic<int> active_callers{0};
  std::atomic<int> active_helpers{0};
};

PoolConcurrencyState& GetPoolState(const Eigen::ThreadPoolInterface* pool) {
  static PoolConcurrencyState states[16];
  const size_t slot = (reinterpret_cast<uintptr_t>(pool) >> 6) & 15;
  return states[slot];
}

struct alignas(64) BatchSlot {
  std::atomic<int> refs{0};
  std::atomic<size_t> next_task{0};
  size_t num_tasks = 0;
  const PreparedRequest* prepared_data = nullptr;
  size_t prepared_size = 0;
  const Task* tasks_data = nullptr;
  absl::BlockingCounter* tasks_counter = nullptr;
};

BatchSlot* AcquireBatchSlot(int initial_refs) {
  static constexpr size_t kNumSlots = 256;
  static BatchSlot slots[kNumSlots];
  static std::atomic<uint32_t> cursor{0};
  for (size_t attempt = 0; attempt < kNumSlots * 2; ++attempt) {
    const uint32_t idx =
        cursor.fetch_add(1, std::memory_order_relaxed) & (kNumSlots - 1);
    int expected = 0;
    if (slots[idx].refs.compare_exchange_strong(
            expected, initial_refs, std::memory_order_acq_rel,
            std::memory_order_relaxed)) {
      return &slots[idx];
    }
  }
  return nullptr;
}

bool IsEmbeddingBagSveCompiled() {
#if defined(XLA_SERVING_HAS_SVE_EMBEDDING_BAG)
  return true;
#else
  return false;
#endif
}

bool CpuSupportsSve() {
#if defined(__linux__) && defined(__aarch64__) && defined(HWCAP_SVE)
  static const bool available = (getauxval(AT_HWCAP) & HWCAP_SVE) != 0;
  return available;
#else
  return false;
#endif
}

}  // namespace

bool IsEmbeddingBagSveAvailable() {
  return IsEmbeddingBagSveCompiled() && CpuSupportsSve();
}

int EmbeddingBagSveVectorLengthBytes() {
  return IsEmbeddingBagSveAvailable() ? sve::VectorLengthBytes() : 0;
}

absl::Status ComputeEmbeddingActivationsF32(const EmbeddingBagInput& input,
                                            EmbeddingBagBackend requested,
                                            EmbeddingBagBackend* selected) {
  ASSIGN_OR_RETURN(PreparedRequest prepared,
                   PrepareRequest(input, 0, requested));
  RETURN_IF_ERROR(AllocateOutputs(absl::MakeSpan(&prepared, 1)));
  RETURN_IF_ERROR(ValidateOutputIsolation(absl::MakeConstSpan(&prepared, 1)));
  if (selected != nullptr) *selected = BackendForCurrentThread(prepared);
  ExecuteTask(absl::MakeConstSpan(&prepared, 1),
              Task{0, 0, input.row_pointers.size() - 1});
  return absl::OkStatus();
}

absl::Status ExecuteEmbeddingBagBatch(
    absl::Span<const EmbeddingBagInput> inputs, EmbeddingBagBackend requested,
    Eigen::ThreadPoolInterface* thread_pool, size_t inline_work_threshold) {
  if (inputs.empty()) {
    return absl::InvalidArgumentError(
        "Embedding-bag batch requires at least one request.");
  }
  if (thread_pool == nullptr || thread_pool->NumThreads() <= 0) {
    return absl::InvalidArgumentError(
        "Embedding-bag batch requires a non-empty thread pool.");
  }

  absl::InlinedVector<PreparedRequest, 32> prepared;
  prepared.reserve(inputs.size());
  for (size_t index = 0; index < inputs.size(); ++index) {
    ASSIGN_OR_RETURN(PreparedRequest request,
                     PrepareRequest(inputs[index], index, requested));
    prepared.push_back(std::move(request));
  }
  RETURN_IF_ERROR(AllocateOutputs(absl::MakeSpan(prepared)));
  RETURN_IF_ERROR(ValidateOutputIsolation(absl::MakeConstSpan(prepared)));

  size_t total_work_size = 0;
  for (const PreparedRequest& request : prepared) {
    const auto rp = request.request->row_pointers;
    const size_t ids_count = static_cast<size_t>(rp.back() - rp.front());
    const size_t dim = request.request->embedding_table->EmbeddingDim();
    const size_t lookup_elements =
        ids_count <=
                (std::numeric_limits<size_t>::max() - request.output_size) / dim
            ? ids_count * dim
            : std::numeric_limits<size_t>::max() - request.output_size;
    const size_t req_work = request.output_size + lookup_elements;
    if (req_work > std::numeric_limits<size_t>::max() - total_work_size) {
      total_work_size = std::numeric_limits<size_t>::max();
    } else {
      total_work_size += req_work;
    }
  }

  auto run_inline = [&]() {
    for (size_t index = 0; index < prepared.size(); ++index) {
      const size_t num_bags =
          prepared[index].request->row_pointers.size() - 1;
      ExecuteTask(absl::MakeConstSpan(prepared), Task{index, 0, num_bags});
    }
  };

  const int pool_threads = thread_pool->NumThreads();
  // Small groups run inline: scheduling and waking pool threads would cost
  // more than the lookups themselves.
  if (total_work_size <= inline_work_threshold || pool_threads == 1 ||
      thread_pool->CurrentThreadId() >= 0) {
    run_inline();
    return absl::OkStatus();
  }

  PoolConcurrencyState& state = GetPoolState(thread_pool);
  const int callers =
      state.active_callers.fetch_add(1, std::memory_order_acq_rel) + 1;
  struct CallerGuard {
    std::atomic<int>& counter;
    ~CallerGuard() { counter.fetch_sub(1, std::memory_order_release); }
  } caller_guard{state.active_callers};

  if (callers >= pool_threads) {
    run_inline();
    return absl::OkStatus();
  }

  // Ceiling division ensures that whenever `callers < pool_threads`, idle pool
  // cores can be recruited instead of sitting unused when e.g. 5 of 8 threads
  // are in ExecuteEmbeddingBagBatch and 3 are in gRPC/XLA.
  const int fair_workers =
      (pool_threads + std::max(1, callers) - 1) / std::max(1, callers);

  ASSIGN_OR_RETURN(
      TaskList tasks,
      BuildTasks(absl::MakeConstSpan(prepared), pool_threads));

  const int max_helpers =
      std::min<int>(fair_workers - 1, static_cast<int>(tasks.size()) - 1);
  if (max_helpers <= 0) {
    run_inline();
    return absl::OkStatus();
  }

  int helper_count = 0;
  int cur_helpers = state.active_helpers.load(std::memory_order_relaxed);
  for (;;) {
    const int cur_callers =
        state.active_callers.load(std::memory_order_relaxed);
    const int available_helpers = pool_threads - cur_callers - cur_helpers;
    const int claim = std::min(max_helpers, available_helpers);
    if (claim <= 0) break;
    if (state.active_helpers.compare_exchange_weak(
            cur_helpers, cur_helpers + claim, std::memory_order_acquire,
            std::memory_order_relaxed)) {
      helper_count = claim;
      break;
    }
  }

  if (helper_count <= 0) {
    run_inline();
    return absl::OkStatus();
  }

  BatchSlot* slot = AcquireBatchSlot(helper_count + 1);
  if (ABSL_PREDICT_FALSE(slot == nullptr)) {
    state.active_helpers.fetch_sub(helper_count, std::memory_order_release);
    run_inline();
    return absl::OkStatus();
  }

  const size_t num_tasks = tasks.size();
  absl::BlockingCounter tasks_counter(static_cast<int>(num_tasks));
  slot->next_task.store(0, std::memory_order_relaxed);
  slot->num_tasks = num_tasks;
  slot->prepared_data = prepared.data();
  slot->prepared_size = prepared.size();
  slot->tasks_data = tasks.data();
  slot->tasks_counter = &tasks_counter;

  for (int worker = 0; worker < helper_count; ++worker) {
    thread_pool->Schedule([slot, &state]() {
      const size_t n = slot->num_tasks;
      for (;;) {
        const size_t index =
            slot->next_task.fetch_add(1, std::memory_order_relaxed);
        if (index >= n) break;
        ExecuteTask(
            absl::MakeConstSpan(slot->prepared_data, slot->prepared_size),
            slot->tasks_data[index]);
        slot->tasks_counter->DecrementCount();
      }
      state.active_helpers.fetch_sub(1, std::memory_order_release);
      slot->refs.fetch_sub(1, std::memory_order_acq_rel);
    });
  }
  for (;;) {
    const size_t index =
        slot->next_task.fetch_add(1, std::memory_order_relaxed);
    if (index >= num_tasks) break;
    ExecuteTask(absl::MakeConstSpan(prepared), tasks[index]);
    tasks_counter.DecrementCount();
  }
  slot->refs.fetch_sub(1, std::memory_order_acq_rel);
  tasks_counter.Wait();
  return absl::OkStatus();
}

}  // namespace xla_serving::data::cpu_embedding::internal

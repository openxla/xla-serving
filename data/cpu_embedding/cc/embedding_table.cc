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

#include "data/cpu_embedding/cc/embedding_table.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <utility>

#include "data/cpu_embedding/cc/array.h"  // from @neptune
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "Eigen/Core"

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

}  // namespace

absl::StatusOr<std::unique_ptr<EmbeddingTable>> EmbeddingTable::Create(
    absl::string_view name, Array data, enum CombineMode combine_mode) {
  if (data.GetShape().dims.size() != 2) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "Embedding table data must be a 2D array. Got shape: (%s)",
        absl::StrJoin(data.GetShape().dims, ",")));
  }
  if (name.empty()) {
    return absl::InvalidArgumentError("Embedding table name cannot be empty.");
  }

  return absl::WrapUnique(
      new EmbeddingTable(name, std::move(data), combine_mode));
}

absl::StatusOr<Array> EmbeddingTable::Lookup(
    absl::Span<const int64_t> ids) const {
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
                               enum CombineMode combine_mode)
    : name_(name),
      data_(std::move(data)),
      combine_mode_(combine_mode) {
  vocab_size_ = data_.GetShape().dims[0];
  embedding_dim_ = data_.GetShape().dims[1];
}

}  // namespace neptune::data::cpu_embedding

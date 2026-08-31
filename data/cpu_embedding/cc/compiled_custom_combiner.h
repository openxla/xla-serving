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

#ifndef THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPILED_CUSTOM_COMBINER_H_
#define THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPILED_CUSTOM_COMBINER_H_

#include <cstdint>
#include <functional>
#include <memory>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune

namespace neptune::data::cpu_embedding {

class CompiledCustomCombiner {
 public:
  struct Options {
    int32_t tile_size = 128;
    int32_t max_valency = 16;
    int32_t embedding_dim = 64;
    int32_t num_weights = 0;
    int32_t output_dim = 64;
  };

  using RawComputeFn = std::function<absl::Status(
      absl::Span<const int32_t> valencies, absl::Span<const float> vectors,
      absl::Span<const float> weights, absl::Span<float> output)>;

  static absl::StatusOr<std::unique_ptr<CompiledCustomCombiner>>
  CreateFromFunction(RawComputeFn fn, Options options);

  static absl::StatusOr<std::unique_ptr<CompiledCustomCombiner>>
  CompileFromStableHlo(
      absl::string_view bytecode_or_text,
      composite::transforms::cpu_embedding::CustomCombinerSpec::Format format,
      Options options);

  virtual ~CompiledCustomCombiner() = default;

  // Evaluates a tile of samples. Thread-safe, stateless, lock-free.
  // inputs:
  //   valencies: Span of size [tile_size] (int32)
  //   vectors:   Span of size [tile_size * max_valency * embedding_dim] (float)
  //   weights:   Span of size [tile_size * num_weights] (float, optional)
  // output:
  //   output:    Span of size [tile_size * output_dim] (float)
  virtual absl::Status ExecuteTile(absl::Span<const int32_t> valencies,
                                   absl::Span<const float> vectors,
                                   absl::Span<const float> weights,
                                   absl::Span<float> output) const = 0;

  const Options& GetOptions() const { return options_; }

 protected:
  explicit CompiledCustomCombiner(Options options) : options_(options) {
    if (options_.output_dim <= 0) {
      options_.output_dim = options_.embedding_dim;
    }
  }

  Options options_;
};

}  // namespace neptune::data::cpu_embedding

#endif  // THIRD_PARTY_NEPTUNE_DATA_CPU_EMBEDDING_CC_COMPILED_CUSTOM_COMBINER_H_

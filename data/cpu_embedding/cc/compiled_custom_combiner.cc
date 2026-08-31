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

#include "data/cpu_embedding/cc/compiled_custom_combiner.h"  // from @neptune

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "data/kernels/cpu_embedding/cpu_embedding.pb.h"  // from @neptune
#include "xla/backends/cpu/nanort/nanort_client.h"  // from @xla
#include "xla/backends/cpu/nanort/nanort_executable.h"  // from @xla
#include "xla/hlo/builder/xla_computation.h"  // from @xla
#include "xla/pjrt/mlir_to_hlo.h"  // from @xla
#include "xla/tsl/concurrency/async_value_ref.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace neptune::data::cpu_embedding {
namespace {

using composite::transforms::cpu_embedding::CustomCombinerSpec;

template <typename T, size_t Alignment = 64>
class AlignedBuffer {
 public:
  AlignedBuffer() : ptr_(nullptr), capacity_(0) {}
  ~AlignedBuffer() {
    if (ptr_) free(ptr_);
  }
  T* Get(size_t count) {
    if (count > capacity_) {
      if (ptr_) free(ptr_);
      if (posix_memalign(reinterpret_cast<void**>(&ptr_), Alignment,
                         count * sizeof(T)) != 0) {
        ptr_ = nullptr;
        capacity_ = 0;
        return nullptr;
      }
      capacity_ = count;
    }
    return ptr_;
  }

 private:
  T* ptr_;
  size_t capacity_;
};

class FunctionCustomCombiner : public CompiledCustomCombiner {
 public:
  FunctionCustomCombiner(RawComputeFn fn, Options options)
      : CompiledCustomCombiner(options), fn_(std::move(fn)) {}

  absl::Status ExecuteTile(absl::Span<const int32_t> valencies,
                           absl::Span<const float> vectors,
                           absl::Span<const float> weights,
                           absl::Span<float> output) const override {
    return fn_(valencies, vectors, weights, output);
  }

 private:
  RawComputeFn fn_;
};

class XlaCustomCombiner : public CompiledCustomCombiner {
 public:
  static absl::StatusOr<std::unique_ptr<XlaCustomCombiner>> Create(
      absl::string_view bytecode_or_text, CustomCombinerSpec::Format format,
      Options options) {
    xla::XlaComputation computation;
    TF_RETURN_IF_ERROR(xla::ParseMlirModuleStringAndConvertToXlaComputation(
        bytecode_or_text, computation,
        /*use_tuple_args=*/false, /*return_tuple=*/false));

    xla::cpu::NanoRtClient client;
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
                        client.Compile(computation));

    return absl::WrapUnique(new XlaCustomCombiner(std::move(executable), options));
  }

  absl::Status ExecuteTile(absl::Span<const int32_t> valencies,
                           absl::Span<const float> vectors,
                           absl::Span<const float> weights,
                           absl::Span<float> output) const override {
    const int32_t tile_size = options_.tile_size;
    const int32_t max_valency = options_.max_valency;
    const int32_t dim = options_.embedding_dim;
    const int32_t output_dim = options_.output_dim;

    if (valencies.size() != static_cast<size_t>(tile_size)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "valencies size %d does not match tile size %d", valencies.size(),
          tile_size));
    }
    if (vectors.size() !=
        static_cast<size_t>(tile_size * max_valency * dim)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "vectors size %d does not match expected %d", vectors.size(),
          tile_size * max_valency * dim));
    }
    if (output.size() != static_cast<size_t>(tile_size * output_dim)) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "output size %d does not match expected %d", output.size(),
          tile_size * output_dim));
    }

    constexpr uintptr_t kAlignmentMask = 63;  // 64-byte alignment
    const int32_t* valencies_ptr = valencies.data();
    if ((reinterpret_cast<uintptr_t>(valencies_ptr) & kAlignmentMask) != 0) {
      thread_local AlignedBuffer<int32_t> tl_aligned_valencies;
      int32_t* aligned = tl_aligned_valencies.Get(valencies.size());
      std::memcpy(aligned, valencies.data(), valencies.size() * sizeof(int32_t));
      valencies_ptr = aligned;
    }

    const float* vectors_ptr = vectors.data();
    if ((reinterpret_cast<uintptr_t>(vectors_ptr) & kAlignmentMask) != 0) {
      thread_local AlignedBuffer<float> tl_aligned_vectors;
      float* aligned = tl_aligned_vectors.Get(vectors.size());
      std::memcpy(aligned, vectors.data(), vectors.size() * sizeof(float));
      vectors_ptr = aligned;
    }

    const float* weights_ptr = weights.data();
    if (options_.num_weights > 0 && !weights.empty() &&
        (reinterpret_cast<uintptr_t>(weights_ptr) & kAlignmentMask) != 0) {
      thread_local AlignedBuffer<float> tl_aligned_weights;
      float* aligned = tl_aligned_weights.Get(weights.size());
      std::memcpy(aligned, weights.data(), weights.size() * sizeof(float));
      weights_ptr = aligned;
    }

    float* output_ptr = output.data();
    const bool output_needs_copy =
        (reinterpret_cast<uintptr_t>(output_ptr) & kAlignmentMask) != 0;
    thread_local AlignedBuffer<float> tl_aligned_output;
    if (output_needs_copy) {
      output_ptr = tl_aligned_output.Get(output.size());
    }

    absl::InlinedVector<xla::cpu::NanoRtExecutable::Argument, 3> arguments;
    arguments.emplace_back(valencies_ptr, valencies.size());
    arguments.emplace_back(vectors_ptr, vectors.size());
    if (options_.num_weights > 0 && !weights.empty()) {
      arguments.emplace_back(weights_ptr, weights.size());
    }

    absl::InlinedVector<xla::cpu::NanoRtExecutable::Result, 1> results;
    results.emplace_back(output_ptr, output.size());

    thread_local AlignedBuffer<std::byte> tl_temp;
    const size_t temp_size = executable_->temp_buffer_size();
    std::byte* temp_ptr = temp_size > 0 ? tl_temp.Get(temp_size) : nullptr;

    xla::cpu::NanoRtExecutable::ExecuteOptions run_options;
    auto event = executable_->Execute(
        arguments, results, absl::MakeSpan(temp_ptr, temp_size), run_options);
    if (!event.IsConcrete()) {
      tsl::BlockUntilReady(event);
    }
    if (event.IsError()) {
      return event.GetError();
    }
    if (output_needs_copy) {
      std::memcpy(output.data(), output_ptr, output.size() * sizeof(float));
    }
    return absl::OkStatus();
  }

 private:
  XlaCustomCombiner(std::unique_ptr<xla::cpu::NanoRtExecutable> executable,
                    Options options)
      : CompiledCustomCombiner(options),
        executable_(std::move(executable)) {}

  std::unique_ptr<xla::cpu::NanoRtExecutable> executable_;
};

}  // namespace

absl::StatusOr<std::unique_ptr<CompiledCustomCombiner>>
CompiledCustomCombiner::CreateFromFunction(RawComputeFn fn, Options options) {
  if (options.tile_size <= 0 || options.max_valency <= 0 ||
      options.embedding_dim <= 0) {
    return absl::InvalidArgumentError("Invalid dimension options provided.");
  }
  return std::make_unique<FunctionCustomCombiner>(std::move(fn), options);
}

absl::StatusOr<std::unique_ptr<CompiledCustomCombiner>>
CompiledCustomCombiner::CompileFromStableHlo(
    absl::string_view bytecode_or_text, CustomCombinerSpec::Format format,
    Options options) {
  if (options.tile_size <= 0 || options.max_valency <= 0 ||
      options.embedding_dim <= 0) {
    return absl::InvalidArgumentError("Invalid dimension options provided.");
  }
  return XlaCustomCombiner::Create(bytecode_or_text, format, options);
}

}  // namespace neptune::data::cpu_embedding

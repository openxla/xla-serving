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
#include <limits>
#include <utility>
#include <vector>

#if defined(__linux__) && defined(__aarch64__)
#include <asm/hwcap.h>
#include <sys/auxv.h>
#endif

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

absl::StatusOr<EmbeddingBagBackend> ResolveBackend(
    const EmbeddingBagRequest& request, EmbeddingBagBackend requested) {
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

absl::StatusOr<size_t> ValidateEmbeddingBagInput(
    const EmbeddingBagRequest& request) {
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
  if (request.row_pointers.front() < 0 ||
      !std::is_sorted(request.row_pointers.begin(),
                      request.row_pointers.end())) {
    return absl::InvalidArgumentError(
        "Row pointers must be non-negative and monotonic.");
  }
  const size_t first = request.row_pointers.front();
  const size_t last = request.row_pointers.back();
  if (last > request.embedding_ids.size()) {
    return absl::InvalidArgumentError(
        "Final row pointer exceeds embedding IDs size.");
  }
  if (!request.weights.empty() && last > request.weights.size()) {
    return absl::InvalidArgumentError(
        "Final row pointer exceeds weights size.");
  }
  if (first != last && request.embedding_ids.data() == nullptr) {
    return absl::InvalidArgumentError("Embedding-ID storage must be non-null.");
  }
  if (!request.weights.empty() && request.weights.data() == nullptr) {
    return absl::InvalidArgumentError("Weight storage must be non-null.");
  }
  for (size_t index = first; index < last; ++index) {
    const int64_t id = request.embedding_ids[index];
    if (id < 0 || id >= request.embedding_table->VocabSize()) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "Embedding ID (%d) at index %d is out of bounds [0, %d).", id, index,
          request.embedding_table->VocabSize()));
    }
  }
  const size_t dim = request.embedding_table->EmbeddingDim();
  const size_t num_bags = request.row_pointers.size() - 1;
  if (num_bags > std::numeric_limits<size_t>::max() / dim) {
    return absl::InvalidArgumentError("Embedding output size overflows.");
  }
  const size_t rows = request.embedding_table->VocabSize();
  if (rows > std::numeric_limits<size_t>::max() / dim) {
    return absl::InvalidArgumentError("Embedding table size overflows.");
  }
  return num_bags * dim;
}

float BagScale(const EmbeddingBagRequest& request, size_t bag) {
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

// Inputs are validated before worker tasks run.
void ReducePortable(const EmbeddingBagRequest& request,
                    absl::Span<float> output_buffer, size_t bag_begin,
                    size_t bag_end) {
  const size_t dim = request.embedding_table->EmbeddingDim();
  const float* table = request.embedding_table->AsTensor<float>().data();
  for (size_t bag = bag_begin; bag < bag_end; ++bag) {
    Eigen::Map<Eigen::VectorXf> output(output_buffer.data() + bag * dim, dim);
    output.setZero();
    const size_t begin = request.row_pointers[bag];
    const size_t end = request.row_pointers[bag + 1];
    for (size_t position = begin; position < end; ++position) {
      Eigen::Map<const Eigen::VectorXf> row(
          table + request.embedding_ids[position] * dim, dim);
      if (request.weights.empty()) {
        output += row;
      } else {
        output += request.weights[position] * row;
      }
    }
    if (begin != end && request.embedding_table->GetCombineMode() !=
                            EmbeddingTable::CombineMode::kSum) {
      output *= BagScale(request, bag);
    }
  }
}

void ReduceSve(const EmbeddingBagRequest& request,
               absl::Span<float> output_buffer, size_t bag_begin,
               size_t bag_end) {
  const size_t dim = request.embedding_table->EmbeddingDim();
  absl::Span<float> output =
      output_buffer.subspan(bag_begin * dim, (bag_end - bag_begin) * dim);
  // CSR offsets are absolute; shift only the row-pointer base.
  std::fill(output.begin(), output.end(), 0.0f);
  sve::AccumulateEmbeddingBagsF32(
      request.embedding_ids.data(), request.row_pointers.data() + bag_begin,
      bag_end - bag_begin,
      request.weights.empty() ? nullptr : request.weights.data(),
      request.embedding_table->AsTensor<float>().data(), dim, output.data());
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
  const EmbeddingBagRequest* request;
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

absl::StatusOr<PreparedRequest> PrepareRequest(
    const EmbeddingBagRequest& request, size_t index,
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

absl::Status AppendRange(const void* data, size_t elements, size_t element_size,
                         size_t request_index, const char* kind,
                         std::vector<AddressRange>* ranges) {
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
  std::vector<AddressRange> outputs;
  std::vector<AddressRange> inputs;
  outputs.reserve(requests.size());
  inputs.reserve(requests.size() * 4);
  for (size_t index = 0; index < requests.size(); ++index) {
    const PreparedRequest& prepared = requests[index];
    const EmbeddingBagRequest& request = *prepared.request;
    RETURN_IF_ERROR(AppendRange(prepared.output.data(), prepared.output.size(),
                                sizeof(float), index, "output", &outputs));
    RETURN_IF_ERROR(AppendRange(request.embedding_ids.data(),
                                request.embedding_ids.size(), sizeof(int64_t),
                                index, "embedding-ID", &inputs));
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

absl::StatusOr<uint64_t> TaskCost(const EmbeddingBagRequest& request,
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

absl::StatusOr<std::vector<Task>> BuildTasks(
    absl::Span<const PreparedRequest> requests, int pool_threads) {
  // Aim for four bag-range tasks per worker without splitting individual bags.
  constexpr uint64_t kTasksPerThread = 4;
  uint64_t total_cost = 0;
  for (const PreparedRequest& prepared : requests) {
    const EmbeddingBagRequest& request = *prepared.request;
    ASSIGN_OR_RETURN(uint64_t cost,
                     TaskCost(request, 0, request.row_pointers.size() - 1));
    if (cost > std::numeric_limits<uint64_t>::max() - total_cost) {
      return absl::InvalidArgumentError("Embedding-bag batch cost overflows.");
    }
    total_cost += cost;
  }
  const uint64_t target_count =
      static_cast<uint64_t>(pool_threads) * kTasksPerThread;
  const uint64_t target_cost = std::max<uint64_t>(
      1, total_cost / target_count + (total_cost % target_count != 0));

  std::vector<Task> tasks;
  for (size_t request_index = 0; request_index < requests.size();
       ++request_index) {
    const EmbeddingBagRequest& request = *requests[request_index].request;
    const size_t num_bags = request.row_pointers.size() - 1;
    size_t tile_begin = 0;
    uint64_t tile_cost = 0;
    for (size_t bag = 0; bag < num_bags; ++bag) {
      ASSIGN_OR_RETURN(uint64_t bag_cost, TaskCost(request, bag, bag + 1));
      if (bag != tile_begin && tile_cost >= target_cost) {
        tasks.push_back({request_index, tile_begin, bag});
        tile_begin = bag;
        tile_cost = 0;
      }
      tile_cost += bag_cost;
    }
    tasks.push_back({request_index, tile_begin, num_bags});
  }
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

absl::Status ComputeEmbeddingActivationsF32(
    absl::Span<const int64_t> embedding_ids,
    absl::Span<const int32_t> row_pointers, absl::Span<const float> weights,
    const EmbeddingTable& embedding_table, absl::Span<float> buffer,
    EmbeddingBagBackend requested, EmbeddingBagBackend* selected) {
  absl::AnyInvocable<absl::Span<float>(size_t)> allocator =
      [buffer](size_t) mutable -> absl::Span<float> { return buffer; };
  EmbeddingBagRequest request{
      .embedding_ids = embedding_ids,
      .row_pointers = row_pointers,
      .embedding_table = &embedding_table,
      .weights = weights,
      .output_allocator = &allocator,
  };
  ASSIGN_OR_RETURN(PreparedRequest prepared,
                   PrepareRequest(request, 0, requested));
  RETURN_IF_ERROR(AllocateOutputs(absl::MakeSpan(&prepared, 1)));
  RETURN_IF_ERROR(ValidateOutputIsolation(absl::MakeConstSpan(&prepared, 1)));
  if (selected != nullptr) *selected = BackendForCurrentThread(prepared);
  ExecuteTask(absl::MakeConstSpan(&prepared, 1),
              Task{0, 0, row_pointers.size() - 1});
  return absl::OkStatus();
}

absl::Status ExecuteEmbeddingBagBatch(
    absl::Span<const EmbeddingBagRequest> requests,
    EmbeddingBagBackend requested, Eigen::ThreadPoolInterface* thread_pool) {
  if (requests.empty()) {
    return absl::InvalidArgumentError(
        "Embedding-bag batch requires at least one request.");
  }
  if (thread_pool == nullptr || thread_pool->NumThreads() <= 0) {
    return absl::InvalidArgumentError(
        "Embedding-bag batch requires a non-empty thread pool.");
  }

  std::vector<PreparedRequest> prepared;
  prepared.reserve(requests.size());
  for (size_t index = 0; index < requests.size(); ++index) {
    ASSIGN_OR_RETURN(PreparedRequest request,
                     PrepareRequest(requests[index], index, requested));
    prepared.push_back(std::move(request));
  }
  ASSIGN_OR_RETURN(
      std::vector<Task> tasks,
      BuildTasks(absl::MakeConstSpan(prepared), thread_pool->NumThreads()));
  RETURN_IF_ERROR(AllocateOutputs(absl::MakeSpan(prepared)));
  RETURN_IF_ERROR(ValidateOutputIsolation(absl::MakeConstSpan(prepared)));

  if (thread_pool->NumThreads() == 1 || thread_pool->CurrentThreadId() >= 0) {
    for (const Task& task : tasks) {
      ExecuteTask(absl::MakeConstSpan(prepared), task);
    }
    return absl::OkStatus();
  }

  const int worker_count = static_cast<int>(std::min<size_t>(
      static_cast<size_t>(thread_pool->NumThreads()), tasks.size()));
  std::atomic<size_t> next_task{0};
  absl::BlockingCounter counter(worker_count);
  for (int worker = 0; worker < worker_count; ++worker) {
    thread_pool->Schedule([&]() {
      for (;;) {
        const size_t index = next_task.fetch_add(1, std::memory_order_relaxed);
        if (index >= tasks.size()) break;
        ExecuteTask(absl::MakeConstSpan(prepared), tasks[index]);
      }
      counter.DecrementCount();
    });
  }
  counter.Wait();
  return absl::OkStatus();
}

}  // namespace xla_serving::data::cpu_embedding::internal

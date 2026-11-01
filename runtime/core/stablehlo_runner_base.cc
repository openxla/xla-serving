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

#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <iterator>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/base/call_once.h"  // from @abseil-cpp
#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/container/inlined_vector.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/log/vlog_is_on.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/cord.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/aliasing_array.h"  // from @xla_serving
#include "runtime/core/device_buffer.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/host_buffer_converter.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_utils.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/context.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/layout.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/transpose.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/executable.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_memory.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/status_macros.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/framework/serving_device_selector.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace xla_serving {

std::atomic<uint32_t> StableHloRunnerBase::launch_id_(1);

namespace {

// Number of distinct IFRT execution stream ids used by Execute(). Kept bounded
// in case an IFRT backend keeps per-stream-id state that is never released;
// this is far above any realistic number of concurrent executions.
constexpr uint32_t kNumExecutionStreams = 1024;

auto* input_transfer_counter =
    ::tsl::monitoring::Counter<1>::New("/xla_serving/input_transfer/dispatched",
                                       "Tracks how many transfers were routed "
                                       "to the Arena vs. Default IFRT path.",
                                       "transfer_path");

absl::Status ValidateDeviceBufferInput(const xla::ifrt::Array& original_input,
                                       xla::ifrt::DType target_dtype,
                                       const xla::ifrt::Shape& target_shape,
                                       int idx_in_stablehlo_spec) {
  if (original_input.dtype() != target_dtype) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input DType mismatch for device buffer at index ",
                     idx_in_stablehlo_spec, ". Expected ", target_dtype,
                     ", but got ", original_input.dtype()));
  }
  if (original_input.shape() != target_shape) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Zero-copy D2D execution requires the input "
        "shape to exactly match the optimized model's target shape. "
        "Expected ",
        target_shape, ", but got ", original_input.shape(), ". ",
        "To fix this, you must pre-reshape the device array to match the "
        "optimized signature. Otherwise, provide a HostBuffer to allow "
        "StableHloRunner to perform the reshape via host memory."));
  }
  return absl::OkStatus();
}

absl::Status ValidateOutputArray(const xla::ifrt::Array& array_for_copy,
                                 xla::ifrt::DType expected_dtype,
                                 const xla::ifrt::Shape& expected_ifrt_shape,
                                 int output_index) {
  if (array_for_copy.dtype() != expected_dtype) {
    return absl::InternalError(absl::StrCat(
        "Output DType mismatch at index ", output_index, ". Expected ",
        expected_dtype, ", but got ", array_for_copy.dtype()));
  }
  if (array_for_copy.shape() != expected_ifrt_shape) {
    return absl::InternalError(absl::StrCat(
        "Output shape mismatch at index ", output_index, ". Expected ",
        expected_ifrt_shape, ", but got ", array_for_copy.shape()));
  }
  return absl::OkStatus();
}

// Transposes a (possibly strided) `input` by `permutation` into a newly
// allocated buffer whose dim `i` is `input` dim `permutation[i]`. The output
// buffer has default (dense, major-to-minor) byte strides, which is the layout
// `TransposePlan` writes.
absl::StatusOr<HostBuffer> TransposeHostBuffer(
    const HostBuffer& input, absl::Span<const int64_t> permutation) {
  const absl::Span<const int64_t> dims = input.shape().dims();
  if (permutation.size() != dims.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Host transpose permutation [", absl::StrJoin(permutation, ","),
        "] does not match input shape ", input.shape()));
  }
  // The compiled program always expects the permuted order for this input, so
  // fail instead of silently passing untransposed data through.
  if (input.dtype() != input.storage_dtype()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Cannot transpose input with dtype ", input.dtype(),
                     " stored as ", input.storage_dtype()));
  }
  const std::optional<int> byte_size = input.dtype().byte_size();
  if (!byte_size.has_value()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Unsupported dtype for host transpose: ", input.dtype()));
  }

  std::vector<int64_t> transposed_dims(dims.size());
  for (size_t i = 0; i < permutation.size(); ++i) {
    if (permutation[i] < 0 ||
        permutation[i] >= static_cast<int64_t>(dims.size())) {
      return absl::InvalidArgumentError(
          absl::StrCat("Invalid host transpose permutation: ",
                       absl::StrJoin(permutation, ",")));
    }
    transposed_dims[i] = dims[permutation[i]];
  }
  HostBuffer transposed_buffer(input.dtype(),
                               xla::ifrt::Shape(transposed_dims));
  if (input.shape().num_elements() == 0) {
    return transposed_buffer;
  }
  xla::TransposePlan::Options options{
      .elem_size_in_bytes = static_cast<size_t>(*byte_size),
      .dims = dims,
      .permutation = permutation,
      .num_threads = 1,
  };
  if (input.byte_strides().has_value()) {
    options.input_striding =
        xla::TransposePlan::Striding{*input.byte_strides()};
  }

  TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::TransposePlan> plan,
                      xla::TransposePlan::Create(options));
  plan->Execute(input.base(), transposed_buffer.base());
  return transposed_buffer;
}

// Returns the PjRt memory space that `sharding` places `device` shards in.
absl::StatusOr<xla::PjRtMemorySpace*> GetPjRtMemorySpace(
    xla::ifrt::Device& device, xla::ifrt::MemoryKind memory_kind) {
  xla::ifrt::Memory* memory = nullptr;
  if (memory_kind.memory_kind().has_value()) {
    for (xla::ifrt::Memory* candidate : device.Memories()) {
      if (candidate->Kind() == memory_kind) {
        memory = candidate;
        break;
      }
    }
    if (memory == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Device ", device.DebugString(),
                       " has no memory of kind ", memory_kind));
    }
  } else {
    TF_ASSIGN_OR_RETURN(memory, device.DefaultMemory());
  }

  auto* pjrt_memory =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleMemory>(memory);
  if (pjrt_memory == nullptr) {
    return absl::UnimplementedError(
        absl::StrCat("Memory ", *memory, " is not PjRt-backed"));
  }

  return pjrt_memory->pjrt_memory();
}

// Allocates device memory for one output of the executable, shaped, sharded
// and laid out the way the executable wants the argument that backs that
// output. The memory is left uninitialized on purpose: it is handed to the
// execution as an aliased argument, whose only job is to be written into, so
// initializing it would only cost a transfer.
//
// `layout` is the layout the executable declares for that argument. It is not
// necessarily the default layout for the shape -- on TPU it routinely is not,
// and allocating with the default layout yields a buffer of the wrong size --
// so it is applied rather than inferred.
absl::StatusOr<xla::ifrt::ArrayRef> MakeUninitializedOutputArray(
    xla::ifrt::Client& client, const ArgSpec& output_spec,
    const xla::ifrt::ShardingRef& sharding,
    std::shared_ptr<const xla::PjRtLayout> layout) {
  auto* pjrt_client =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleClient>(&client);
  if (pjrt_client == nullptr) {
    return absl::UnimplementedError(
        "Input aliasing can only allocate output buffers on a PjRt-backed "
        "IFRT client. Pass the buffers in `ExecuteOptions::output_arrays` "
        "instead.");
  }

  const xla::ifrt::Shape shape = output_spec.GetIfrtShape();
  TF_ASSIGN_OR_RETURN(const xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(output_spec.dtype));
  using ShardList =
      std::vector<std::pair<xla::ifrt::Shape, xla::ifrt::ShardingRef>>;
  TF_ASSIGN_OR_RETURN(
      const ShardList shards,
      sharding->Disassemble(
          shape, xla::ifrt::SingleDeviceShardSemantics::kAddressableShards));

  xla::ifrt::PjRtArray::PjRtBuffers buffers;
  buffers.reserve(shards.size());
  for (const auto& [shard_shape, shard_sharding] : shards) {
    const absl::Span<xla::ifrt::Device* const> shard_devices =
        shard_sharding->devices()->devices();
    TF_RET_CHECK(shard_devices.size() == 1)
        << "Expected a single-device shard, but got " << shard_devices.size()
        << " devices";
    TF_ASSIGN_OR_RETURN(xla::PjRtMemorySpace * memory_space,
                        GetPjRtMemorySpace(*shard_devices.front(),
                                           shard_sharding->memory_kind()));
    xla::Shape shard_xla_shape =
        xla::ShapeUtil::MakeShape(primitive_type, shard_shape.dims());
    if (layout != nullptr) {
      *shard_xla_shape.mutable_layout() = layout->xla_layout();
    }
    TF_ASSIGN_OR_RETURN(std::unique_ptr<xla::PjRtBuffer> buffer,
                        pjrt_client->pjrt_client()->CreateUninitializedBuffer(
                            shard_xla_shape, memory_space));
    buffers.push_back(std::shared_ptr<xla::PjRtBuffer>(std::move(buffer)));
  }

  TF_ASSIGN_OR_RETURN(tsl::RCReference<xla::ifrt::PjRtArray> array,
                      xla::ifrt::PjRtArray::Create(
                          pjrt_client, output_spec.dtype, shape, sharding,
                          std::move(buffers), std::move(layout)));
  return xla::ifrt::ArrayRef(std::move(array));
}

}  // namespace

absl::StatusOr<ModelIoMetadata> ModelIoMetadata::CreateFromStableHlo(
    const StableHlo& stablehlo, int num_weights) {
  // The following checks are defensive and should never fail, since some of
  // them are already checked in `StableHloRunner::Create`
  if (stablehlo.specs().empty()) {
    return absl::InvalidArgumentError(
        "`StableHlo` should have at least one spec");
  }
  if (num_weights < 0) {
    return absl::InvalidArgumentError(
        "`num_weights` should be non-negative when creating ModelIoMetadata");
  }

  if (stablehlo.specs()[0].inputs.size() < num_weights) {
    return absl::InvalidArgumentError(absl::StrFormat(
        "StableHLO module has fewer inputs (%d) than the number of "
        "weights (%d)",
        stablehlo.specs()[0].inputs.size(), num_weights));
  }
  // A model exported for input aliasing ends its argument list with one
  // aliased buffer per output. The caller does not supply those, so they must
  // not be counted as user inputs.
  const auto num_user_inputs_of = [&](int i) -> absl::StatusOr<int> {
    const StableHlo::Spec& spec = stablehlo.specs()[i];
    const int num_non_weight_args = spec.inputs.size() - num_weights;
    if (spec.num_input_aliased_output < 0 ||
        num_non_weight_args < spec.num_input_aliased_output) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "StableHLO module #%d has %d arguments left after its %d weights, "
          "which does not fit the %d input-aliased output buffers it was "
          "declared to have",
          i, num_non_weight_args, num_weights, spec.num_input_aliased_output));
    }
    return num_non_weight_args - spec.num_input_aliased_output;
  };
  TF_ASSIGN_OR_RETURN(int num_user_inputs, num_user_inputs_of(0));
  int num_outputs = stablehlo.specs()[0].outputs.size();

  // Verifies that all StableHLO modules (shape buckets) have a consistent
  // number of user inputs and outputs. This ensures that dynamic shape
  // bucketing is structurally sound before execution.
  for (int i = 1; i < stablehlo.specs().size(); ++i) {
    if (stablehlo.specs()[i].inputs.size() < num_weights) {
      return absl::InvalidArgumentError(absl::StrFormat(
          "StableHLO module has fewer inputs (%d) than the number of "
          "weights (%d)",
          stablehlo.specs()[i].inputs.size(), num_weights));
    }
    TF_ASSIGN_OR_RETURN(int cur_user_inputs, num_user_inputs_of(i));
    int cur_outputs = stablehlo.specs()[i].outputs.size();

    if (cur_user_inputs != num_user_inputs) {
      return absl::InvalidArgumentError(
          "Mismatched number of user inputs across buckets");
    }
    if (cur_outputs != num_outputs) {
      return absl::InvalidArgumentError(
          "Mismatched number of outputs across buckets");
    }
    for (int j = 0; j < num_outputs; ++j) {
      if (stablehlo.specs()[i].outputs[j].placement !=
          stablehlo.specs()[0].outputs[j].placement) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Mismatched output placement across buckets for output #", j,
            ": bucket 0 vs bucket ", i));
      }
    }
  }

  // Get specs for weights that are ever used by any of the StableHLOs.
  // When a weight spec is used, verify that its spec is consistent across all
  // StableHLOs that require this weight.
  // When a weight spec is not used, it means the weight is unused by any of
  // the StableHLOs, thus setting spec to std::nullopt.
  std::vector<std::optional<ArgSpec>> weight_specs(num_weights, std::nullopt);
  for (int spec_idx = 0; spec_idx < stablehlo.specs().size(); ++spec_idx) {
    const StableHlo::Spec& cur_spec = stablehlo.specs()[spec_idx];
    for (int i : cur_spec.mlir_module_kept_var_idx) {
      if (i >= num_weights) {
        break;
      }
      const auto& cur_weight_spec = cur_spec.inputs[i];
      if (!weight_specs[i].has_value()) {
        weight_specs[i] = cur_weight_spec;
      } else if (*weight_specs[i] != cur_weight_spec) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Weights should have the same dtype, shape and sharding among"
            " StableHLOs, but found mismatch for weight #",
            i, " : ", cur_weight_spec.DebugString(), " vs ",
            weight_specs[i].value().DebugString()));
      }
    }
  }

  // Verify that all weights have concrete shapes. There shouldn't be a case
  // where a weight is polymorphic.
  for (const auto& spec : weight_specs) {
    if (!spec.has_value()) {
      continue;
    }
    for (const auto& dim : spec.value().shape) {
      if (!dim.has_value()) {
        return absl::InvalidArgumentError(
            absl::StrCat("Weight should have concrete shape, but found "
                         "weight with polymorphic shape: ",
                         spec.value().DebugString()));
      }
    }
  }

  return ModelIoMetadata{
      .weight_specs = std::move(weight_specs),
      .output_specs = stablehlo.specs()[0].outputs,
      .num_user_inputs = num_user_inputs,
      .num_outputs = num_outputs,
      // TODO: Set has_streaming based on AOT executable metadata
      // when AOT + streaming execution is formally supported.
      .has_streaming = false,
  };
}

StableHloRunnerBase::DataConverter::DataConverter(
    absl_nonnull std::shared_ptr<xla::ifrt::Client> ifrt_client,
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* absl_nonnull thread_pool,
    std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor)
    : ifrt_client_(std::move(ifrt_client)),
      device_list_(std::move(device_list)),
      thread_pool_(thread_pool),
      h2d_transfer_executor_(std::move(h2d_transfer_executor)),
      arena_cell_(input_transfer_counter->GetCell("arena")),
      default_cell_(input_transfer_counter->GetCell("default")) {}

absl::StatusOr<absl::Span<const xla::ifrt::ShardingRef>>
StableHloRunnerBase::DataConverter::ValidateAndResolveShardings(
    absl::Span<const HostBuffer> inputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const int> stablehlo_indices,
    absl::InlinedVector<xla::ifrt::ShardingRef, 16>& mixed_shardings_storage)
    const {
  if (stablehlo_indices.size() != inputs.size()) {
    return absl::InvalidArgumentError(
        "Mismatched inputs and stablehlo_indices size.");
  }

  // Validate data types and shapes before passing to executor to prevent
  // buffer over-reads during memcpy.
  for (int i = 0; i < inputs.size(); ++i) {
    int idx_in_stablehlo_spec = stablehlo_indices[i];
    if (idx_in_stablehlo_spec >= executable_bundle.io_spec.inputs.size()) {
      return absl::InvalidArgumentError("Index out of bounds.");
    }
    const ArgSpec& input_spec =
        executable_bundle.io_spec.inputs[idx_in_stablehlo_spec];
    if (inputs[i].dtype() != input_spec.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Input dtype mismatch. Expected: ", input_spec.dtype,
                       ", but got: ", inputs[i].dtype()));
    }
    TF_RETURN_IF_ERROR(InputShapeEqual(input_spec.shape, inputs[i].shape()));
  }

  absl::Span<const xla::ifrt::ShardingRef> all_shardings;
  if (const auto* portable_map = std::get_if<
          StableHloRunnerBase::CachedExecutableBundle::PortableShardingMap>(
          &executable_bundle.arg_shardings)) {
    TF_RET_CHECK(!device_list_->devices().empty())
        << "Device list cannot be empty";
    auto it = portable_map->find(device_list_->devices()[0]->Id());
    if (it == portable_map->end()) {
      return absl::FailedPreconditionError(
          absl::StrCat("No portable single device sharding found for device ",
                       device_list_->devices()[0]->Id()));
    }
    all_shardings = absl::MakeConstSpan(it->second);
  } else if (const auto* sharding_list = std::get_if<
                 StableHloRunnerBase::CachedExecutableBundle::ShardingList>(
                 &executable_bundle.arg_shardings)) {
    all_shardings = absl::MakeConstSpan(*sharding_list);
  } else {
    return absl::InternalError("Arg shardings are uninitialized.");
  }

  // Check if requested indices are strictly contiguous (The Fast Path)
  bool is_contiguous = true;
  for (int i = 0; i < stablehlo_indices.size(); ++i) {
    if (stablehlo_indices[i] != executable_bundle.input_offset + i) {
      is_contiguous = false;
      break;
    }
  }

  if (is_contiguous) {
    if (executable_bundle.input_offset + stablehlo_indices.size() >
        all_shardings.size()) {
      return absl::InternalError(
          "IFRT Executable arg shardings size mismatch.");
    }
    // O(1) Zero-allocation, zero-refcount-bump fast path
    return all_shardings.subspan(executable_bundle.input_offset,
                                 stablehlo_indices.size());
  }

  // Fallback for mixed inputs, pushing into the caller-owned InlinedVector
  mixed_shardings_storage.reserve(stablehlo_indices.size());
  for (int idx : stablehlo_indices) {
    if (idx >= all_shardings.size()) {
      return absl::InternalError(
          "IFRT Executable arg shardings size mismatch.");
    }
    mixed_shardings_storage.push_back(all_shardings[idx]);
  }
  return absl::MakeConstSpan(mixed_shardings_storage);
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunnerBase::DataConverter::ExecuteDefaultTransfer(
    absl::Span<const HostBuffer> inputs,
    absl::Span<const xla::ifrt::ShardingRef> shardings_span) const {
  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  converted_inputs.reserve(inputs.size());

  for (int i = 0; i < inputs.size(); ++i) {
    const HostBuffer& input_data = inputs[i];
    VLOG(2) << "Converting input of shape " << input_data.shape();

    xla::ifrt::ArrayRef array;
    TF_ASSIGN_OR_RETURN(
        array, sharding_internal::MakeArrayFromHostBuffer(
                   *ifrt_client_, input_data, device_list_, shardings_span[i],
                   *thread_pool_,
                   xla::ifrt::Client::HostBufferSemantics::kImmutableZeroCopy));

    converted_inputs.push_back(std::move(array));
  }
  return converted_inputs;
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunnerBase::DataConverter::ConvertInput(
    absl::Span<const HostBuffer> inputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const int> stablehlo_indices) const {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(h2d_transfer_executor_ != nullptr
                                            ? "ArenaTransferFromHostToDevice"
                                            : "TransferFromHostToDevice",
                                        {});
  });

  // Storage must live for the duration of ConvertInput to prevent dangling span
  // pointers
  absl::InlinedVector<xla::ifrt::ShardingRef, 16> mixed_shardings_storage;
  TF_ASSIGN_OR_RETURN(
      absl::Span<const xla::ifrt::ShardingRef> shardings_span,
      ValidateAndResolveShardings(inputs, executable_bundle, stablehlo_indices,
                                  mixed_shardings_storage));

  if (h2d_transfer_executor_ != nullptr) {
    arena_cell_->IncrementBy(1);
    return h2d_transfer_executor_->ExecuteTransfer(inputs, shardings_span);
  }

  default_cell_->IncrementBy(1);
  return ExecuteDefaultTransfer(inputs, shardings_span);
}

absl::StatusOr<std::vector<xla_serving::Value>>
StableHloRunnerBase::DataConverter::ConvertOutput(
    absl::Span<const xla::ifrt::ArrayRef> stablehlo_outputs,
    const CachedExecutableBundle& executable_bundle,
    absl::Span<const Placement> placements,
    absl::Span<const std::optional<HostBuffer>> output_host_buffers) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("TransferFromDeviceToHost", {});
  });
  if (placements.size() != stablehlo_outputs.size()) {
    return absl::InvalidArgumentError(
        absl::StrCat("Mismatched placements size (", placements.size(),
                     ") and outputs size (", stablehlo_outputs.size(), ")."));
  }
  if (!output_host_buffers.empty() &&
      output_host_buffers.size() != stablehlo_outputs.size()) {
    return absl::InvalidArgumentError(absl::StrCat(
        "Mismatched output host buffers size (", output_host_buffers.size(),
        ") and outputs size (", stablehlo_outputs.size(), ")."));
  }
  std::vector<std::optional<tsl::Future<HostBuffer>>> output_futures(
      stablehlo_outputs.size());
  // Copies into caller-provided host buffers, which complete into memory the
  // caller already holds rather than into a new `HostBuffer`.
  std::vector<std::optional<tsl::Future<>>> copy_futures(
      stablehlo_outputs.size());

  TF_RET_CHECK(!device_list_->devices().empty())
      << "Device list cannot be empty";
  TF_ASSIGN_OR_RETURN(
      absl::Span<const xla::ifrt::ShardingRef> output_shardings,
      executable_bundle.GetOutputShardings(device_list_->devices()[0]->Id()));

  TF_RET_CHECK(output_shardings.size() == stablehlo_outputs.size())
      << "Output shardings count (" << output_shardings.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";
  TF_RET_CHECK(executable_bundle.io_spec.outputs.size() ==
               stablehlo_outputs.size())
      << "IO spec outputs count (" << executable_bundle.io_spec.outputs.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";
  TF_RET_CHECK(executable_bundle.original_io_spec.outputs.size() ==
               stablehlo_outputs.size())
      << "Original IO spec outputs count ("
      << executable_bundle.original_io_spec.outputs.size()
      << ") does not match outputs count (" << stablehlo_outputs.size() << ")";

  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    const xla::ifrt::ArrayRef& array_for_copy = stablehlo_outputs[i];

    if (VLOG_IS_ON(2)) {
      TF_RETURN_IF_ERROR(ValidateOutputArray(
          *array_for_copy, executable_bundle.io_spec.outputs[i].dtype,
          executable_bundle.io_spec.outputs[i].GetIfrtShape(), i));
    }

    if (!output_host_buffers.empty() && output_host_buffers[i].has_value()) {
      const HostBuffer& destination = *output_host_buffers[i];
      const ArgSpec& expected = executable_bundle.original_io_spec.outputs[i];
      if (placements[i] != Placement::kHost) {
        return absl::InvalidArgumentError(absl::StrCat(
            "Output #", i,
            " was given a host buffer to be copied into, but is placed on the "
            "device."));
      }
      if (destination.dtype() != expected.dtype ||
          destination.shape() != expected.GetIfrtShape()) {
        return absl::InvalidArgumentError(absl::StrCat(
            "The host buffer for output #", i, " has dtype ",
            destination.dtype(), " and shape ", destination.shape(),
            ", but the output has dtype ", expected.dtype, " and shape ",
            expected.GetIfrtShape(), "."));
      }
      copy_futures[i] = sharding_internal::CopyArrayToHostBuffer(
          *ifrt_client_, *array_for_copy, output_shardings[i], destination);
    } else if (placements[i] == Placement::kHost) {
      // IFRT's return does not contain sufficient information; so we use
      // sharding spec from metadata.
      VLOG(2) << "Output sharding: " << array_for_copy->sharding();
      output_futures[i] = sharding_internal::MakeHostBufferFromArray(
          *ifrt_client_, *array_for_copy, output_shardings[i], device_list_,
          *thread_pool_);
    }
  }

  std::vector<xla_serving::Value> converted_outputs;
  converted_outputs.reserve(stablehlo_outputs.size());
  for (int i = 0; i < stablehlo_outputs.size(); ++i) {
    if (copy_futures[i].has_value()) {
      // The result is the caller's own memory, now holding the output.
      TF_RETURN_IF_ERROR(copy_futures[i]->Await());
      converted_outputs.push_back(xla_serving::Value(*output_host_buffers[i]));
    } else if (placements[i] == Placement::kHost) {
      TF_ASSIGN_OR_RETURN(auto host_buffer, output_futures[i]->Await());
      const xla::ifrt::Shape expected_ifrt_shape =
          executable_bundle.original_io_spec.outputs[i].GetIfrtShape();
      if (host_buffer.shape() != expected_ifrt_shape) {
        if (host_buffer.shape().num_elements() !=
            expected_ifrt_shape.num_elements()) {
          return absl::InvalidArgumentError(absl::StrCat(
              "Output shape mismatch. Expected: ", expected_ifrt_shape,
              ", but got: ", host_buffer.shape()));
        }
        HostBuffer reshaped_buffer(host_buffer.base(), host_buffer.dtype(),
                                   expected_ifrt_shape, host_buffer.owner());
        converted_outputs.push_back(xla_serving::Value(std::move(reshaped_buffer)));
      } else {
        converted_outputs.push_back(xla_serving::Value(std::move(host_buffer)));
      }
    } else {
      converted_outputs.push_back(
          xla_serving::Value(xla_serving::DeviceBuffer{.ifrt_client = ifrt_client_,
                                       .ifrt_array = stablehlo_outputs[i],
                                       .sharding = output_shardings[i],
                                       .device_list = device_list_}));
    }
  }
  return converted_outputs;
}

StableHloRunnerBase::StableHloRunnerBase(
    ModelIoMetadata model_io_metadata,
    absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
    IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
    std::vector<std::string> weights,
    xla::ifrt::DeviceListRef assigned_device_list,
    absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
    ServingCoreSelector* absl_nullable serving_core_selector,
    std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory,
    std::optional<int64_t> callback_id)
    : model_io_metadata_(std::move(model_io_metadata)),
      assigned_device_list_(std::move(assigned_device_list)),
      ifrt_client_(std::move(client)),
      assigned_devices_(std::move(assigned_devices)),
      streaming_host_callback_id_(callback_id),
      ifrt_loaded_variable_registry_(*ifrt_loaded_variable_registry),
      weights_(std::move(weights)),
      serving_core_selector_(serving_core_selector),
      h2d_transfer_executor_factory_(std::move(h2d_transfer_executor_factory)) {
}

absl::Status StableHloRunnerBase::Init(
    tsl::thread::ThreadPool* absl_nonnull thread_pool) {
  if (IsPortableExecutable()) {
    for (size_t i = 0; i < ifrt_client_->addressable_device_count(); ++i) {
      TF_ASSIGN_OR_RETURN(xla::ifrt::Device * device,
                          ifrt_client_->LookupDevice(xla::ifrt::DeviceId(i)));
      TF_ASSIGN_OR_RETURN(xla::ifrt::DeviceListRef device_list,
                          ifrt_client_->MakeDeviceList({device}));
      // Create resources for all devices.
      TF_RETURN_IF_ERROR(
          CreateResourcesPerDeviceList(device_list, thread_pool));
    }
  } else {
    TF_RETURN_IF_ERROR(
        CreateResourcesPerDeviceList(assigned_device_list_, thread_pool));
  }
  return absl::OkStatus();
}

absl::Status StableHloRunnerBase::CreateResourcesPerDeviceList(
    xla::ifrt::DeviceListRef device_list,
    tsl::thread::ThreadPool* thread_pool) {
  std::unique_ptr<H2DTransferExecutor> h2d_transfer_executor;
  if (h2d_transfer_executor_factory_ != nullptr) {
    TF_ASSIGN_OR_RETURN(h2d_transfer_executor,
                        h2d_transfer_executor_factory_->Create(
                            ifrt_client_.get(), device_list, thread_pool));
  }
  device_list_to_resources_map_[device_list] =
      std::make_unique<ResourcesPerDeviceList>(
          ifrt_client_, device_list, thread_pool,
          std::move(h2d_transfer_executor));
  return absl::OkStatus();
}

bool StableHloRunnerBase::IsPortableExecutable() const {
  return serving_core_selector_ != nullptr &&
         assigned_devices_->num_replicas == 1 &&
         assigned_devices_->num_cores_per_replica == 1;
}

absl::StatusOr<std::vector<std::optional<xla::ifrt::ArrayRef>>>
StableHloRunnerBase::LookupWeights(std::optional<size_t> device_id) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("StableHloRunnerBase::LookupWeights",
                                        {});
  });
  std::vector<int> device_ids;
  if (device_id.has_value()) {
    device_ids.push_back(*device_id);
  } else {
    device_ids.reserve(assigned_device_list_->size());
    for (const auto& device : assigned_device_list_->devices()) {
      device_ids.push_back(device->Id().value());
    }
  }

  const int num_weights = weights_.size();

  // Ifrt Arrays for the weights. When an element is set to std::nullopt, it
  // means the weight is unused (i.e. does not needed to be passed to the
  // StableHLO for execution).
  std::vector<std::optional<xla::ifrt::ArrayRef>> arrays(num_weights,
                                                         std::nullopt);

  // Get weight specs. The weight spec will be nullopt if the weight is unused
  // by the StableHLO.
  const std::vector<std::optional<ArgSpec>>& weights_specs =
      model_io_metadata_.weight_specs;
  if (weights_specs.size() != num_weights) {
    return absl::FailedPreconditionError(
        absl::StrCat("Expected ", weights_.size(), " weight specs, but got ",
                     weights_specs.size(), " weight specs."));
  }

  // Build IFRT arrays for the weights. The array will be nullopt if the weight
  // is unused.
  for (int i = 0; i < num_weights; i++) {
    if (!weights_specs[i].has_value()) {
      // Unused weights. They should never be needed when executing StableHLO.
      continue;
    }

    const ArgSpec& weight_spec = *weights_specs[i];
    const auto& weight_name = weights_[i];
    std::optional<xla::Layout> layout = std::nullopt;
    if (weight_spec.layout.has_value()) {
      layout = *weight_spec.layout;
    }
    IfrtLoadedVariableRegistry::Key key{
        .device_ids = device_ids,
        .input_name = weight_name,
        .hlo_sharding = device_id.has_value()
                            ? xla::HloSharding::SingleDevice(*device_id)
                            : weight_spec.sharding,
        .layout = layout,
    };
    TF_ASSIGN_OR_RETURN(auto loaded_weight,
                        ifrt_loaded_variable_registry_.GetLoadedVariable(key));
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::ArrayRef loaded_weight_ifrt_array,
        loaded_weight.value.Get<tsl::Future<xla::ifrt::ArrayRef>>().Await());
    // Check the shape and DType of loaded weight matches the weight spec.
    // Make sure the shape and dtype of IFRT array for the weight matches the
    // the spec of the weight.
    if (loaded_weight_ifrt_array->shape() != ToIfrtShape(weight_spec.shape)) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " (", weights_[i],
          ") to have shape: ", ToIfrtShape(weight_spec.shape), ", but got ",
          loaded_weight_ifrt_array->shape(), " from the actual weight."));
    }
    if (loaded_weight_ifrt_array->dtype() != weight_spec.dtype) {
      return absl::FailedPreconditionError(absl::StrCat(
          "StableHlo expected weight #", i, " '", weights_[i],
          "' to have dtype: ", weight_spec.dtype, ", but got ",
          loaded_weight_ifrt_array->dtype(), " from the actual weight."));
    }
    TF_RETURN_IF_ERROR(
        InputShapeEqual(weight_spec.shape, loaded_weight_ifrt_array->shape()));

    arrays[i] = std::move(loaded_weight_ifrt_array);
  }
  return arrays;
}

absl::StatusOr<std::vector<xla::ifrt::ArrayRef>>
StableHloRunnerBase::ResolveInputAliasedOutputArrays(
    const CachedExecutableBundle& executable_bundle,
    const xla::ifrt::DeviceListRef& device_list,
    std::vector<xla::ifrt::ArrayRef> caller_provided) {
  const int num_outputs = executable_bundle.num_input_aliased_output;
  TF_RET_CHECK(num_outputs == executable_bundle.io_spec.outputs.size())
      << "Input-aliased output argument count (" << num_outputs
      << ") does not match output count ("
      << executable_bundle.io_spec.outputs.size() << ")";
  TF_RET_CHECK(!device_list->devices().empty())
      << "Device list cannot be empty";
  TF_ASSIGN_OR_RETURN(
      const absl::Span<const xla::ifrt::ShardingRef> output_shardings,
      executable_bundle.GetOutputShardings(device_list->devices()[0]->Id()));

  TF_RET_CHECK(executable_bundle.input_aliased_output_arg_layouts.size() ==
               num_outputs)
      << "Input-aliased output argument layout count ("
      << executable_bundle.input_aliased_output_arg_layouts.size()
      << ") does not match output count (" << num_outputs << ")";

  // Empty means the caller supplies no output memory at all.
  if (caller_provided.empty()) {
    caller_provided.resize(num_outputs);
  }
  if (caller_provided.size() != num_outputs) {
    return absl::InvalidArgumentError(absl::StrCat("Expected ", num_outputs,
                                                   " output arrays, but got ",
                                                   caller_provided.size()));
  }
  for (int i = 0; i < num_outputs; ++i) {
    xla::ifrt::ArrayRef& array = caller_provided[i];
    // The caller supplied no memory for this output, so allocate it. The
    // executable aliases every output, so each one needs an argument.
    if (array == nullptr) {
      TF_ASSIGN_OR_RETURN(
          array, MakeUninitializedOutputArray(
                     *ifrt_client_, executable_bundle.io_spec.outputs[i],
                     output_shardings[i],
                     executable_bundle.input_aliased_output_arg_layouts[i]));
      continue;
    }
    // Say so here rather than letting PjRt fail on it obscurely.
    if (array->IsDeleted()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Output array #", i, " has been deleted."));
    }
    // Verify the caller-provided array matches the output it backs.
    const ArgSpec& output_spec = executable_bundle.io_spec.outputs[i];
    if (array->dtype() != output_spec.dtype) {
      return absl::InvalidArgumentError(
          absl::StrCat("Output array #", i, " has dtype ", array->dtype(),
                       ", but output ", i, " has dtype ", output_spec.dtype));
    }
    if (array->shape() != output_spec.GetIfrtShape()) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Output array #", i, " has shape ", array->shape(), ", but output ",
          i, " has shape ", output_spec.GetIfrtShape()));
    }
    // Resharding here would reintroduce exactly the copies this feature
    // exists to avoid, so an array sharded differently from the output it
    // backs is an error rather than something to fix up.
    if (!array->sharding().HasSamePartitioning(*output_shardings[i])) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Output array #", i, " has sharding ", array->sharding(),
          ", but output ", i, " has sharding ", *output_shardings[i]));
    }
    // PjRt only compares byte sizes when validating execution arguments, so a
    // layout mismatch that preserves the byte count (any permutation of
    // `minor_to_major`) is not caught there. Because the alias is compiled
    // into the executable, such a buffer is still used to back the output and
    // is written using the layout the executable expects, silently producing
    // wrong data. Reject it here instead.
    //
    // An array that reports no layout is using the default layout for its
    // shape on its device rather than having none, so read the concrete layout
    // off the underlying buffer instead of treating null as a mismatch.
    TF_ASSIGN_OR_RETURN(std::shared_ptr<const xla::PjRtLayout> layout,
                        array->pjrt_layout());
    if (layout == nullptr) {
      if (auto* pjrt_array =
              xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleArray>(array.get());
          pjrt_array != nullptr && !pjrt_array->pjrt_buffers().empty()) {
        layout = pjrt_array->pjrt_buffers().front()->layout();
      }
    }
    const std::shared_ptr<const xla::PjRtLayout>& expected_layout =
        executable_bundle.input_aliased_output_arg_layouts[i];
    if (layout != nullptr && expected_layout != nullptr &&
        *layout != *expected_layout) {
      return absl::InvalidArgumentError(absl::StrCat(
          "Output array #", i, " has layout ", layout->ToString(),
          ", but output ", i, " has layout ", expected_layout->ToString()));
    }
    // The execution writes the result into the memory of the argument by
    // consuming it, through donation. Donate an alias of the caller's memory
    // rather than the caller's array, so that the caller keeps its array, and
    // shares the memory with the output instead of handing it over.
    TF_ASSIGN_OR_RETURN(array, MakeAliasingArray(*array, *ifrt_client_,
                                                 AliasAccess::kWritable));
  }
  return caller_provided;
}

absl::StatusOr<std::vector<xla_serving::Value>> StableHloRunnerBase::Execute(
    absl::Span<const xla_serving::Value> inputs, ExecuteOptions options,
    const RunContext& run_context) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode("StableHloRunner::Execute", {});
  });

  VLOG(2) << "Start StableHloRunner execution";

  bool has_streaming = model_io_metadata_.has_streaming;
  int64_t expected_num_inputs = model_io_metadata_.num_user_inputs;
  int32_t xla_launch_id = static_cast<int32_t>(launch_id_.fetch_add(1));
  if (inputs.size() != expected_num_inputs &&
      (!has_streaming || inputs.size() != expected_num_inputs + 1)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Expected ", expected_num_inputs, " user inputs, but got ",
                     inputs.size()));
  }

  // TODO: Consider moving this to `Init()`. Maybe also remove the
  // `call_once` then.
  absl::call_once(weights_lookup_once_, [&]() {
    if (IsPortableExecutable()) {
      for (auto& [device_list, resources] : device_list_to_resources_map_) {
        resources->weight_arrays =
            LookupWeights(device_list->devices()[0]->Id().value());
      }
    } else {
      device_list_to_resources_map_[assigned_device_list_]->weight_arrays =
          LookupWeights();
    }
  });

  // Convert host values of other types (e.g. `tensorflow::Tensor`) to
  // `HostBuffer` with the converter registered for their type. Copy the inputs
  // only if some input needs converting.
  std::vector<xla_serving::Value> host_converted_inputs;
  for (int i = 0; i < inputs.size(); ++i) {
    const xla_serving::Value& input = inputs[i];
    if (input.IsType<HostBuffer>() || input.IsType<xla_serving::DeviceBuffer>()) {
      continue;
    }
    std::shared_ptr<const HostBufferConverter> converter =
        FindHostBufferConverter(input);
    if (converter == nullptr) {
      return absl::InvalidArgumentError(absl::StrCat(
          "StableHloRunner inputs must be HostBuffer or xla_serving::DeviceBuffer. "
          "Input #",
          i,
          " is neither, and no HostBuffer converter is registered for its "
          "type. To pass tensorflow::Tensor inputs, link "
          "//third_party/neptune/runtime/tensorflow:"
          "interpreter_tensorflow_support (it registers the TF converter), or "
          "convert the input to a HostBuffer before calling."));
    }
    absl::StatusOr<HostBuffer> host_buffer = converter->Convert(input);
    if (!host_buffer.ok()) {
      // Keep the converter's code and payloads; only add context.
      const absl::Status& error = host_buffer.status();
      absl::Status annotated(
          error.code(),
          absl::StrCat("Failed to convert StableHloRunner input #", i,
                       " to HostBuffer (converter '", converter->name(),
                       "'): ", error.message()));
      error.ForEachPayload(
          [&](absl::string_view type_url, const absl::Cord& payload) {
            annotated.SetPayload(type_url, payload);
          });
      return annotated;
    }
    if (host_converted_inputs.empty()) {
      host_converted_inputs.assign(inputs.begin(), inputs.end());
    }
    host_converted_inputs[i] = xla_serving::Value(*std::move(host_buffer));
  }
  if (!host_converted_inputs.empty()) {
    inputs = host_converted_inputs;
  }

  std::vector<xla::ifrt::Shape> input_shapes;
  input_shapes.reserve(inputs.size() + 1);
  std::vector<xla::ifrt::DType> input_dtypes;
  input_dtypes.reserve(inputs.size() + 1);

  // Gather shapes before appending streaming IDs to match cache signature.
  for (const auto& input : inputs) {
    if (input.IsType<HostBuffer>()) {
      const auto& host_buf = input.Get<HostBuffer>();
      input_shapes.push_back(host_buf.shape());
      input_dtypes.push_back(host_buf.dtype());
    } else {
      const auto& device_buf = input.Get<xla_serving::DeviceBuffer>();
      input_shapes.push_back(device_buf.ifrt_array->shape());
      input_dtypes.push_back(device_buf.ifrt_array->dtype());
    }
  }

  // If the model is streaming and the caller did not provide the request ID
  // input, we need to append the shape and dtype of the request ID input
  // for the cache lookup.
  if (has_streaming && inputs.size() == expected_num_inputs) {
    if (!run_context.request_ids.empty()) {
      input_shapes.push_back(xla::ifrt::Shape(
          {static_cast<int64_t>(run_context.request_ids.size())}));
    } else {
      input_shapes.push_back(xla::ifrt::Shape({}));
    }
    input_dtypes.push_back(xla::ifrt::DType(xla::ifrt::DType::Kind::kS32));
  }

  TF_ASSIGN_OR_RETURN(
      SharedCachedExecutableBundle executable_bundle,
      LookUpOrCreateExecutable(input_shapes, input_dtypes).Await());

  // Find the resources per the device(s) to use. If the executable is portable,
  // use the core selector to reserve a device first.
  tsl::DeviceReservation device_reservation(/*device_index=*/-1,
                                            /*selector=*/nullptr);
  xla::ifrt::DeviceListRef device_list = assigned_device_list_;
  if (IsPortableExecutable()) {
    device_reservation = serving_core_selector_->ReserveDevice(
        executable_bundle->program_fingerprint_with_timestamp);
    const int device_id = device_reservation.device_index();
    TF_ASSIGN_OR_RETURN(
        xla::ifrt::Device * device,
        ifrt_client_->LookupDevice(xla::ifrt::DeviceId(device_id)));
    TF_ASSIGN_OR_RETURN(device_list, ifrt_client_->MakeDeviceList({device}));
  }
  const auto resources_it = device_list_to_resources_map_.find(device_list);
  if (resources_it == device_list_to_resources_map_.end()) {
    return absl::InternalError(
        absl::StrCat("Device list resources not initialized for device list: ",
                     *device_list));
  }
  TF_ASSIGN_OR_RETURN(const auto& weights_array,
                      resources_it->second->weight_arrays);
  DataConverter* data_converter = resources_it->second->data_converter.get();

  std::vector<xla_serving::Value> unpacked_inputs(inputs.begin(), inputs.end());

  // `used_input_indices` is used to track the index of used inputs in
  // stablehlo spec because some inputs may be presented in the model signature
  // but not used by StableHlo.
  std::vector<int> used_input_indices;
  used_input_indices.reserve(executable_bundle->module_kept_var_indices.size());
  for (int idx : executable_bundle->module_kept_var_indices) {
    if (idx >= weights_array.size()) {
      used_input_indices.push_back(idx - weights_array.size());
    }
  }
  // A model exported for input aliasing ends its argument list with one
  // aliased buffer per output. Those arguments are not supplied by the caller;
  // they are resolved and appended after the user inputs below.
  if (executable_bundle->num_input_aliased_output > 0) {
    if (static_cast<int>(used_input_indices.size()) <
        executable_bundle->num_input_aliased_output) {
      return absl::InternalError(absl::StrCat(
          "Executable aliases ", executable_bundle->num_input_aliased_output,
          " outputs, but only ", used_input_indices.size(),
          " non-weight arguments are kept in its signature."));
    }
    used_input_indices.resize(used_input_indices.size() -
                              executable_bundle->num_input_aliased_output);
  }

  std::vector<HostBuffer> host_buffers_to_convert;
  std::vector<int> host_buffers_stablehlo_indices;
  host_buffers_to_convert.reserve(used_input_indices.size());
  host_buffers_stablehlo_indices.reserve(used_input_indices.size());

  std::vector<std::optional<xla::ifrt::ArrayRef>> final_used_inputs(
      used_input_indices.size(), std::nullopt);

  for (int i = 0; i < used_input_indices.size(); ++i) {
    int idx_in_stablehlo_spec = i + executable_bundle->input_offset;
    int original_input_idx = used_input_indices[i];
    auto& original_input_value = unpacked_inputs[original_input_idx];
    if (original_input_value.IsType<xla_serving::DeviceBuffer>()) {
      auto& device_buffer = original_input_value.Get<xla_serving::DeviceBuffer>();
      const xla::ifrt::Shape target_shape =
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec]
              .GetIfrtShape();
      xla::ifrt::DType target_dtype =
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec].dtype;
      TF_RETURN_IF_ERROR(ValidateDeviceBufferInput(*device_buffer.ifrt_array,
                                                   target_dtype, target_shape,
                                                   idx_in_stablehlo_spec));
      final_used_inputs[i] = std::move(device_buffer.ifrt_array);
    } else if (original_input_value.IsType<HostBuffer>()) {
      auto& original_input = original_input_value.Get<HostBuffer>();
      if (auto it = executable_bundle->host_transpose_permutations.find(
              idx_in_stablehlo_spec);
          it != executable_bundle->host_transpose_permutations.end()) {
        // The optimized module expects this input transposed on the host.
        TF_ASSIGN_OR_RETURN(original_input,
                            TransposeHostBuffer(original_input, it->second));
      }
      const xla::ifrt::Shape target_shape =
          executable_bundle->io_spec.inputs[idx_in_stablehlo_spec]
              .GetIfrtShape();
      if (original_input.shape() == target_shape) {
        // Shapes match, use the buffer directly.
        host_buffers_to_convert.push_back(std::move(original_input));
      } else {
        // Shapes differ (e.g., due to optimization). Perform a zero-copy
        // reshape to match the model's expected signature.
        HostBuffer reshaped_buffer(original_input.base(),
                                   original_input.dtype(), target_shape,
                                   original_input.owner());
        host_buffers_to_convert.push_back(std::move(reshaped_buffer));
      }
      host_buffers_stablehlo_indices.push_back(idx_in_stablehlo_spec);
    }
  }

  std::vector<xla::ifrt::ArrayRef> converted_inputs;
  if (!host_buffers_to_convert.empty()) {
    TF_ASSIGN_OR_RETURN(converted_inputs,
                        data_converter->ConvertInput(
                            host_buffers_to_convert, *executable_bundle,
                            host_buffers_stablehlo_indices));
  }

  int host_buf_idx = 0;
  for (int i = 0; i < used_input_indices.size(); ++i) {
    if (!final_used_inputs[i].has_value()) {
      final_used_inputs[i] = std::move(converted_inputs[host_buf_idx++]);
    }
  }

  std::vector<xla::ifrt::ArrayRef> inputs_with_weights;
  inputs_with_weights.reserve(
      executable_bundle->module_kept_var_indices.size());
  for (int idx : executable_bundle->module_kept_var_indices) {
    if (idx < weights_array.size()) {
      const auto& weight_array = weights_array[idx];
      if (!weight_array.has_value()) {
        return absl::FailedPreconditionError(absl::StrCat(
            "Weight #", idx,
            " is required by the StableHLO program but was not loaded. This "
            "usually indicates the weight was incorrectly treated as unused, "
            "thus not being loaded."));
      }
      inputs_with_weights.push_back(*weight_array);
    }
  }
  // `final_used_inputs` only contains used inputs, so we just insert them
  // at the end of the list.
  for (auto& arr : final_used_inputs) {
    inputs_with_weights.push_back(std::move(*arr));
  }

  // The memory the results are written into is passed in like any other
  // argument, at the very end, and donated to the execution. It is either an
  // alias of memory the caller supplied or memory the runner just allocated;
  // either way the executable aliases output `i` onto argument
  // `num_inputs + i`.
  if (executable_bundle->num_input_aliased_output > 0) {
    TF_ASSIGN_OR_RETURN(
        std::vector<xla::ifrt::ArrayRef> input_aliased_outputs,
        ResolveInputAliasedOutputArrays(*executable_bundle, device_list,
                                        std::move(options.output_arrays)));
    inputs_with_weights.insert(
        inputs_with_weights.end(),
        std::make_move_iterator(input_aliased_outputs.begin()),
        std::make_move_iterator(input_aliased_outputs.end()));
  } else if (absl::c_any_of(options.output_arrays,
                            [](const xla::ifrt::ArrayRef& array) {
                              return array != nullptr;
                            })) {
    return absl::InvalidArgumentError(
        "`ExecuteOptions::output_arrays` requires a model exported for input "
        "aliasing, but this model was not.");
  }

  TF_ASSIGN_OR_RETURN(
      xla::ifrt::LoadedExecutable::ExecuteResult execution_result,
      executable_bundle->ifrt_executable->Execute(
          absl::MakeSpan(inputs_with_weights),
          /*options=*/
          {
              .launch_id = xla_launch_id,
              .fill_status = true,
              // Executions issued by this runner are independent: data
              // dependencies are tracked by IFRT array readiness, and Execute()
              // blocks until completion so chained calls are already ordered.
              // Use a distinct stream per execution so that backends honoring
              // `execution_stream_id` (e.g. PjRt CPU, Pathways) do not
              // serialize concurrent requests on the default stream 0.
              .execution_stream_id =
                  1 + static_cast<uint32_t>(xla_launch_id) %
                          kNumExecutionStreams,
          },
          // IFRT requires an explicit device mapping when executing portable
          //  or unbound executables. If the executable already has
          // baked-in devices (bound), we pass std::nullopt to defer to the
          // binary's ground truth.
          /*devices=*/
          !executable_bundle->ifrt_executable->devices().has_value()
              ? std::make_optional(data_converter->device_list())
              : std::nullopt));
  TF_RETURN_IF_ERROR(execution_result.status.Await());

  std::vector<Placement> placements;
  // Branch 1: Explicit runtime placement overrides.
  // When the caller explicitly supplies `options.placements`, it takes
  // precedence over model manifest declarations.
  if (!options.placements.empty()) {
    if (options.placements.size() != execution_result.outputs.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected ", execution_result.outputs.size(),
                       " placements, but got ", options.placements.size()));
    }
    placements = std::move(options.placements);
  } else {
    // Branch 2: Manifest-declared placement with default host fallback.
    // When runtime placement overrides are absent:
    // 1. Defer to the model manifest placement if explicitly annotated in
    // `io_spec.outputs[i]`.
    // 2. Otherwise, fall back to Placement::kHost (D2H transfer).
    placements.reserve(execution_result.outputs.size());
    for (size_t i = 0; i < execution_result.outputs.size(); ++i) {
      if (i < executable_bundle->io_spec.outputs.size() &&
          executable_bundle->io_spec.outputs[i].placement.has_value()) {
        placements.push_back(*executable_bundle->io_spec.outputs[i].placement);
      } else {
        placements.push_back(Placement::kHost);
      }
    }
  }

  // An output the caller gave host memory for goes to the host, whatever was
  // asked for above: that memory is where the caller expects to find it.
  if (!options.output_host_buffers.empty()) {
    if (options.output_host_buffers.size() != execution_result.outputs.size()) {
      return absl::InvalidArgumentError(
          absl::StrCat("Expected ", execution_result.outputs.size(),
                       " output host buffers, but got ",
                       options.output_host_buffers.size()));
    }
    for (size_t i = 0; i < placements.size(); ++i) {
      if (options.output_host_buffers[i].has_value()) {
        placements[i] = Placement::kHost;
      }
    }
  }

  return data_converter->ConvertOutput(absl::MakeSpan(execution_result.outputs),
                                       *executable_bundle, placements,
                                       options.output_host_buffers);
}

absl::Status StableHloRunnerBase::MarkFullyInitialized() {
  absl::MutexLock lock(mutex_);
  fully_initialized_ = true;
  return absl::OkStatus();
}

}  // namespace xla_serving

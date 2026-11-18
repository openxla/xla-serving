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

#include "runtime/core/dynamic_dma_utils.h"  // from @xla_serving

#include <algorithm>
#include <cstdint>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/layout.h"  // from @xla
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_compiler.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_dtype.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/shape_util.h"  // from @xla
#include "xla/tsl/lib/monitoring/counter.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla

namespace xla_serving {

namespace {

auto* dynamic_dma_bytes_counter = ::tsl::monitoring::Counter<1>::New(
    "/xla_serving/input_transfer/dynamic_dma_bytes",
    "Bytes of inputs transferred with dynamic DMA. `allocated` counts the "
    "device buffers, which have the static shape. `transferred` counts the "
    "host data copied into them.",
    "kind");

}  // namespace

absl::StatusOr<DynamicDmaTransfer> PlanDynamicDmaTransfer(
    const HostBuffer& input, const xla::ifrt::MemoryKind& memory_kind,
    const H2DTransferExecutor::StaticShape& static_shape,
    const xla::CommonPjRtClient& client, xla::PjRtMemorySpace* memory_space) {
  // The host data must be a dimension-0 prefix of the device buffer.
  const absl::Span<const int64_t> dims = input.shape().dims();
  const absl::Span<const int64_t> static_dims = static_shape.shape.dims();
  if (dims.empty() || dims.size() != static_dims.size() ||
      dims[0] > static_dims[0] || dims.subspan(1) != static_dims.subspan(1)) {
    return absl::InvalidArgumentError(
        absl::StrCat("Input of shape ", input.shape(),
                     " is not a dimension-0 prefix of its static shape ",
                     static_shape.shape, "."));
  }
  // Strings and packed sub-byte types are not copied to the device as they
  // are stored on the host.
  if (input.dtype().kind() == xla::ifrt::DType::kString ||
      input.dtype() != input.storage_dtype()) {
    return absl::UnimplementedError(
        absl::StrCat("Dynamic DMA does not support inputs of dtype ",
                     input.dtype(), " (input of shape ", input.shape(),
                     " with static shape ", static_shape.shape, ")."));
  }
  if (!memory_kind.is_default()) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA only supports the default memory kind, but the input of "
        "shape ",
        input.shape(), " with static shape ", static_shape.shape,
        " has memory kind ", memory_kind, "."));
  }
  if (xla::IsTpuId(client.platform_id()) &&
      input.dtype().bit_size().value_or(0) > 32) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA does not support element types wider than 32 bits on "
        "TPU, but the input of shape ",
        input.shape(), " with static shape ", static_shape.shape, " has dtype ",
        input.dtype(), "."));
  }
  if (static_shape.shape.num_elements() == 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA does not support a static shape without "
        "elements, but the input of shape ",
        input.shape(), " has static shape ", static_shape.shape, "."));
  }
  TF_ASSIGN_OR_RETURN(xla::PrimitiveType primitive_type,
                      xla::ifrt::ToPrimitiveType(input.dtype()));

  DynamicDmaTransfer transfer;
  // The device buffer has the static shape in the layout that the executable
  // expects.
  const xla::Layout* layout = static_shape.layout != nullptr
                                  ? &static_shape.layout->xla_layout()
                                  : nullptr;
  TF_ASSIGN_OR_RETURN(
      transfer.device_shape,
      client.MakeDefaultShapeForMemorySpace(
          memory_space, xla::ShapeUtil::MakeShape(primitive_type, static_dims),
          layout));
  // The host data is linearized as an array of that layout with fewer rows.
  // It is a prefix of the device buffer only if dimension 0 is the major-most
  // dimension of the layout.
  if (transfer.device_shape.layout().minor_to_major().empty() ||
      transfer.device_shape.layout().minor_to_major().back() != 0) {
    return absl::UnimplementedError(absl::StrCat(
        "Dynamic DMA requires dimension 0 to be the major-most dimension of "
        "the device layout, but the device shape for static shape ",
        static_shape.shape, " is ",
        transfer.device_shape.ToString(/*print_layout=*/true), "."));
  }
  TF_ASSIGN_OR_RETURN(
      transfer.device_bytes,
      client.GetOnDeviceBytesCount(memory_space, transfer.device_shape));
  transfer.transfer_shape = transfer.device_shape;
  transfer.transfer_shape.set_dimensions(0, dims[0]);
  if (dims[0] == 0) {
    // Nothing to linearize or to copy.
    return transfer;
  }
  TF_ASSIGN_OR_RETURN(
      transfer.linearized_bytes,
      client.GetOnDeviceBytesCount(memory_space, transfer.transfer_shape));
  if (transfer.linearized_bytes > transfer.device_bytes) {
    return absl::InternalError(absl::StrCat(
        "The rows of an input take ", transfer.linearized_bytes, " bytes as ",
        transfer.transfer_shape.ToString(/*print_layout=*/true),
        ", more than the ", transfer.device_bytes,
        " bytes of the device buffer ",
        transfer.device_shape.ToString(/*print_layout=*/true), "."));
  }
  // The device may round the size of an array up, for example to whole tiles.
  // Only the bytes up to the end of the last DMA granule with data have to be
  // copied.
  TF_ASSIGN_OR_RETURN(int64_t dma_bytes,
                      client.GetDmaByteCount(transfer.transfer_shape));
  transfer.dma_bytes = std::min(dma_bytes, transfer.linearized_bytes);
  return transfer;
}

void RecordDynamicDmaBytes(int64_t allocated_bytes, int64_t transferred_bytes) {
  dynamic_dma_bytes_counter->GetCell("allocated")->IncrementBy(allocated_bytes);
  dynamic_dma_bytes_counter->GetCell("transferred")
      ->IncrementBy(transferred_bytes);
}

}  // namespace xla_serving

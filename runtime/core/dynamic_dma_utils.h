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

#ifndef XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_UTILS_H_
#define XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_UTILS_H_

#include <cstdint>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/shape.h"  // from @xla

namespace xla_serving {

// Dynamic DMA transfers an input that has fewer rows than its static shape
// (`H2DTransferExecutor::StaticShape`). The device buffer is allocated for the
// static shape, and only the rows of the input are copied, to the front of
// it. The rest of the buffer is left uninitialized. The H2D transfer executors
// share the helpers below for it.

// The shapes and sizes of a dynamic DMA transfer.
struct DynamicDmaTransfer {
  // The static shape in the layout that the executable expects. The device
  // buffer has this shape and takes `device_bytes` bytes.
  xla::Shape device_shape;
  int64_t device_bytes = 0;
  // `device_shape` with the rows of the input along dimension 0. The host data
  // is linearized as an array of this shape, into `linearized_bytes` bytes.
  xla::Shape transfer_shape;
  int64_t linearized_bytes = 0;
  // The leading bytes of the linearized data that have to be copied to the
  // front of the device buffer. The bytes after them are padding. Zero if the
  // input has no rows.
  int64_t dma_bytes = 0;
};

// Returns how to transfer `input`, which has fewer rows than `static_shape`,
// to `memory_space` of `client` with dynamic DMA. `memory_kind` is the memory
// kind of the sharding of the input.
//
// Returns InvalidArgument if the input is not a dimension-0 prefix of the
// static shape. Returns Unimplemented if the rows of the input would not be
// the front of the device buffer, or cannot be copied as they are:
//
// - Dimension 0 is not the major-most dimension of the layout. The layout is
//   the one in `static_shape`, or the default layout of the device for the
//   static shape if that is null. TPU chooses its layouts to minimize padding,
//   and for an array of rank 2 or more the major-most dimension is often
//   another one (f32[8,2] gets `{0,1:T(2,128)}`). On TPU such an input has to
//   be flattened to rank 1 first.
// - On TPU, the element type is wider than 32 bits. TPU stores a wider
//   element as several 32-bit words, in one plane per word: the buffer of an
//   s64 array holds the low words of all elements and then the high words.
//   The high words of an input with fewer rows would not be where the program
//   reads them.
// - The input holds strings, or elements of a packed sub-byte type.
// - The memory kind is not the default one.
// - The static shape has no elements.
absl::StatusOr<DynamicDmaTransfer> PlanDynamicDmaTransfer(
    const HostBuffer& input, const xla::ifrt::MemoryKind& memory_kind,
    const H2DTransferExecutor::StaticShape& static_shape,
    const xla::CommonPjRtClient& client, xla::PjRtMemorySpace* memory_space);

// Adds the bytes that dynamic DMA transfers allocated on the device and the
// bytes that they copied there to the cells "allocated" and "transferred" of
// `/xla_serving/input_transfer/dynamic_dma_bytes`.
void RecordDynamicDmaBytes(int64_t allocated_bytes, int64_t transferred_bytes);

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_UTILS_H_

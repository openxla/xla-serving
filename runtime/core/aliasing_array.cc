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

#include "runtime/core/aliasing_array.h"  // from @xla_serving

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "xla/pjrt/common_pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/pjrt_layout.h"  // from @xla
#include "xla/pjrt/raw_buffer.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/rtti.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_array.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/shape.h"  // from @xla
#include "xla/tsl/concurrency/ref_count.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace xla_serving {
namespace {

// Returns a buffer in `memory_space` that aliases the device memory of
// `source`. See `MakeAliasingArray`.
absl::StatusOr<std::shared_ptr<xla::PjRtBuffer>> MakeAliasingBuffer(
    std::shared_ptr<xla::PjRtBuffer> source, xla::PjRtMemorySpace* memory_space,
    AliasAccess access) {
  auto* client = dynamic_cast<xla::CommonPjRtClient*>(memory_space->client());
  if (client == nullptr) {
    return absl::UnimplementedError(
        "Aliasing device memory is only supported on clients built on "
        "`xla::CommonPjRtClient`.");
  }

  // The view has no definition events: nothing that reads or writes it waits
  // for the work that defines `source`. Make sure that work is done instead.
  TF_RETURN_IF_ERROR(source->GetReadyFuture().Await());

  // Share `source`'s raw buffer rather than importing its device memory as
  // foreign memory: not every backend can import device memory (e.g. TPU), and
  // the raw buffer is reference counted, so it keeps the memory allocated for
  // as long as `source` or the view uses it. The view is as mutable as
  // `source`.
  TF_ASSIGN_OR_RETURN(xla::PjRtRawBufferRef raw_buffer,
                      xla::PjRtRawBuffer::CreateRawAliasOfBuffer(source.get()));
  if (access == AliasAccess::kWritable && !raw_buffer->is_mutable()) {
    return absl::FailedPreconditionError(
        "Cannot write into an alias of immutable device memory.");
  }
  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<xla::PjRtBuffer> view,
      client->DefineBuffer(source->on_device_shape(), memory_space,
                           std::move(raw_buffer),
                           /*definition_device_events=*/{}));
  return std::shared_ptr<xla::PjRtBuffer>(std::move(view));
}

}  // namespace

absl::StatusOr<xla::ifrt::ArrayRef> MakeAliasingArray(
    xla::ifrt::Array& source, xla::ifrt::Client& target_client,
    AliasAccess access) {
  auto* pjrt_source =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleArray>(&source);
  auto* pjrt_target_client =
      xla::ifrt::dyn_cast<xla::ifrt::PjRtCompatibleClient>(&target_client);
  if (pjrt_source == nullptr || pjrt_target_client == nullptr) {
    return absl::UnimplementedError(
        "Aliasing device memory is only supported for PjRt-backed arrays and "
        "clients.");
  }
  if (source.client() != &target_client) {
    return absl::UnimplementedError(
        "Aliasing the device memory of an array of a different client is not "
        "supported.");
  }

  TF_ASSIGN_OR_RETURN(std::shared_ptr<const xla::PjRtLayout> layout,
                      source.pjrt_layout());
  xla::ifrt::PjRtCompatibleClient::PjRtBuffers views;
  for (const std::shared_ptr<xla::PjRtBuffer>& buffer :
       pjrt_source->pjrt_buffers()) {
    if (buffer == nullptr) {
      return absl::InvalidArgumentError(
          "Cannot alias an array that has a missing buffer.");
    }
    xla::PjRtMemorySpace* const memory_space = buffer->memory_space();
    TF_ASSIGN_OR_RETURN(std::shared_ptr<xla::PjRtBuffer> view,
                        MakeAliasingBuffer(buffer, memory_space, access));
    views.push_back(std::move(view));
  }
  TF_ASSIGN_OR_RETURN(
      tsl::RCReference<xla::ifrt::PjRtArray> array,
      xla::ifrt::PjRtArray::Create(pjrt_target_client, source.dtype(),
                                   source.shape(), source.shared_ptr_sharding(),
                                   std::move(views), std::move(layout)));
  return array;
}

}  // namespace xla_serving

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_

#include <memory>
#include <vector>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @neptune
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace neptune {

// Interface for synchronous Host-to-Device (Arena) Transfers.
// Implementations of this interface handle linear packing of HostBuffers along
// with efficient multi-slice allocation & execution to the underlying devices.
//
// NOTE: Any concrete implementation of this interface MUST be completely
// thread-safe and stateless (or internally synchronized), because
// `ExecuteTransfer` will be invoked concurrently by multiple threads from the
// inference serving thread pool.
class H2DTransferExecutor {
 public:
  virtual ~H2DTransferExecutor() = default;

  // Converts the given `inputs` (HostBuffers) to XLA IFRT Arrays.
  // Performs synchronous transfer of inputs to avoid unnecessary future
  // overhead.
  //
  // `inputs`: The user-provided HostBuffers to transfer.
  // `input_shardings`: The exact IFRT Shardings required for these arrays.
  virtual absl::StatusOr<std::vector<xla::ifrt::ArrayRef>> ExecuteTransfer(
      absl::Span<const HostBuffer> inputs,
      absl::Span<const xla::ifrt::ShardingRef> input_shardings) = 0;
};

// Factory interface for creating an `H2DTransferExecutor`.
class H2DTransferExecutorFactory {
 public:
  virtual ~H2DTransferExecutorFactory() = default;

  // Creates the `H2DTransferExecutor`.
  //
  // `client`: The IFRT client used for allocation/transfer.
  // `device_list`: The devices targeted for this execution.
  // `thread_pool`: A thread pool to use for concurrent packing/conversion.
  virtual absl::StatusOr<std::unique_ptr<H2DTransferExecutor>> Create(
      std::shared_ptr<xla::ifrt::Client> client,
      xla::ifrt::DeviceListRef device_list,
      tsl::thread::ThreadPool* thread_pool) = 0;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_H_

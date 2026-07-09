#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_

#include <memory>

#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla

namespace neptune {

// A metadata-rich container representing an array that already resides in
// device memory. This struct bundles the physical `xla::ifrt::ArrayRef` with
// the execution context (client, device list) and the `xla::ifrt::ShardingRef`
// layout. Downstream operators require this metadata to accurately orchestrate
// Device-to-Host (D2H) transfers (e.g., via MakeHostBufferFromArray).
// (Do not add constructors or destructors to this struct. We want it to remain
// a simple aggregate so we can use designated initializers).
struct DeviceBuffer {
  // Shared context used for underlying IFRT operations (e.g. transfers).
  std::shared_ptr<xla::ifrt::Client> ifrt_client;
  // The physical on-device array payload.
  xla::ifrt::ArrayRef ifrt_array;
  // The IFRT sharding (physical/logical layout) of the array.
  xla::ifrt::ShardingRef sharding;
  // The specific devices across which the array is distributed.
  xla::ifrt::DeviceListRef device_list;
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_CORE_DEVICE_BUFFER_H_

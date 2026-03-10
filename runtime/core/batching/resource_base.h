#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_RESOURCE_BASE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_RESOURCE_BASE_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"  // from @com_google_absl
#include "tsl/platform/refcount.h"  // from @tsl

namespace jsv {

// This is the base class of all resource classes. Each resource must be
// represented as a subclass of ResourceBase (which is reference counted) to be
// able to work with resource facilities such ResourceHandle and ResourceMgr.
class ResourceBase : public tsl::core::WeakRefCounted {
 public:
  // Returns a debug string for *this.
  virtual std::string DebugString() const = 0;

  // Returns a name for ref-counting handles.
  virtual std::string MakeRefCountingHandleName(int64_t resource_id) const {
    return absl::StrFormat("Resource-%d-at-%p", resource_id, this);
  }

  // Returns memory used by this resource.
  virtual int64_t MemoryUsed() const { return 0; }
};
}  //  namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_RESOURCE_BASE_H_

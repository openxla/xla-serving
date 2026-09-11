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

#ifndef XLA_SERVING_RUNTIME_BATCHING_RESOURCE_BASE_H_
#define XLA_SERVING_RUNTIME_BATCHING_RESOURCE_BASE_H_

#include <cstdint>
#include <string>

#include "absl/strings/str_format.h"  // from @abseil-cpp
#include "tsl/platform/refcount.h"  // from @tsl

namespace xla_serving {

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
}  //  namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_BATCHING_RESOURCE_BASE_H_

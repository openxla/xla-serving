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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_SHARED_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_SHARED_H_

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "tensorstore/context.h"  // from @tensorstore
#include "tensorstore/util/result.h"  // from @tensorstore

namespace neptune {


inline absl::StatusOr<tensorstore::Context> GetOcdbtContext() {
  auto context = tensorstore::Context::FromJson(
      {{"file_io_concurrency", {{"limit", 128}}},
       {"cache_pool#ocdbt", {{"total_bytes_limit", 100000000}}}});
  if (!context.ok()) {
    return tensorstore::GetStatus(context);
  }
  return context.value();
}

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_SHARED_H_

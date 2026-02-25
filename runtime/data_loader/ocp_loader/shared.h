#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_SHARED_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_SHARED_H_

#include "third_party/absl/status/statusor.h"
#include "third_party/tensorstore/context.h"
#include "third_party/tensorstore/util/result.h"

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

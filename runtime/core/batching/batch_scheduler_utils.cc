#include "runtime/core/batching/batch_scheduler_utils.h"  // from @neptune

#include <algorithm>
#include <cstdint>
#include <vector>

#include "absl/algorithm/container.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp

namespace jsv {

int GetNextAllowedBatchSize(int batch_size,
                            const std::vector<int32_t>& allowed_batch_sizes,
                            bool disable_padding) {
  if (disable_padding || allowed_batch_sizes.empty()) {
    return batch_size;
  }
  DCHECK(absl::c_is_sorted(allowed_batch_sizes));
  DCHECK_GT(batch_size, 0);
  for (int allowed_size : allowed_batch_sizes) {
    if (allowed_size >= batch_size) {
      return allowed_size;
    }
  }
  LOG(ERROR) << "Batch size " << batch_size
             << " is greater than largest allowed size; ignoring allowed sizes "
                "constraint.";
  return batch_size;
}

int32_t GetPrevAllowedBatchSize(int batch_size,
                                const std::vector<int32_t>& allowed_batch_sizes,
                                bool disable_padding) {
  if (disable_padding || allowed_batch_sizes.empty()) {
    return batch_size;
  }

  DCHECK(absl::c_is_sorted(allowed_batch_sizes));
  DCHECK_GT(batch_size, 0);

  // First from the end allowed_batch_size not larger than batch_size.
  auto result = std::find_if(
      allowed_batch_sizes.rbegin(), allowed_batch_sizes.rend(),
      [&](int allowed_size) { return allowed_size <= batch_size; });

  if (result == allowed_batch_sizes.rend()) {
    // No such element exists.
    return batch_size;
  }

  return *result;
}

}  // namespace jsv

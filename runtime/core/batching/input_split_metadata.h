#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_INPUT_SPLIT_METADATA_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_INPUT_SPLIT_METADATA_H_

#include <string>

#include "absl/container/fixed_array.h"  // from @abseil-cpp

namespace neptune {
namespace internal {
// InputSplitMetadata represents the task sizes of a batch task after it's
// tailored according to queue status (`open_batch_remaining_slot` and
// `batch_size_limit`).
//
// This is an internal helper class, and the implementation is shared
// shared across different instantiations of internal::Queue<TaskType>
// in input-split mode (QueueOptions.enable_large_batch_splitting is true).
class InputSplitMetadata {
 public:
  InputSplitMetadata(int input_task_size, int open_batch_remaining_slot,
                     int batch_size_limit);

  // Returns underlying task sizes.
  const absl::FixedArray<int>& task_sizes() const;

  // Serializes task split metadata into a string for debugging.
  std::string DebugString() const;

 private:
  absl::FixedArray<int> generate_task_sizes(int input_task_size,
                                            int open_batch_remaining_slot,
                                            int batch_size_limit) const;

  const absl::FixedArray<int> task_sizes_;
};
}  // namespace internal
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_BATCHING_INPUT_SPLIT_METADATA_H_

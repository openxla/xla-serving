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

#ifndef XLA_SERVING_RUNTIME_KERNEL_BATCHING_UPPER_BOUND_INPUT_PROVIDER_H_
#define XLA_SERVING_RUNTIME_KERNEL_BATCHING_UPPER_BOUND_INPUT_PROVIDER_H_

#include <optional>
#include <vector>

#include "xla/python/ifrt/shape.h"  // from @xla

namespace xla_serving {

// Implemented by kernels that accept an input with fewer rows than its compiled
// shape while still returning outputs of the compiled batch size.
// `CreateBatchRunnerOp()` queries the first batched kernel with
// `dynamic_cast<const UpperBoundInputProvider*>()` and skips host padding on
// the marked inputs.
class UpperBoundInputProvider {
 public:
  virtual ~UpperBoundInputProvider() = default;

  // Returns the shapes of the inputs whose dimension 0 is an upper bound.
  // Either empty, or one entry per shape bucket that the kernel is compiled
  // for. A bucket has one entry per input, where `std::nullopt` marks an input
  // that must be passed with its full shape, and the same inputs are marked in
  // every bucket. The kernel runs its inputs on the smallest bucket that fits
  // them.
  virtual std::vector<std::vector<std::optional<xla::ifrt::Shape>>>
  upper_bound_input_shapes() const = 0;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_BATCHING_UPPER_BOUND_INPUT_PROVIDER_H_

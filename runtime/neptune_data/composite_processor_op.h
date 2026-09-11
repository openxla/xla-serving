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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_OP_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_OP_H_

#include <memory>
#include <string>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "data/processor/cc/core/composite_processor.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune

namespace xla_serving {
namespace composite {

// A kernel that executes a neptune::data::composite::CompositeProcessor.
class CompositeProcessorOp : public CancellableKernel {
 public:
  explicit CompositeProcessorOp(
      absl_nonnull
      std::unique_ptr<neptune::data::composite::core::CompositeProcessor>
          processor);

  std::string Name() const override;
  IODataTypes GetIODataTypes() const override;
  int GetNumInputs() const override;
  int GetNumOutputs() const override;

 private:
  absl::Status RunInternal(ExecuteContext& context) override;

  absl_nonnull
  std::unique_ptr<neptune::data::composite::core::CompositeProcessor>
      processor_;
};

}  // namespace composite
}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_NEPTUNE_DATA_COMPOSITE_PROCESSOR_OP_H_

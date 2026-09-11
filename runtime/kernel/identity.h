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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_

#include <memory>
#include <string>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune

namespace xla_serving {
class IdentityKernel : public CancellableKernel {
 public:
  ~IdentityKernel() override = default;

 public:
  static absl::StatusOr<std::unique_ptr<Kernel>> Create(
      const KernelBuilderContext& context);

  std::string Name() const override;

  Kernel::IODataTypes GetIODataTypes() const override {
    // TODO: add new data types for the identity kernel or remove
    // this kernel.
    return {.input = KernelIODataType::kHostBuffer,
            .output = KernelIODataType::kHostBuffer};
  }

  int GetNumInputs() const override { return num_host_buffers_; }

  int GetNumOutputs() const override { return num_host_buffers_; }

 private:
  explicit IdentityKernel(int num_host_buffers)
      : num_host_buffers_(num_host_buffers) {}

  absl::Status RunInternal(ExecuteContext& context) override;

  int num_host_buffers_;
};
}  // namespace xla_serving

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_

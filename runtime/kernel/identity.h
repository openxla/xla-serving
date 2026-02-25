#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_

#include <memory>
#include <string>

#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/neptune/runtime/kernel/kernel_builder_context.h"
#include "third_party/neptune/runtime/kernel/kernel_io_data_type.h"

namespace jsv {
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
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_IDENTITY_H_

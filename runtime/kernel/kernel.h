#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/interpreter/context.h"  // from @neptune
#include "absl/base/thread_annotations.h"  // from @abseil-cpp
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/functional/any_invocable.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/synchronization/mutex.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/kernel/kernel_io_data_type.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla

namespace neptune {

// Kernel interface definition. All kernels should implement this interface.
class Kernel {
 public:
  // The data type of the kernel input and output.
  struct IODataTypes {
    KernelIODataType input;
    KernelIODataType output;
  };

  struct ExecuteMetadata {
    absl::string_view model_name;
    int64_t model_version;
    uint64_t trace_id;
  };

  // The context of the kernel execution. The kernel is expected to take the
  // inputs from the `context.inputs` and put the outputs to the
  // `context.outputs`.
  struct ExecuteContext {
    ExecuteMetadata metadata;
    absl::Span<const Value> inputs;
    std::vector<Value> outputs;  // allocated by the kernel.
    RunContext run_context;
    // Holds name-value pairs for tensors in context that can be used by
    // kernels to get input based on keys.
    // TODO: Implement an eviction mechanism for entries in this
    // map that are no longer needed.
    absl::flat_hash_map<std::string, Value> named_values;
  };

  virtual ~Kernel() = default;

  // Gets the name of the kernel for debugging. DO NOT use the name to infer the
  // runtime type of a kernel instance.
  virtual std::string Name() const = 0;

  // Runs the kernel by taking the inputs from the `context.inputs` and putting
  // the outputs to the `context.outputs`.
  virtual absl::Status Run(ExecuteContext& context) = 0;

  // Returns true if the kernel can be run asynchronously.
  virtual bool CanRunAsync() { return false; }

  // Returns the data type of the kernel input and output.
  virtual IODataTypes GetIODataTypes() const = 0;

  // Returns the number of inputs  of the kernel.
  virtual int GetNumInputs() const = 0;

  // Returns the number of outputs of the kernel.
  virtual int GetNumOutputs() const = 0;

  // Marks the kernel to fully initialized. After this is called, time-consuming
  // tasks not suitable for the production environment will be disallowed.
  virtual absl::Status MarkFullyInitialized() { return absl::OkStatus(); }
};

class CancellableKernel : public Kernel {
 public:
  // Runs the kernel by taking the inputs from the `context.inputs` and putting
  // the outputs to the `context.outputs`. Returns `absl::CancelledError` upon
  // cancellation.
  absl::Status Run(ExecuteContext& context) override;

 private:
  virtual absl::Status RunInternal(ExecuteContext& context) = 0;
};

class CancellableAsyncKernel : public Kernel {
 public:
  using Kernel::Kernel;

  // Only `CancellableAsyncKernel` can be run asynchronously.
  bool CanRunAsync() override { return true; }

  // Runs the kernel asynchronously. The kernel is expected to take the inputs
  // from the `context.inputs` and put the outputs to the `context.outputs`.
  // `context` (including the object backing `context.inputs`) is expected
  // to be valid until the returned future is ready. Returns
  // `absl::CancelledError` upon cancellation.
  tsl::Future<> RunAsync(ExecuteContext& context);

  absl::Status Run(ExecuteContext& context) override {
    return RunAsync(context).Await();
  }

 private:
  virtual tsl::Future<> RunAsyncInternal(ExecuteContext& context) = 0;
};

// Registries all kernels. Each type of kernel should have a unique name and
// register its builder here.
class KernelRegistry {
 public:
  absl::Status Register(
      absl::string_view name,
      absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Kernel>>(
          const KernelBuilderContext& context)>
          builder);

  absl::StatusOr<std::unique_ptr<Kernel>> Create(
      absl::string_view name, const KernelBuilderContext& context);

 private:
  mutable absl::Mutex mutex_;
  absl::flat_hash_map<
      std::string, absl::AnyInvocable<absl::StatusOr<std::unique_ptr<Kernel>>(
                       const KernelBuilderContext& context)>>
      kernel_builders_ ABSL_GUARDED_BY(mutex_);
};

KernelRegistry& GetGlobalKernelRegistry();

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_H_

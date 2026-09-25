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

#ifndef XLA_SERVING_RUNTIME_STABLEHLO_RUNNER_H_
#define XLA_SERVING_RUNTIME_STABLEHLO_RUNNER_H_

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/loaded_stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/stablehlo_runner_base.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/dtype.h"  // from @xla
#include "xla/python/ifrt/shape.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

class StableHloRunner : public StableHloRunnerBase {
 public:
  static absl::StatusOr<std::unique_ptr<StableHloRunner>> Create(
      absl_nonnull std::shared_ptr<StableHlo> stablehlo,
      std::optional<xla::CompileOptionsProto> compile_options,
      absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
      IfrtPersistentCompilationCache* absl_nullable
          persistent_compilation_cache,
      IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
      std::vector<std::string> weights,
      tsl::protobuf::Message* absl_nullable compilation_environment_proto,
      tsl::thread::ThreadPool* absl_nonnull thread_pool,
      StablehloOptimizer* absl_nullable optimizer,
      absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
      ServingCoreSelector* absl_nullable serving_core_selector,
      std::unique_ptr<H2DTransferExecutorFactory>
          h2d_transfer_executor_factory = nullptr,
      std::optional<int64_t> callback_id = std::nullopt);

  // Virtual destructor is required for safe polymorphic destruction of
  // subclasses.
  ~StableHloRunner() override = default;

  // Marks the runner to fully initialized. Releases all MLIR modules.
  absl::Status MarkFullyInitialized() override;

 private:
  // 'weights' are expected to be in the same order as the variable arguments in
  // the computation. It is also assumed that variables will always be at the
  // beginning of the argument list.
  StableHloRunner(
      absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo,
      std::optional<xla::CompileOptionsProto> compile_options,
      absl_nonnull std::shared_ptr<xla::ifrt::Client> client,
      IfrtLoadedVariableRegistry* absl_nonnull ifrt_loaded_variable_registry,
      std::vector<std::string> weights,
      xla::ifrt::DeviceListRef assigned_device_list,
      tsl::protobuf::Message* compilation_environment_proto,
      IfrtPersistentCompilationCache* absl_nullable
          persistent_compilation_cache,
      StablehloOptimizer* absl_nullable optimizer,
      absl_nonnull std::shared_ptr<const AssignedDevices> assigned_devices,
      ServingCoreSelector* absl_nullable serving_core_selector,
      std::unique_ptr<H2DTransferExecutorFactory> h2d_transfer_executor_factory,
      std::optional<int64_t> callback_id);

  // Looks up the executable bundle in the cache, uses the shapes of the
  // input (excluding the weights) to determine which executable to use. If not
  // found, creates a new executable bundle and inserts it into the cache.
  // Args
  // - `input_shapes` and `input_dtypes` are the shapes and dtypes of the
  // inputs (excluding the weights) irrespective of whether they are used by the
  // module "main" function.
  tsl::Future<SharedCachedExecutableBundle> LookUpOrCreateExecutable(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes) override;

  // Creates a new executable bundle synchronously.
  // `input_shapes` and `input_dtypes` are the shapes and dtypes of the inputs
  // (excluding the weights) irrespective of whether they are used by the
  // module "main" function.
  absl::StatusOr<SharedCachedExecutableBundle> CreateExecutableSynchronously(
      absl::Span<const xla::ifrt::Shape> input_shapes,
      absl::Span<const xla::ifrt::DType> input_dtypes,
      std::unique_ptr<LoadedStableHlo::ModuleBundle> module_bundle,
      const StablehloOptimizer::Args& args, bool is_streaming = false);

  absl_nonnull std::unique_ptr<LoadedStableHlo> loaded_stablehlo_;
  std::optional<xla::CompileOptionsProto> compile_options_;
  // The persistent compilation cache is a global cache and is not owned by
  // this StableHloRunner. When it is nullptr, the persistent compilation cache
  // is disabled at ifrt serving level.
  IfrtPersistentCompilationCache* absl_nullable persistent_compilation_cache_;
  tsl::protobuf::Message* absl_nullable
      compilation_environment_proto_;  // NOT OWNED.
  // If nullptr, no optimization will be enabled.
  StablehloOptimizer* absl_nullable optimizer_;

  friend class StableHloRunnerTestPeer;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_STABLEHLO_RUNNER_H_

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

#ifndef XLA_SERVING_RUNTIME_KERNEL_TEST_UTILS_H_
#define XLA_SERVING_RUNTIME_KERNEL_TEST_UTILS_H_

#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "xla/tsl/platform/resource_loader.h"
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/host_buffer.h"  // from @xla_serving
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @xla_serving
#include "runtime/core/sharding_utils.h"  // from @xla_serving
#include "runtime/core/stablehlo.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/core/value.h"  // from @xla_serving
#include "runtime/interpreter/manifest_utils.h"  // from @xla_serving
#include "runtime/kernel/kernel.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_constants.h"  // from @xla_serving
#include "runtime/kernel/kernel_builder_context.h"  // from @xla_serving
#include "runtime/kernel/stablehlo_runner_op.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "xla/hlo/ir/hlo_sharding.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device_list.h"  // from @xla
#include "xla/python/ifrt/memory.h"  // from @xla
#include "xla/python/ifrt/sharding.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/xla_sharding.h"  // from @xla
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/status.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/platform/protobuf.h"  // from @tsl

namespace xla_serving {

class StableHloRunnerOpTestHelper {
 public:
  explicit StableHloRunnerOpTestHelper() {
    auto client = xla::ifrt::test_util::GetClient();
    TF_CHECK_OK(client.status());
    client_ = *std::move(client);

    thread_pool_ = std::make_unique<tsl::thread::ThreadPool>(
        tsl::Env::Default(), tsl::ThreadOptions(), "IfrtSharding",
        kThreadPoolNumThreads);
  }

  absl::StatusOr<std::string> GetStableHloStrFromDataDependency(
      absl::string_view data_dependency_path) {
    std::string mlir_path = tsl::GetDataDependencyFilepath(
        std::string(data_dependency_path));
    std::string stablehlo_str;
    absl::Status status =
        tsl::ReadFileToString(tsl::Env::Default(), mlir_path, &stablehlo_str);
    if (!status.ok()) {
      LOG(ERROR) << "Failed to read StableHlo from " << mlir_path << ": "
                 << status;
      return status;
    }
    return stablehlo_str;
  }

  absl::StatusOr<std::unique_ptr<Kernel>> MakeRunnerOp(
      std::shared_ptr<const StableHlo::Spec> stablehlo_spec,
      std::shared_ptr<const AssignedDevices> assigned_devices) {
    KernelBuilderContext builder_context;
    TF_ASSIGN_OR_RETURN(std::shared_ptr<StableHlo> stablehlo,
                        StableHlo::Create({*stablehlo_spec}));
    TF_RETURN_IF_ERROR(builder_context.SetConfig(kStableHlo, stablehlo));
    TF_RETURN_IF_ERROR(builder_context.SetConfig(kIfrtClient, client_));
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kPersistentCompilationCache,
        static_cast<xla_serving::IfrtPersistentCompilationCache*>(nullptr)));
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kIfrtLoadedVariableRegistry, &ifrt_loaded_variable_registry_));
    // This function needs to be called after all the variables are registered.
    TF_RETURN_IF_ERROR(builder_context.SetConfig(kWeights, loaded_variables_));
    TF_RETURN_IF_ERROR(
        builder_context.SetConfig(kIfrtWorkerThreadPool, thread_pool_.get()));
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kCompilationEnvProto, static_cast<tsl::protobuf::Message*>(nullptr)));
    TF_RETURN_IF_ERROR(
        builder_context.SetConfig(kAssignedDevices, assigned_devices));
    // Set the optimizer to nullptr and not test it on
    // `stablehlo_runner_op_test` because this behavior is tested in
    // `stablehlo_runner_test` and `stablehlo_runner_tpu_test`.
    TF_RETURN_IF_ERROR(builder_context.SetConfig(
        kStablehloOptimizer, static_cast<StablehloOptimizer*>(nullptr)));
    return StableHloRunnerOp::Create(builder_context);
  }

  absl::Status RegisterVariable(const HostBuffer& host_buffer,
                                absl::Span<const int> device_ids,
                                xla::OpSharding op_sharding,
                                absl::string_view variable_name) {
    TF_ASSIGN_OR_RETURN(xla::HloSharding hlo_sharding,
                        xla::HloSharding::FromProto(op_sharding));

    auto loaded_variable_producer =
        [&]() -> absl::StatusOr<IfrtLoadedVariableRegistry::LoadedVariable> {
      TF_ASSIGN_OR_RETURN(
          xla::ifrt::DeviceListRef device_list,
          xla::ifrt::test_util::GetDevices(client_.get(), device_ids));
      xla::ifrt::ShardingRef sharding = xla::ifrt::HloSharding::Create(
          device_list, xla::ifrt::MemoryKind(), hlo_sharding);
      TF_ASSIGN_OR_RETURN(xla::ifrt::ArrayRef array,
                          xla_serving::sharding_internal::MakeArrayFromHostBuffer(
                              *client_, host_buffer, device_ids,
                              std::move(sharding), *thread_pool_));
      auto [promise, future] = tsl::MakePromise<xla::ifrt::ArrayRef>();
      promise.Set(std::move(array));
      return IfrtLoadedVariableRegistry::LoadedVariable(
          {.value = Value(std::move(future))});
    };

    loaded_variables_.push_back(std::string(variable_name));

    return ifrt_loaded_variable_registry_.TryRegisterLoadedVariable(
        {.device_ids = std::vector<int>(device_ids.begin(), device_ids.end()),
         .input_name = std::string(variable_name),
         .hlo_sharding = hlo_sharding},
        loaded_variable_producer);
  }

  IfrtLoadedVariableRegistry* GetLoadedVariableRegistry() {
    return &ifrt_loaded_variable_registry_;
  }

  std::shared_ptr<xla::ifrt::Client> GetIfrtClient() { return client_; }

 private:
  static constexpr int kThreadPoolNumThreads = 16;

  std::shared_ptr<xla::ifrt::Client> client_;
  std::unique_ptr<tsl::thread::ThreadPool> thread_pool_;
  IfrtLoadedVariableRegistry ifrt_loaded_variable_registry_;
  std::vector<std::string> loaded_variables_;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_KERNEL_TEST_UTILS_H_

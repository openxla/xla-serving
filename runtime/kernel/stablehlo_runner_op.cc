#include "runtime/kernel/stablehlo_runner_op.h"  // from @neptune

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "runtime/interpreter/manifest_utils.h"  // from @neptune
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/memory/memory.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/flat_signature.h"  // from @neptune
#include "runtime/core/host_buffer.h"  // from @neptune
#include "runtime/core/ifrt_loaded_variable_registry.h"  // from @neptune
#include "runtime/core/stablehlo.h"  // from @neptune
#include "runtime/core/stablehlo_optimizer.h"  // from @neptune
#include "runtime/core/stablehlo_runner.h"  // from @neptune
#include "runtime/core/value_utils.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "runtime/kernel/kernel_builder_constants.h"  // from @neptune
#include "runtime/kernel/kernel_builder_context.h"  // from @neptune
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @neptune
#include "google/protobuf/message.h"
#include "xla/pjrt/proto/compile_options.pb.h"  // from @xla
#include "xla/python/ifrt/array.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/device.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "xla/xla_data.pb.h"  // from @xla
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {

absl::StatusOr<std::unique_ptr<Kernel>> StableHloRunnerOp::Create(
    const KernelBuilderContext& context) {
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<StableHlo> stablehlo,
      context.GetConfig<std::shared_ptr<StableHlo>>(kStableHlo));
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<xla::ifrt::Client> ifrt_client,
      context.GetConfig<std::shared_ptr<xla::ifrt::Client>>(kIfrtClient));
  TF_ASSIGN_OR_RETURN(
      IfrtPersistentCompilationCache * persistent_compilation_cache,
      context.GetConfig<IfrtPersistentCompilationCache*>(
          kPersistentCompilationCache));
  TF_ASSIGN_OR_RETURN(
      tsl::thread::ThreadPool * thread_pool,
      context.GetConfig<tsl::thread::ThreadPool*>(kIfrtWorkerThreadPool));
  TF_ASSIGN_OR_RETURN(
      IfrtLoadedVariableRegistry * ifrt_loaded_variable_registry,
      context.GetConfig<IfrtLoadedVariableRegistry*>(
          kIfrtLoadedVariableRegistry));
  TF_ASSIGN_OR_RETURN(
      tsl::protobuf::Message * compilation_environment_proto,
      context.GetConfig<tsl::protobuf::Message*>(kCompilationEnvProto));
  TF_ASSIGN_OR_RETURN(std::vector<std::string> weights,
                      context.GetConfig<std::vector<std::string>>(kWeights));
  absl::StatusOr<xla::CompileOptionsProto> compile_options =
      context.GetConfig<xla::CompileOptionsProto>(kXlaCompileOptions);

  // If the compile options is not found, we will use the default compile
  // options.
  if (!compile_options.ok() && !absl::IsNotFound(compile_options.status())) {
    return compile_options.status();
  }

  absl::StatusOr<StablehloOptimizer*> optimizer =
      context.GetConfig<StablehloOptimizer*>(kStablehloOptimizer);
  // If StablehloOptimizer is not set in the kernel builder context, we will
  // not use the optimizer.
  if (absl::IsNotFound(optimizer.status())) {
    optimizer = nullptr;
  }

  TF_ASSIGN_OR_RETURN(std::shared_ptr<const AssignedDevices> assigned_devices,
                      context.GetConfig<std::shared_ptr<const AssignedDevices>>(
                          kAssignedDevices));
  absl::StatusOr<FlatSignature> input_sig_status =
      context.GetConfig<FlatSignature>(kFunctionFlatInputSignature);
  std::unique_ptr<FlatSignature> input_sig;
  if (input_sig_status.ok()) {
    input_sig = std::make_unique<FlatSignature>(*input_sig_status);
  } else if (!absl::IsNotFound(input_sig_status.status())) {
    return input_sig_status.status();
  }
  absl::StatusOr<FlatSignature> output_sig_status =
      context.GetConfig<FlatSignature>(kFunctionFlatOutputSignature);
  std::unique_ptr<FlatSignature> output_sig;
  if (output_sig_status.ok()) {
    output_sig = std::make_unique<FlatSignature>(*output_sig_status);
  } else if (!absl::IsNotFound(output_sig_status.status())) {
    return output_sig_status.status();
  }

  TF_ASSIGN_OR_RETURN(
      std::unique_ptr<StableHloRunner> stablehlo_runner,
      StableHloRunner::Create(
          stablehlo,
          compile_options.ok() ? std::make_optional(*compile_options)
                               : std::nullopt,
          ifrt_client, persistent_compilation_cache,
          ifrt_loaded_variable_registry, weights, compilation_environment_proto,
          thread_pool, *optimizer, assigned_devices,
          /*serving_core_selector=*/nullptr));

  return absl::WrapUnique(new StableHloRunnerOp(std::move(stablehlo_runner),
                                                std::move(input_sig),
                                                std::move(output_sig)));
  ;
}

std::string StableHloRunnerOp::Name() const { return "StableHloRunnerOp"; }

absl::Status StableHloRunnerOp::RunInternal(ExecuteContext& context) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "StableHloRunnerOp::Run",
        {{"model_id",
          absl::StrCat(context.metadata.model_name, ":",
                       context.metadata.model_version)}});
  });
  TF_ASSIGN_OR_RETURN(
      std::vector<HostBuffer> inputs,
      internal::UnpackFromValues<HostBuffer>(std::move(context.inputs)));

  TF_ASSIGN_OR_RETURN(std::vector<HostBuffer> outputs,
                      stablehlo_runner_->Execute(inputs));
  context.outputs = internal::PackAsValues(std::move(outputs));
  return absl::OkStatus();
}

}  // namespace neptune

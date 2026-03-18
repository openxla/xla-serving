#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace jsv {

inline constexpr absl::string_view kStableHlo = "stablehlo";
// Some kernels's implementation can take different numbers of inputs. This
// constant is used to specify the number of inputs.
inline constexpr absl::string_view kNumKernelInputs = "num_kernel_inputs";
inline constexpr absl::string_view kIfrtClient = "ifrt_client";
inline constexpr absl::string_view kIfrtWorkerThreadPool = "thread_pool";
inline constexpr absl::string_view kIfrtLoadedVariableRegistry =
    "ifrt_loaded_variable_registry";
inline constexpr absl::string_view kCompilationEnvProto =
    "compilation_env_proto";
inline constexpr absl::string_view kPersistentCompilationCache =
    "persistent_compilation_cache";
inline constexpr absl::string_view kWeights = "weights_names";
inline constexpr absl::string_view kStablehloOptimizer = "stablehlo_optimizer";
inline constexpr absl::string_view kXlaCompileOptions = "xla_compile_options";
inline constexpr absl::string_view kAssignedDevices = "assigned_devices";
inline constexpr absl::string_view kFunctionFlatInputSignature =
    "function_flat_input_signature";
inline constexpr absl::string_view kFunctionFlatOutputSignature =
    "function_flat_output_signature";

}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

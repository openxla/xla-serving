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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace neptune {

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
inline constexpr absl::string_view kServingCoreSelector =
    "serving_core_selector";
inline constexpr absl::string_view kStableCallbackId = "stable_callback_id";
}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

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

#ifndef XLA_SERVING_RUNTIME_TENSORFLOW_KERNEL_KERNEL_BUILDER_CONSTANTS_H_
#define XLA_SERVING_RUNTIME_TENSORFLOW_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

#include "absl/strings/string_view.h"  // from @abseil-cpp

namespace xla_serving {

inline constexpr absl::string_view kTfSavedModelRunner =
    "tf_saved_model_runner";
inline constexpr absl::string_view kTfFunctionSignatureName =
    "tf_function_signature_name";
inline constexpr absl::string_view kTfFunctionOrderedInputNames =
    "tf_function_input_names";
inline constexpr absl::string_view kTfFunctionOrderedOutputNames =
    "tf_function_output_names";
inline constexpr absl::string_view kTfFunctionFlatInputSignature =
    "tf_function_flat_input_signature";
inline constexpr absl::string_view kTfFunctionFlatOutputSignature =
    "tf_function_flat_output_signature";
inline constexpr absl::string_view kTfFunctionPreDataConversion =
    "pre_data_conversion";
inline constexpr absl::string_view kTfFunctionPostDataConversion =
    "post_data_conversion";

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_TENSORFLOW_KERNEL_KERNEL_BUILDER_CONSTANTS_H_

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

#ifndef XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_INTERPRETER_CREATE_OPTIONS_H_
#define XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_INTERPRETER_CREATE_OPTIONS_H_

#include <optional>

#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "tensorflow/core/tfrt/saved_model/saved_model.h"  // from @org_tensorflow

namespace xla_serving {

// The options to load the TF SavedModel.
struct TfrtOptions {
  // The options to load the TF SavedModel. The `tensorflow::tfrt_stub::Runtime`
  // used to initialized this object needs to be alive as long as the TF
  // SavedModel is alive.
  std::optional<tensorflow::tfrt_stub::SavedModel::Options> saved_model_options;
};

// The options to load the Orbax model with TensorFlow preprocessing and
// postprocessing functions.
struct TfCompatibleInterpreterCreateOptions : public InterpreterCreateOptions {
  TfrtOptions tfrt_options;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_INTERPRETER_CREATE_OPTIONS_H_

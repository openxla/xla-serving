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

#include <functional>
#include <memory>
#include <optional>

#include "runtime/interpreter/interpreter_create_options.h"  // from @xla_serving
#include "tensorflow/core/tfrt/runtime/work_queue_interface.h"  // from @org_tensorflow
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
  // Optional. Called once per TF function execution to create a per-request
  // work queue for inter/intra-op work. If empty, or if it returns nullptr, the
  // runtime's default request queue is used. Anything captured by the callable
  // must outlive the interpreter.
  std::function<std::unique_ptr<tensorflow::tfrt_stub::WorkQueueInterface>()>
      work_queue_factory;
};

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_TENSORFLOW_TF_COMPATIBLE_INTERPRETER_CREATE_OPTIONS_H_

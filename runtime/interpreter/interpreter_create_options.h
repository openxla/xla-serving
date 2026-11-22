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

#ifndef XLA_SERVING_RUNTIME_INTERPRETER_INTERPRETER_CREATE_OPTIONS_H_
#define XLA_SERVING_RUNTIME_INTERPRETER_INTERPRETER_CREATE_OPTIONS_H_

#include <cstdint>
#include <memory>
#include <string>

#include "absl/base/nullability.h"  // from @abseil-cpp
#include "protos/manifest.pb.h"  // from @xla_serving
#include "runtime/core/allocator.h"  // from @xla_serving
#include "runtime/core/default_allocator.h"  // from @xla_serving
#include "runtime/core/stablehlo_optimizer.h"  // from @xla_serving
#include "runtime/libs/ifrt/persistent_compilation_cache/ifrt_persistent_compilation_cache.h"  // from @xla_serving
#include "runtime/libs/ifrt/serving_core_selector/serving_core_selector.h"  // from @xla_serving
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/protobuf.h"  // from @tsl

namespace xla_serving {

// The options to create the interpreter. This is a virtual class to allow
// subclasses to add more options. Therefore, one should always use the pointer
// or reference instead of the object itself.
struct InterpreterCreateOptions {
  virtual ~InterpreterCreateOptions() = default;

  // Optional: The name of the model.
  std::string model_name;
  // Optional: The version of the model.
  int64_t model_version = 0;
  // The path to the model. It should be the parent directory of the model's
  // manifest proto.
  std::string model_path;
  // The inter-op thread pool to use for executing the model. It can't be null.
  // Not owned.
  tsl::thread::ThreadPool* execute_worker_thread_pool = nullptr;
  // The IFRT client to use for executing the model.
  std::shared_ptr<xla::ifrt::Client> ifrt_client = nullptr;
  // The IFRT client to use for host execution.
  std::shared_ptr<xla::ifrt::Client> host_ifrt_client = nullptr;
  // The intra-op thread pool to use for executing the model. It can't be null.
  // Not owned.
  tsl::thread::ThreadPool* computation_thread_pool = nullptr;
  // The compilation environment proto to use for `xla::ExecutableBuildOptions`.
  tsl::protobuf::Message* absl_nullable compilation_env = nullptr;
  int min_num_batch_threads = 1;
  int min_max_enqueued_batches = 1;
  std::unique_ptr<StablehloOptimizer> optimizer = nullptr;
  // The default allocator is DefaultAllocator. It can be overridden to a
  // custom allocator which extends Allocator.
  Allocator* absl_nonnull allocator = DefaultAllocator::GetInstance();
  // The platform name to use for `xla::ExecutableBuildOptions`. If not set,
  // the default platform name will be "TPU".
  // platform_name should be the name of xla_saved_model_manifest::Platform name.
  std::string platform_name = xla_saved_model_manifest::Platform_Name(
      xla_saved_model_manifest::Platform::TPU);
  // If true, the interpreter will only load the pipeline when it is requested.
  bool lazy_loading_pipelines = false;
  // If true, the device assignment in the manifest will be ignored.
  bool ignore_device_assignment_in_manifest = false;
  // If true, in-graph batching is enabled. If false, in-graph batching is
  // disabled, but callers can still do out-of-graph batching.
  bool enable_batching = true;
  // If true, every batched pipeline uses priority-aware batching, overriding
  // the `priority_batching_policy` in the model's exported `BatchOptions`. If
  // false, the exported policy is used. Priority-aware batching currently also
  // requires `--jsv_use_queue_base`; otherwise the legacy queue is used and the
  // policy is ignored.
  bool enable_priority_aware_batching = false;
  // Providing a thread pool enables loading checkpoints in the background.
  // A null pointer disables background checkpoint loading. Not owned.
  tsl::thread::ThreadPool* absl_nullable checkpoint_loader_thread_pool =
      nullptr;
  // Persistent compilation cache for single-host serving. If nullptr, the
  // compilation cache is disabled.
  std::unique_ptr<IfrtPersistentCompilationCache>
      persistent_compilation_cache = nullptr;
  // Core selector for balancing serving load for IFRT portable executables. If
  // nullptr, LB-policy-based core selection is disabled.
  ServingCoreSelector* absl_nullable serving_core_selector = nullptr;

  // AOT Flag. If true, allows falling back to JIT compilation if the AOT
  // artifact is missing or the hardware topology mismatches. By default,
  // this is false, meaning models exported with AOT payloads will fail
  // loudly on a mismatch. (Note: Legacy models exported without AOT payloads
  // will naturally bypass AOT loading and still JIT compile).
  bool allow_jit_fallback_for_aot = false;

  // AOT Flag. Coarse routing target topology label required for matching
  // against compiled_exec.target_topology_label() in the AOT manifest.
  std::string target_topology_label;
};
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_INTERPRETER_INTERPRETER_CREATE_OPTIONS_H_

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

#include "runtime/interpreter/linear_graph_execute.h"  // from @neptune

#include <cstdint>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "runtime/interpreter/context.h"  // from @neptune
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/log/check.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/errors.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/profiler/lib/connected_traceme.h"  // from @tsl
#include "tsl/profiler/lib/context_types.h"  // from @tsl
#include "tsl/profiler/lib/traceme.h"  // from @tsl

namespace neptune {

namespace {

// AsyncState holds state for the async kernel execution.
struct AsyncState {
  tsl::Future<std::vector<Value>>
      inputs_futures;          // Input for the current kernel.
  tsl::Future<> chain_future;  // Keep alive the returned status future
                               // from the current async kernel.It becomes
                               // ready when the current kernel finishes.
                               // Not available for sync kernels.
  Kernel::ExecuteContext execute_context;
};

// A helper function to launch the kernel asynchronously.
// For sync kernels, we schedule the kernel execution on the
// threadpool to emulate the async behavior.
// The caller keeps the async_state alive until the returned future is ready.
tsl::Future<std::vector<Value>> AsyncLaunchKernel(
    Kernel& kernel, const RunContext run_context, AsyncState& async_state,
    tsl::thread::ThreadPool* thread_pool, absl::string_view model_name,
    int64_t model_version) {
  auto [next_promise, next_future] = tsl::MakePromise<std::vector<Value>>();

  // Currently we run the eigen computation and kernel execution on the same
  // threadpool.
  // We also always dispatch the next kernel execution to the threadpool. A
  // potential optimization is to dispatch the next kernel execution to the
  // threadpool only if the previous kernel execution is not finished or this
  // is the first kernel.
  async_state.inputs_futures.OnReady(
      [thread_pool = thread_pool, model_name = model_name,
       model_version = model_version, next_promise = std::move(next_promise),
       run_context = run_context, &async_state,
       &kernel](absl::StatusOr<std::vector<Value>> ready_inputs) mutable {
        if (!ready_inputs.ok()) {
          std::move(next_promise).Set(ready_inputs.status());
          return;
        }
        if (kernel.CanRunAsync()) {
          // For async kernels, we run the kernel asynchronously.
          std::vector<Value> ready_values = *std::move(ready_inputs);
          async_state.execute_context.inputs = absl::MakeSpan(ready_values);
          async_state.execute_context.metadata.model_name = model_name;
          async_state.execute_context.metadata.model_version = model_version;
          async_state.execute_context.run_context = run_context;
          CancellableAsyncKernel* cancellable_kernel =
              dynamic_cast<CancellableAsyncKernel*>(&kernel);
          if (cancellable_kernel == nullptr) {
            std::move(next_promise)
                .Set(absl::InternalError(
                    "Kernel claims to be async but does not inherit from "
                    "CancellableAsyncKernel."));
            return;
          }
          async_state.chain_future =
              cancellable_kernel->RunAsync(async_state.execute_context);

          async_state.chain_future.OnReady(
              [next_promise = std::move(next_promise),
               ready_values = std::move(ready_values),
               &async_state](absl::Status status) mutable {
                if (!status.ok()) {
                  std::move(next_promise).Set(status);
                  return;
                }
                std::move(next_promise)
                    .Set(std::move(async_state.execute_context.outputs));
              });
        } else {
          // TODO: Let kernel returns name and put it in the
          // trace name.
          // TODO: Wrap thread pool and do the producer/consumer
          // automatically.
          tsl::profiler::TraceMeProducer trace_me(
              [&]() {
                return tsl::profiler::TraceMeEncode(
                    "AsyncLaunchSyncKernel",
                    {{"model_id",
                      absl::StrCat(model_name, ":", model_version)}});
              },
              tsl::profiler::ContextType::kJaxServingExecutor);
          async_state.execute_context.metadata.trace_id =
              trace_me.GetContextId();
          // For sync kernels, we schedule the kernel execution on the
          // threadpool to emulate the async behavior.
          thread_pool->Schedule(
              [ready_inputs = *std::move(ready_inputs),
               next_promise = std::move(next_promise).ToShared(),
               model_name = model_name, model_version = model_version,
               run_context = run_context, &kernel, &async_state]() mutable {
                tsl::profiler::TraceMeConsumer trace_me(
                    [&]() {
                      return tsl::profiler::TraceMeEncode(
                          "AsyncLaunchSyncKernel",
                          {{"model_id",
                            absl::StrCat(model_name, ":", model_version)}});
                    },
                    tsl::profiler::ContextType::kJaxServingExecutor,
                    async_state.execute_context.metadata.trace_id);
                async_state.execute_context.metadata.model_name = model_name;
                async_state.execute_context.metadata.model_version =
                    model_version;
                async_state.execute_context.inputs =
                    absl::MakeSpan(ready_inputs);
                async_state.execute_context.run_context = run_context;
                if (auto status = kernel.Run(async_state.execute_context);
                    !status.ok()) {
                  std::move(next_promise)->Set(status);
                } else {
                  std::move(next_promise)
                      ->Set(std::move(async_state.execute_context.outputs));
                }
              });
        }
      });
  return next_future;
}

}  // namespace

absl::StatusOr<std::vector<Value>> ExecuteLinearGraph(
    const RunContext run_context, absl::Span<const Value> inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    absl::string_view model_name, int64_t model_version) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "ExecuteLinearGraph",
        {{"model_id", absl::StrCat(model_name, ":", model_version)}});
  });
  // TODO: type check inputs

  Kernel::ExecuteContext execute_context;
  execute_context.metadata.model_name = model_name;
  execute_context.metadata.model_version = model_version;
  execute_context.inputs = inputs;
  execute_context.run_context = run_context;
  std::vector<Value> outputs;
  for (auto& kernel : graph) {
    tsl::profiler::TraceMe trace_me([&]() {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("ExecuteLinearGraphKernel-", kernel->Name()),
          {{"model_id", absl::StrCat(model_name, ":", model_version)}});
    });
    TF_RETURN_IF_ERROR(kernel->Run(execute_context));

    // Hold the output of the previous kernel. This will also clear the outputs
    // of the current kernel.
    outputs = std::move(execute_context.outputs);

    execute_context.inputs = absl::MakeSpan(outputs);
  }
  // TODO:  type check outputs
  return outputs;
}

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
ExecuteLinearGraphWithKeywordArgs(
    const RunContext run_context,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    absl::string_view model_name, int64_t model_version) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "ExecuteLinearGraphWithKeywordArgs",
        {{"model_id", absl::StrCat(model_name, ":", model_version)}});
  });

  Kernel::ExecuteContext execute_context;
  execute_context.metadata.model_name = model_name;
  execute_context.metadata.model_version = model_version;
  execute_context.named_values = keyword_inputs;
  execute_context.run_context = run_context;
  std::vector<Value> outputs;
  for (auto& kernel : graph) {
    tsl::profiler::TraceMe trace_me([&]() {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("ExecuteLinearGraphKernel-", kernel->Name()),
          {{"model_id", absl::StrCat(model_name, ":", model_version)}});
    });
    TF_RETURN_IF_ERROR(kernel->Run(execute_context));
  }
  return execute_context.named_values;
};

tsl::Future<std::vector<Value>> AsyncExecuteLinearGraph(
    const RunContext run_context, absl::Span<const Value> inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    tsl::thread::ThreadPool* thread_pool, absl::string_view model_name,
    int64_t model_version) {
  tsl::profiler::TraceMe trace_me([&]() {
    return tsl::profiler::TraceMeEncode(
        "AsyncExecuteLinearGraph",
        {{"model_id", absl::StrCat(model_name, ":", model_version)}});
  });
  CHECK(thread_pool != nullptr);
  // We use a bootstrap promise/future to kick start the async computation.
  // Kernel N-1 produces async_results as input to Kernel N.
  auto [bootstrap_promise, async_results] =
      tsl::MakePromise<std::vector<Value>>();

  // Assume the inputs are either reference counted or simple copyable such that
  // making copies of the inputs can keep inputs alive.
  std::vector<Value> inputs_copy(inputs.begin(), inputs.end());
  std::move(bootstrap_promise).Set(inputs_copy);

  std::vector<AsyncState> async_states;
  async_states.reserve(graph.size());

  for (auto& kernel : graph) {
    tsl::profiler::TraceMe trace_me([&]() {
      return tsl::profiler::TraceMeEncode(
          absl::StrCat("AsyncExecuteLinearGraphKernel-", kernel->Name()),
          {{"model_id", absl::StrCat(model_name, ":", model_version)}});
    });
    async_states.push_back(
        AsyncState{.inputs_futures = std::move(async_results)});
    AsyncState& async_state = async_states.back();
    async_results = AsyncLaunchKernel(*kernel, run_context, async_state,
                                      thread_pool, model_name, model_version);
  }

  async_results.OnReady([async_states = std::move(async_states)](
                            absl::StatusOr<std::vector<Value>> ready_outputs) {
    // keep async states alive until the final output is ready.
  });
  return async_results;
}

}  // namespace neptune

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

#ifndef XLA_SERVING_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_
#define XLA_SERVING_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "runtime/interpreter/context.h"  // from @neptune
#include "absl/container/flat_hash_map.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/types/span.h"  // from @abseil-cpp
#include "runtime/core/value.h"  // from @neptune
#include "runtime/kernel/kernel.h"  // from @neptune
#include "xla/tsl/concurrency/future.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla

namespace xla_serving {

// In this file, we use the term "linear graph" to refer to a graph of kernels
// that are executed sequentially, aka, each node corresponds to one jsv kernel.
// Each kernel executes a function such as a StableHlo computation or a TF
// function.

// Executes the linear graph with the given inputs synchronously.
// This may block the current thread.
absl::StatusOr<std::vector<Value>> ExecuteLinearGraph(
    RunContext run_context, absl::Span<const Value> inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    absl::string_view model_name, int64_t model_version);

absl::StatusOr<absl::flat_hash_map<std::string, Value>>
ExecuteLinearGraphWithKeywordArgs(
    RunContext run_context,
    const absl::flat_hash_map<std::string, Value>& keyword_inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    absl::string_view model_name, int64_t model_version);

// Executes the linear graph with the given inputs asynchronously.
// TODO: avoid blocking the current thread if a kernel can run
// asynchronously.
tsl::Future<std::vector<Value>> AsyncExecuteLinearGraph(
    RunContext run_context, absl::Span<const Value> inputs,
    absl::Span<const std::unique_ptr<Kernel>> graph,
    tsl::thread::ThreadPool* thread_pool, absl::string_view model_name,
    int64_t model_version);
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_

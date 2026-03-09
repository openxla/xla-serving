#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "third_party/neptune/runtime/interpreter/context.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/absl/types/span.h"
#include "third_party/neptune/runtime/core/value.h"
#include "third_party/neptune/runtime/kernel/kernel.h"
#include "third_party/tensorflow/compiler/xla/tsl/concurrency/future.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/threadpool.h"

namespace jsv {

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

absl::StatusOr<std::vector<Value>> ExecuteLinearGraphWithKeywordArgs(
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
}  // namespace jsv

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_INTERPRETER_LINEAR_GRAPH_EXECUTE_H_

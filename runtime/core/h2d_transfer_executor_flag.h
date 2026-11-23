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

#ifndef XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_FLAG_H_
#define XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_FLAG_H_

#include <memory>
#include <string>

#include "absl/flags/declare.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving

// Selects the `H2DTransferExecutor` that the JIT runner of a model transfers
// its host inputs with:
//
// - "" (the default): no executor. The runner uses the default IFRT transfer.
// - "arena": `ArenaH2DTransferExecutor`. It does not run on CPU, so a model
//   fails to load there.
// - "batched": `BatchedH2DTransferExecutor`.
//
// With any other value a model fails to load. The flag is read when a model is
// loaded.
//
// TODO: Replace this flag with the one that enables the arena
// transfer.
ABSL_DECLARE_FLAG(std::string, xla_serving_h2d_transfer_executor);

namespace xla_serving {

// Returns the factory of the executor that
// `--xla_serving_h2d_transfer_executor` selects, or null if it selects none.
// Returns InvalidArgument if the flag does not name an executor.
absl::StatusOr<std::unique_ptr<H2DTransferExecutorFactory>>
CreateH2DTransferExecutorFactoryFromFlag();

}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_CORE_H2D_TRANSFER_EXECUTOR_FLAG_H_

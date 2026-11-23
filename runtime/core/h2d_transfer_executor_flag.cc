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

#include "runtime/core/h2d_transfer_executor_flag.h"  // from @xla_serving

#include <memory>
#include <string>

#include "absl/flags/flag.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "runtime/core/arena_h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/batched_h2d_transfer_executor.h"  // from @xla_serving
#include "runtime/core/h2d_transfer_executor.h"  // from @xla_serving

ABSL_FLAG(std::string, xla_serving_h2d_transfer_executor, "",
          "The H2D transfer executor that JIT-compiled models transfer their "
          "host inputs with: \"arena\" for ArenaH2DTransferExecutor, which "
          "does not run on CPU, or \"batched\" for "
          "BatchedH2DTransferExecutor. If empty, they use the default IFRT "
          "transfer. The flag is read when a model is loaded.");

namespace xla_serving {

absl::StatusOr<std::unique_ptr<H2DTransferExecutorFactory>>
CreateH2DTransferExecutorFactoryFromFlag() {
  const std::string executor =
      absl::GetFlag(FLAGS_xla_serving_h2d_transfer_executor);
  if (executor.empty()) {
    return nullptr;
  }
  if (executor == "arena") {
    return std::make_unique<ArenaH2DTransferExecutorFactory>();
  }
  if (executor == "batched") {
    return std::make_unique<BatchedH2DTransferExecutorFactory>();
  }
  return absl::InvalidArgumentError(
      absl::StrCat("--xla_serving_h2d_transfer_executor must be \"arena\", "
                   "\"batched\" or empty, but is \"",
                   executor, "\"."));
}

}  // namespace xla_serving

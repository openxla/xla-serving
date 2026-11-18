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

#ifndef XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_FLAG_H_
#define XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_FLAG_H_

#include "absl/flags/declare.h"  // from @abseil-cpp

// Enables dynamic DMA: an input whose dimension 0 the model signature marks as
// an upper bound is served with only the rows that the requests have. The
// batching kernel does not pad it, and the runner allocates the device buffer
// for the compiled shape but copies only those rows into it.
//
// The flag only turns the padding off, and only for a model whose runner takes
// the input with fewer rows: a JIT runner with an `H2DTransferExecutor`
// (`--xla_serving_h2d_transfer_executor`) that can copy fewer rows on its PjRt
// client. Otherwise the input is padded on the host to the compiled shape,
// like any other input: while the flag is false, without an executor, with the
// AOT runner, and with `BatchedH2DTransferExecutor` on a PjRt client that is
// not derived from `xla::CommonPjRtClient`, such as `PjRtCApiClient`. The flag
// is read when a model is loaded.
//
// Once the batching kernel passes on fewer rows there is no fallback to
// padding: a batch with fewer rows fails if the executor cannot copy fewer
// rows of that input. `PlanDynamicDmaTransfer()` in dynamic_dma_utils.h lists
// what the input must satisfy. On TPU the input has to be flattened to rank 1,
// and its element type must be at most 32 bits wide.
ABSL_DECLARE_FLAG(bool, xla_serving_enable_dynamic_dma);

#endif  // XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_FLAG_H_

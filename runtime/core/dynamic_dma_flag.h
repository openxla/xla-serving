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
// batching kernel does not pad it, and the JIT runner allocates the device
// buffer for the compiled shape but copies only those rows into it.
//
// While the flag is false, such an input is padded on the host to the compiled
// shape, like any other input. The flag is read when a model is loaded.
//
// There is no fallback to padding while the flag is true: a batch with fewer
// rows fails if `BatchedH2DTransferExecutor` cannot copy fewer rows of the
// input. See its class comment for what the client and the input must satisfy.
// On TPU the input has to be flattened to rank 1, and its element type must be
// at most 32 bits wide.
ABSL_DECLARE_FLAG(bool, xla_serving_enable_dynamic_dma);

#endif  // XLA_SERVING_RUNTIME_CORE_DYNAMIC_DMA_FLAG_H_

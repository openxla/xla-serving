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

#include "runtime/core/dynamic_dma_flag.h"  // from @xla_serving

#include "absl/flags/flag.h"  // from @abseil-cpp

ABSL_FLAG(bool, xla_serving_enable_dynamic_dma, false,
          "If true, an input whose dimension 0 is marked as an upper bound in "
          "the model signature is served with only the rows that the requests "
          "have: the batching kernel does not pad it, and only those rows are "
          "copied to the device. JIT-compiled models then transfer all their "
          "host inputs with BatchedH2DTransferExecutor. The flag is read when "
          "a model is loaded.");

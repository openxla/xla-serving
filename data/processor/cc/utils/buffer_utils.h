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

#ifndef THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_UTILS_BUFFER_UTILS_H_
#define THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_UTILS_BUFFER_UTILS_H_

#include <cstdint>

#include "Eigen/Core"  // IWYU pragma: keep, b/376693552

// go/lsc-fix-eigen-includes
// IWYU pragma: no_include "Eigen/src/Core/util/ConfigureVectorization.h"

namespace neptune::data::composite::utils {

#if EIGEN_MAX_ALIGN_BYTES != 0
inline bool IsBufferEigenAligned(const void* data_ptr) {
  return reinterpret_cast<uintptr_t>(data_ptr) % EIGEN_MAX_ALIGN_BYTES == 0;
}
#else
inline bool IsBufferEigenAligned(const void* data_ptr) { return true; }
#endif

}  // namespace neptune::data::composite::utils

#endif  // THIRD_PARTY_NEPTUNE_DATA_PROCESSOR_CC_UTILS_BUFFER_UTILS_H_


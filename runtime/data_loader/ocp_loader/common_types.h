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

#ifndef THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_
#define THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_

namespace neptune {

enum class CheckpointType {
  // In standard checkpoint mode, each param is stored in its sub directory and
  // each partition in its own file..
  kStandard,
  // OCDBT mode is an optimization for lots of small params. The param names are
  // not human-readable in the checkpoint file directory.
  // https://google.github.io/tensorstore/kvstore/ocdbt/index.html
  kOcdbt,
};

}  // namespace neptune

#endif  // THIRD_PARTY_NEPTUNE_RUNTIME_DATA_LOADER_OCP_LOADER_COMMON_TYPES_H_

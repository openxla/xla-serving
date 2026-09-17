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

#ifndef XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_FLAGS_H_
#define XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_FLAGS_H_

#include <string>
#include "absl/flags/declare.h"  // from @abseil-cpp

ABSL_DECLARE_FLAG(std::string, orbax_grpc_server_platform);

#endif  // XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_FLAGS_H_

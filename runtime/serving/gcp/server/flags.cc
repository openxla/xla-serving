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

#include "runtime/serving/gcp/server/flags.h"  // from @xla_serving

#include <string>
#include "absl/flags/flag.h"  // from @abseil-cpp

ABSL_FLAG(std::string, orbax_grpc_server_platform, "cpu",
          "The platform to use. It can be cpu, tpu, or cuda.");

ABSL_FLAG(int, num_execution_streams, 1,
          "Number of IFRT execution streams to spread StableHLO executions "
          "across. Executions on different streams may run concurrently. "
          "On the cpu platform the default of 1 runs one batch at a time "
          "regardless of the number of batch threads; 4 is a reasonable "
          "starting point on a 16 vCPU host. Has little effect on gpu/tpu, "
          "where a device executes in stream order on hardware.");

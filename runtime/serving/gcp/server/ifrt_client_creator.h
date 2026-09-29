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

#ifndef XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_
#define XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_

#include <memory>
#include <optional>

#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/python/ifrt/client.h"  // from @xla

namespace Eigen {
class ThreadPoolInterface;
}  // namespace Eigen

namespace xla_serving {
namespace serving {

// Options for creating an IFRT client. Platform-specific settings are grouped
// into nested structs and are ignored on other platforms.
struct IfrtClientOptions {
  // Options for the CPU platform.
  struct CpuOptions {
    // If true, creates a NanoRt IFRT client (`xla::cpu::NanoIfrtClient`)
    // instead of a PjRt-IFRT CPU client.
    bool use_nanort = true;

    // Number of CPU devices. If unset, both the PjRt and NanoRt CPU clients
    // use `--xla_force_host_platform_device_count` (defaults to 1).
    std::optional<int> device_count;

    // NanoRt only. Intra-op thread pool used by the NanoRt client. It must
    // outlive the client and all pending executions. If null, a process-wide
    // default thread pool sized to the host's max parallelism is used.
    Eigen::ThreadPoolInterface* nanort_intra_op_threadpool = nullptr;
  };

  CpuOptions cpu;
};

// Creates an IFRT client for `pjrt_platform` (e.g. "cpu", "gpu", "tpu"). Only
// the platform compiled into the binary is supported.
absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetIfrtClient(
    absl::string_view pjrt_platform, const IfrtClientOptions& options = {});

}  // namespace serving
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_IFRT_CLIENT_CREATOR_H_

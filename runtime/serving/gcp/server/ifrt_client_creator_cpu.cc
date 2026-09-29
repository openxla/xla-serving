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

#include "runtime/serving/gcp/server/ifrt_client_creator.h"  // from @xla_serving

#include <memory>
#include <utility>

#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "xla/backends/cpu/nanort/ifrt_client.h"  // from @xla
#include "xla/debug_options_flags.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/env.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla
#include "xla/tsl/platform/threadpool.h"  // from @xla
#include "tsl/platform/cpu_info.h"  // from @tsl

namespace xla_serving {
namespace serving {

namespace {

// Returns a process-wide intra-op thread pool for the NanoRt client. It is
// never destroyed so that it outlives any client and pending execution.
tsl::thread::ThreadPool& GetDefaultNanoRtIntraOpThreadPool() {
  static auto* const thread_pool = new tsl::thread::ThreadPool(
      tsl::Env::Default(), tsl::ThreadOptions(), "NanoRtIntraOpThreadPool",
      tsl::port::MaxParallelism());
  return *thread_pool;
}

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreateNanoRtClient(
    const IfrtClientOptions::CpuOptions& options) {
  // Match the PjRt CPU client: fall back to
  // `--xla_force_host_platform_device_count` (defaults to 1).
  int num_devices = options.device_count.value_or(
      xla::GetDebugOptionsFromFlags().xla_force_host_platform_device_count());
  if (num_devices <= 0) {
    return absl::InvalidArgumentError(
        absl::StrCat("CPU device count must be positive, got ", num_devices));
  }
  xla::cpu::NanoIfrtOptions nano_options{
      .intra_op_threadpool = options.nanort_intra_op_threadpool,
  };
  if (nano_options.intra_op_threadpool == nullptr) {
    nano_options.intra_op_threadpool =
        GetDefaultNanoRtIntraOpThreadPool().AsEigenThreadPool();
  }
  return xla::cpu::NanoIfrtClient::CreateWithDevices(num_devices,
                                                     nano_options);
}

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreatePjRtCpuClient(
    const IfrtClientOptions::CpuOptions& options) {
  xla::CpuClientOptions cpu_options;
  cpu_options.cpu_device_count = options.device_count;
  TF_ASSIGN_OR_RETURN(auto pjrt_client, xla::GetXlaPjrtCpuClient(cpu_options));
  return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
}

}  // namespace

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> GetIfrtClient(
    absl::string_view pjrt_platform, const IfrtClientOptions& options) {
  if (pjrt_platform == "cpu") {
    if (options.cpu.use_nanort) {
      return CreateNanoRtClient(options.cpu);
    }
    return CreatePjRtCpuClient(options.cpu);
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported or uncompiled PJRT platform: ", pjrt_platform));
}

}  // namespace serving
}  // namespace xla_serving

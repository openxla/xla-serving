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

#include "runtime/serving/gcp/server/tpu_env_init.h"  // from @xla_serving

#include <arpa/inet.h>
#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/cleanup/cleanup.h"  // from @abseil-cpp
#include "absl/functional/function_ref.h"  // from @abseil-cpp
#include "absl/log/log.h"  // from @abseil-cpp
#include "absl/status/status.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/ascii.h"  // from @abseil-cpp
#include "absl/strings/match.h"  // from @abseil-cpp
#include "absl/strings/numbers.h"  // from @abseil-cpp
#include "absl/strings/str_cat.h"  // from @abseil-cpp
#include "absl/strings/str_join.h"  // from @abseil-cpp
#include "absl/strings/str_split.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/clock.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp
#include "xla/tsl/platform/env.h"  // from @xla
#include "tsl/platform/path.h"  // from @tsl

extern char** environ;

namespace xla_serving {
namespace serving {
namespace {

// PCI vendor id of Google devices.
constexpr absl::string_view kGooglePciVendorId = "0x1ae0";

// PCI device id -> TPU generation, as in JAX's `hardware_utils.py`.
constexpr std::pair<absl::string_view, absl::string_view> kTpuPciDeviceIds[] = {
    {"0x0027", "v3"},    {"0x0056", "plc"},   {"0x005e", "v4"},
    {"0x0062", "v5p"},   {"0x0063", "v5e"},   {"0x006f", "v6e"},
    {"0x0076", "tpu7x"}, {"0x0083", "tpu8i"}, {"0x007c", "tpu8t"},
};

// Variables that together describe the slice topology and this host's place
// in it. If the user set any of them we do not touch the others.
constexpr absl::string_view kTopologyEnvVars[] = {
    "TPU_ACCELERATOR_TYPE", "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_HOST_BOUNDS",
    "TPU_WORKER_ID",        "TPU_WORKER_HOSTNAMES",
};

// Returns the target of the symlink at `path`, or "" if it is not a symlink.
std::string ReadLink(const std::string& path) {
  char buf[PATH_MAX];
  ssize_t n = ::readlink(path.c_str(), buf, sizeof(buf) - 1);
  if (n < 0) return "";
  return std::string(buf, n);
}

// Waits until `fd` is ready for `events` or `deadline` passes.
bool PollUntil(int fd, short events, absl::Time deadline) {  // NOLINT
  while (true) {
    absl::Duration remaining = deadline - absl::Now();
    if (remaining <= absl::ZeroDuration()) return false;
    pollfd pfd{fd, events, 0};
    int rc = ::poll(&pfd, 1,
                    static_cast<int>(absl::ToInt64Milliseconds(
                        std::min(remaining, absl::Seconds(1)))));
    if (rc > 0) return (pfd.revents & (events | POLLERR | POLLHUP)) != 0;
    if (rc < 0 && errno != EINTR) return false;
  }
}

std::optional<std::string> RealGetEnv(absl::string_view name) {
  std::string name_str(name);
  const char* value = std::getenv(name_str.c_str());
  if (value == nullptr) return std::nullopt;
  return std::string(value);
}

bool RealFileExists(absl::string_view path) {
  return tsl::Env::Default()->FileExists(std::string(path)).ok();
}

// One-line dump of every `TPU_*` / `LIBTPU_*` variable in the environment.
std::string DescribeTpuEnv() {
  std::vector<std::string> entries;
  for (char** e = environ; e != nullptr && *e != nullptr; ++e) {
    absl::string_view entry(*e);
    if (absl::StartsWith(entry, "TPU_") || absl::StartsWith(entry, "LIBTPU_")) {
      entries.emplace_back(entry);
    }
  }
  std::sort(entries.begin(), entries.end());
  return entries.empty() ? "<none>" : absl::StrJoin(entries, " ");
}

}  // namespace

TpuHostInfo ScanTpuHost(absl::string_view sysfs_pci_root,
                        FileExistsFn device_node_exists) {
  TpuHostInfo info;
  tsl::Env* env = tsl::Env::Default();
  const std::string root(sysfs_pci_root);
  std::vector<std::string> children;
  if (absl::Status status = env->GetChildren(root, &children); !status.ok()) {
    VLOG(1) << "Cannot list PCI devices under " << root << ": " << status;
    return info;
  }
  std::sort(children.begin(), children.end());
  for (const std::string& child : children) {
    // Like JAX, only count PCI function 0 of each device.
    if (!absl::EndsWith(child, ".0")) continue;
    const std::string dev_dir = tsl::io::JoinPath(root, child);
    std::string vendor;
    if (!tsl::ReadFileToString(env, tsl::io::JoinPath(dev_dir, "vendor"),
                               &vendor)
             .ok()) {
      continue;
    }
    if (absl::StripAsciiWhitespace(vendor) != kGooglePciVendorId) continue;
    std::string device;
    if (!tsl::ReadFileToString(env, tsl::io::JoinPath(dev_dir, "device"),
                               &device)
             .ok()) {
      continue;
    }
    const absl::string_view device_id = absl::StripAsciiWhitespace(device);
    absl::string_view generation;
    for (const auto& [id, gen] : kTpuPciDeviceIds) {
      if (id == device_id) {
        generation = gen;
        break;
      }
    }
    if (generation.empty()) {
      LOG(WARNING) << "Ignoring Google PCI device " << child
                   << " with unknown device id " << device_id;
      continue;
    }

    const std::string driver = std::string(
        tsl::io::Basename(ReadLink(tsl::io::JoinPath(dev_dir, "driver"))));
    if (driver == "vfio-pci") {
      const std::string group = std::string(tsl::io::Basename(
          ReadLink(tsl::io::JoinPath(dev_dir, "iommu_group"))));
      const std::string device_path = absl::StrCat("/dev/vfio/", group);
      if (group.empty() || !device_node_exists(device_path)) {
        LOG(INFO) << "Ignoring TPU " << generation << " chip " << child
                  << " bound to vfio-pci: its device node " << device_path
                  << " is not accessible from this process (not passed to "
                     "the container?).";
        continue;
      }
      info.vfio_device_paths.push_back(device_path);
    }
    ++info.chip_count;
    info.generation = std::string(generation);
  }
  return info;
}

TpuHostInfo ScanTpuHost() {
  return ScanTpuHost(kDefaultSysfsPciDevicesRoot, RealFileExists);
}

std::optional<SingleHostTopology> SingleHostTopologyFor(
    absl::string_view generation, int chip_count) {
  // Single-host shapes from the Cloud TPU documentation. Accelerator types
  // for v4/v5p count TensorCores (two per chip), v5e/v6e count chips.
  if (generation == "v6e" || generation == "v5e") {
    const absl::string_view prefix = generation == "v6e" ? "v6e" : "v5litepod";
    switch (chip_count) {
      case 1:
        return SingleHostTopology{absl::StrCat(prefix, "-1"), "1,1,1"};
      case 4:
        return SingleHostTopology{absl::StrCat(prefix, "-4"), "2,2,1"};
      case 8:
        return SingleHostTopology{absl::StrCat(prefix, "-8"), "2,4,1"};
      default:
        return std::nullopt;
    }
  }
  if ((generation == "v5p" || generation == "v4") && chip_count == 4) {
    return SingleHostTopology{absl::StrCat(generation, "-8"), "2,2,1"};
  }
  return std::nullopt;
}

bool MetadataHasTpuEnv(absl::Duration timeout, absl::string_view host_port) {
  const std::pair<absl::string_view, absl::string_view> parts =
      absl::StrSplit(host_port, absl::MaxSplits(':', 1));
  int port = 0;
  sockaddr_in addr{};
  addr.sin_family = AF_INET;
  if (!absl::SimpleAtoi(parts.second, &port) || port <= 0 || port > 65535 ||
      ::inet_pton(AF_INET, std::string(parts.first).c_str(), &addr.sin_addr) !=
          1) {
    LOG(WARNING) << "Invalid metadata server address: " << host_port;
    return false;
  }
  addr.sin_port = htons(static_cast<uint16_t>(port));

  const absl::Time deadline = absl::Now() + timeout;
  const int fd =
      ::socket(AF_INET, SOCK_STREAM | SOCK_NONBLOCK | SOCK_CLOEXEC, 0);
  if (fd < 0) return false;
  absl::Cleanup close_fd = [fd] { ::close(fd); };

  if (::connect(fd, reinterpret_cast<const sockaddr*>(&addr), sizeof(addr)) !=
      0) {
    if (errno != EINPROGRESS) return false;
    if (!PollUntil(fd, POLLOUT, deadline)) return false;
    int error = 0;
    socklen_t len = sizeof(error);
    if (::getsockopt(fd, SOL_SOCKET, SO_ERROR, &error, &len) != 0 ||
        error != 0) {
      return false;
    }
  }

  const std::string request =
      "GET /computeMetadata/v1/instance/attributes/tpu-env HTTP/1.0\r\n"
      "Host: metadata.google.internal\r\n"
      "Metadata-Flavor: Google\r\n"
      "\r\n";
  size_t sent = 0;
  while (sent < request.size()) {
    if (!PollUntil(fd, POLLOUT, deadline)) return false;
    const ssize_t n =
        ::send(fd, request.data() + sent, request.size() - sent, MSG_NOSIGNAL);
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
      return false;
    }
    sent += static_cast<size_t>(n);
  }

  // Only the status line matters: "HTTP/1.x 200 OK".
  std::string response;
  char buf[512];
  while (response.find("\r\n") == std::string::npos && response.size() < 4096) {
    if (!PollUntil(fd, POLLIN, deadline)) return false;
    const ssize_t n = ::recv(fd, buf, sizeof(buf), 0);
    if (n == 0) break;
    if (n < 0) {
      if (errno == EAGAIN || errno == EWOULDBLOCK || errno == EINTR) continue;
      return false;
    }
    response.append(buf, static_cast<size_t>(n));
  }
  const std::vector<absl::string_view> status_line = absl::StrSplit(
      absl::string_view(response).substr(0, response.find("\r\n")), ' ',
      absl::SkipEmpty());
  return status_line.size() >= 2 &&
         absl::StartsWith(status_line[0], "HTTP/1.") && status_line[1] == "200";
}

std::vector<std::pair<std::string, std::string>> ComputeTpuEnvDefaults(
    const TpuHostInfo& host, bool metadata_has_tpu_env, GetEnvFn getenv) {
  std::vector<std::pair<std::string, std::string>> defaults;
  auto set_default = [&](absl::string_view name, absl::string_view value) {
    if (!getenv(name).has_value()) defaults.emplace_back(name, value);
  };

  if (host.chip_count == 0) {
    // Same as JAX: no chips, so there is nothing the metadata server could
    // tell libtpu; make it fail fast instead of retrying for minutes.
    set_default("TPU_SKIP_MDS_QUERY", "1");
    return defaults;
  }

  if (!metadata_has_tpu_env) {
    set_default("TPU_SKIP_MDS_QUERY", "1");
    const bool topology_set_by_user = std::any_of(
        std::begin(kTopologyEnvVars), std::end(kTopologyEnvVars),
        [&](absl::string_view name) { return getenv(name).has_value(); });
    if (topology_set_by_user) {
      LOG(INFO) << "TPU topology environment variables are already set; not "
                   "applying single-host defaults.";
    } else {
      set_default("TPU_HOST_BOUNDS", "1,1,1");
      set_default("TPU_WORKER_ID", "0");
      set_default("TPU_WORKER_HOSTNAMES", "localhost");
      if (std::optional<SingleHostTopology> topology =
              SingleHostTopologyFor(host.generation, host.chip_count);
          topology.has_value()) {
        set_default("TPU_ACCELERATOR_TYPE", topology->accelerator_type);
        set_default("TPU_CHIPS_PER_HOST_BOUNDS",
                    topology->chips_per_host_bounds);
      } else {
        LOG(WARNING) << "Found " << host.chip_count << " TPU "
                     << host.generation
                     << " chip(s) but no GCE 'tpu-env' metadata, and this is "
                        "not a known single-host shape. If the TPU runtime "
                        "fails to start, set TPU_ACCELERATOR_TYPE and "
                        "TPU_CHIPS_PER_HOST_BOUNDS (and TPU_HOST_BOUNDS, "
                        "TPU_WORKER_ID, TPU_WORKER_HOSTNAMES for multi-host "
                        "slices) explicitly.";
      }
    }
  }

  if (!host.vfio_device_paths.empty()) {
    if (getenv("TPU_VISIBLE_CHIPS").has_value()) {
      LOG(INFO) << "TPU_VISIBLE_CHIPS is set; not defaulting "
                   "TPU_VISIBLE_DEVICE_PATHS.";
    } else {
      set_default("TPU_VISIBLE_DEVICE_PATHS",
                  absl::StrJoin(host.vfio_device_paths, ","));
    }
  }
  return defaults;
}

void ApplyTpuEnvDefaults() {
  const TpuHostInfo host = ScanTpuHost();
  LOG(INFO) << "TPU host scan: " << host.chip_count << " chip(s)"
            << (host.generation.empty() ? ""
                                        : absl::StrCat(" of ", host.generation))
            << (host.vfio_device_paths.empty()
                    ? ""
                    : absl::StrCat(" bound to vfio-pci (",
                                   absl::StrJoin(host.vfio_device_paths, ","),
                                   ")"));

  bool metadata_has_tpu_env = false;
  if (host.chip_count > 0) {
    // Respect an explicit decision about the metadata server; otherwise ask.
    bool skip_mds = false;
    if (std::optional<std::string> value = RealGetEnv("TPU_SKIP_MDS_QUERY");
        value.has_value() && absl::SimpleAtob(*value, &skip_mds)) {
      metadata_has_tpu_env = !skip_mds;
    } else {
      metadata_has_tpu_env = MetadataHasTpuEnv();
      LOG(INFO) << "GCE metadata 'tpu-env' attribute: "
                << (metadata_has_tpu_env ? "present" : "absent");
    }
  }

  for (const auto& [name, value] :
       ComputeTpuEnvDefaults(host, metadata_has_tpu_env, RealGetEnv)) {
    ::setenv(name.c_str(), value.c_str(), /*overwrite=*/0);
    LOG(INFO) << "Defaulting " << name << "=" << value;
  }

  if (!host.vfio_device_paths.empty()) {
    const std::optional<std::string> init_args = RealGetEnv("LIBTPU_INIT_ARGS");
    if (!init_args.has_value() ||
        !absl::StrContains(*init_args, "deepsea_hal_test_skip_slicebuilder")) {
      LOG(INFO)
          << "TPU chips are bound to vfio-pci, which usually means a "
             "host without the Cloud TPU host agents. If the runtime "
             "fails to connect to the slice builder, try "
             "LIBTPU_INIT_ARGS=--deepsea_hal_test_skip_slicebuilder=true.";
    }
  }

  LOG(INFO) << "Effective TPU environment: " << DescribeTpuEnv();
}

absl::StatusOr<std::string> ResolveTpuLibraryPath(absl::string_view flag_value,
                                                  GetEnvFn getenv,
                                                  FileExistsFn file_exists) {
  if (!flag_value.empty()) {
    if (file_exists(flag_value)) return std::string(flag_value);
    return absl::NotFoundError(
        absl::StrCat("--tpu_library_path=", flag_value, " does not exist."));
  }
  std::vector<std::string> tried;
  if (std::optional<std::string> from_env = getenv("TPU_LIBRARY_PATH");
      from_env.has_value() && !from_env->empty()) {
    if (file_exists(*from_env)) return *from_env;
    LOG(WARNING) << "TPU_LIBRARY_PATH is set to a non-existent path: "
                 << *from_env << ". Falling back to the default locations.";
    tried.push_back(absl::StrCat("$TPU_LIBRARY_PATH=", *from_env));
  }
  for (absl::string_view path : kDefaultTpuLibraryPaths) {
    if (file_exists(path)) return std::string(path);
    tried.emplace_back(path);
  }
  return absl::NotFoundError(absl::StrCat(
      "Could not find the TPU PJRT plugin (libtpu.so). Tried: ",
      absl::StrJoin(tried, ", "),
      ". Pass --tpu_library_path=<path to libtpu.so> or set TPU_LIBRARY_PATH. "
      "The gcr.io/neptune-serving-engine/tpu-jax-only image ships it at "
      "/usr/lib/libtpu.so; in a Python environment with `pip install "
      "\"jax[tpu]\"` it is at `python3 -c 'import libtpu; "
      "print(libtpu.get_library_path())'`."));
}

absl::StatusOr<std::string> ResolveTpuLibraryPath(
    absl::string_view flag_value) {
  return ResolveTpuLibraryPath(flag_value, RealGetEnv, RealFileExists);
}

}  // namespace serving
}  // namespace xla_serving

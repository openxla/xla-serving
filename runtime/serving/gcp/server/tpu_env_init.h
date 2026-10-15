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

// Host-side TPU environment setup for the dynamically loaded (OSS) TPU PJRT
// plugin, i.e. a C++ port of the parts of JAX's `cloud_tpu_init()` that matter
// for a serving binary, plus single-host topology defaults.
//
// libtpu reads its configuration (topology, worker identity, device paths,
// ...) from `TPU_*` environment variables and, failing that, from the GCE
// metadata server's `tpu-env` attribute. That attribute only exists on VMs
// created through the Cloud TPU API; on GCE-API VMs (`ct*-standard-*`), bare
// metal or stock Linux images libtpu retries the metadata server for minutes
// before failing with an unhelpful error. The functions here derive sensible
// single-host defaults from sysfs so that `docker run <image>` just works on
// such hosts.
//
// All defaults use `setdefault` semantics: a variable that is already set
// (by the user, by the GKE TPU device plugin, ...) is never overridden.
//
// The pure functions take their inputs explicitly so that they can be unit
// tested against a fake sysfs tree and a fake environment;
// `ApplyTpuEnvDefaults` wires them up to the real host.

#ifndef XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_TPU_ENV_INIT_H_
#define XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_TPU_ENV_INIT_H_

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "absl/functional/function_ref.h"  // from @abseil-cpp
#include "absl/status/statusor.h"  // from @abseil-cpp
#include "absl/strings/string_view.h"  // from @abseil-cpp
#include "absl/time/time.h"  // from @abseil-cpp

namespace xla_serving {
namespace serving {

// Looks up an environment variable. Returns `std::nullopt` if it is not set.
// Mirrors `std::getenv` so that tests can inject a fake environment.
using GetEnvFn =
    absl::FunctionRef<std::optional<std::string>(absl::string_view name)>;

// Returns true if `path` exists on the filesystem.
using FileExistsFn = absl::FunctionRef<bool(absl::string_view path)>;

// Default location of the GCE metadata server.
inline constexpr absl::string_view kDefaultGceMetadataHostPort =
    "169.254.169.254:80";

// Default sysfs directory listing PCI devices.
inline constexpr absl::string_view kDefaultSysfsPciDevicesRoot =
    "/sys/bus/pci/devices";

// Where the serving images install the bundled libtpu, in search order.
inline constexpr absl::string_view kDefaultTpuLibraryPaths[] = {
    "/usr/lib/libtpu.so",
    "/lib/libtpu.so",
};

// What the PCI bus says about the TPU chips attached to this host and usable
// by this process.
struct TpuHostInfo {
  // Number of TPU chips (PCI functions `*.0` with Google's vendor id and a
  // known TPU device id). Chips bound to `vfio-pci` are only counted if their
  // `/dev/vfio/<iommu group>` node is present, see `ScanTpuHost`.
  int chip_count = 0;
  // TPU generation as used in Cloud TPU accelerator type names ("v4", "v5p",
  // "v5e", "v6e", "tpu7x", ...). Empty if no chips were found or the device id
  // is unknown.
  std::string generation;
  // `/dev/vfio/<iommu group>` for every counted chip that is bound to the
  // `vfio-pci` driver (stock Linux images). Empty on TPU VM images, where the
  // chips are bound to the Google accelerator driver and show up as
  // `/dev/accel*`.
  std::vector<std::string> vfio_device_paths;
};

// Scans `sysfs_pci_root` (normally `/sys/bus/pci/devices`) for TPU chips.
//
// sysfs always shows every chip on the host, but a container started with
// `--device /dev/vfio/<group>` for some of them (or a GKE pod that was
// allocated a subset of the host's chips) can only use the chips whose device
// node it has. Chips bound to `vfio-pci` are therefore only counted when
// `device_node_exists("/dev/vfio/<group>")` is true, so that the derived
// topology and `TPU_VISIBLE_DEVICE_PATHS` describe what this process can
// actually open. Chips bound to other drivers (`/dev/accel*`) are always
// counted.
//
// Never fails: an unreadable or missing tree yields `chip_count == 0`.
TpuHostInfo ScanTpuHost(absl::string_view sysfs_pci_root,
                        FileExistsFn device_node_exists);

// Same as above against the real sysfs and `/dev`.
TpuHostInfo ScanTpuHost();

// The two libtpu variables that describe a single-host slice.
struct SingleHostTopology {
  std::string accelerator_type;       // e.g. "v6e-4"
  std::string chips_per_host_bounds;  // e.g. "2,2,1"
};

// Returns the single-host topology for `chip_count` chips of `generation`, or
// `std::nullopt` if that combination is not a known single-host Cloud TPU
// shape (multi-host slices are configured through metadata or GKE, never
// guessed here).
std::optional<SingleHostTopology> SingleHostTopologyFor(
    absl::string_view generation, int chip_count);

// Returns true if the GCE metadata server at `host_port` serves the
// `instance/attributes/tpu-env` attribute, i.e. libtpu can configure itself
// from metadata. Returns false on any error or after `timeout`, so that a host
// without a metadata server (where the connect never completes) costs at most
// `timeout` of startup time.
bool MetadataHasTpuEnv(
    absl::Duration timeout = absl::Seconds(2),
    absl::string_view host_port = kDefaultGceMetadataHostPort);

// Computes the `(name, value)` pairs that should be added to the environment
// for `host`, given whether libtpu will be able to read `tpu-env` metadata.
// Only variables for which `getenv(name)` is unset are returned. Pure.
//
//   * No chips: only `TPU_SKIP_MDS_QUERY=1` (what JAX does), so that a server
//     that is started on the wrong host fails fast instead of hanging.
//   * Chips but no metadata: `TPU_SKIP_MDS_QUERY=1`, the single-host worker
//     identity (`TPU_HOST_BOUNDS=1,1,1`, `TPU_WORKER_ID=0`,
//     `TPU_WORKER_HOSTNAMES=localhost`) and, when the shape is known,
//     `TPU_ACCELERATOR_TYPE` and `TPU_CHIPS_PER_HOST_BOUNDS`. If any of the
//     five topology variables is already set the topology is left entirely to
//     the user to avoid mixing sources.
//   * Chips bound to vfio-pci: `TPU_VISIBLE_DEVICE_PATHS` with the
//     `/dev/vfio/*` nodes, unless the user already selected chips through
//     `TPU_VISIBLE_CHIPS` (libtpu's own per-process chip selector).
std::vector<std::pair<std::string, std::string>> ComputeTpuEnvDefaults(
    const TpuHostInfo& host, bool metadata_has_tpu_env, GetEnvFn getenv);

// Probes the real host (sysfs, metadata server), applies
// `ComputeTpuEnvDefaults` to the process environment without overriding
// anything that is already set, and logs the effective `TPU_*` /
// `LIBTPU_INIT_ARGS` environment. Must run before the TPU plugin is loaded.
void ApplyTpuEnvDefaults();

// Resolves the libtpu shared library to load, in order of preference:
// `flag_value` (`--tpu_library_path`), `$TPU_LIBRARY_PATH`, then
// `kDefaultTpuLibraryPaths`. Returns `NotFoundError` with an actionable message
// listing every candidate that was tried if none exists.
absl::StatusOr<std::string> ResolveTpuLibraryPath(absl::string_view flag_value,
                                                  GetEnvFn getenv,
                                                  FileExistsFn file_exists);

// Same as above against the real environment and filesystem.
absl::StatusOr<std::string> ResolveTpuLibraryPath(absl::string_view flag_value);

}  // namespace serving
}  // namespace xla_serving

#endif  // XLA_SERVING_RUNTIME_SERVING_GCP_SERVER_TPU_ENV_INIT_H_

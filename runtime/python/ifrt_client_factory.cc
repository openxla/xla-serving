#include "third_party/neptune/runtime/python/ifrt_client_factory.h"

#include <memory>
#include <string>
#include <utility>

#include "third_party/absl/base/no_destructor.h"
#include "third_party/absl/container/flat_hash_map.h"
#include "third_party/absl/status/status.h"
#include "third_party/absl/status/statusor.h"
#include "third_party/absl/strings/str_cat.h"
#include "third_party/absl/strings/string_view.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_gpu/xla_gpu_client_options.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_gpu/xla_gpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_tpu/xla_tpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace neptune {

namespace {

absl::flat_hash_map<std::string, IfrtClientFactory>* GetRegistry() {
  static absl::NoDestructor<absl::flat_hash_map<std::string, IfrtClientFactory>>
      registry;
  return registry.get();
}

}  // namespace

void RegisterIfrtClientFactory(std::string platform_name,
                               IfrtClientFactory factory) {
  (*GetRegistry())[platform_name] = std::move(factory);
}

absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> CreateIfrtClient(
    absl::string_view platform) {
  absl::flat_hash_map<std::string, IfrtClientFactory>* registry = GetRegistry();
  auto it = registry->find(platform);
  if (it != registry->end()) {
    absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> client =
        it->second(platform);
    if (client.ok()) {
      return client;
    }
    if (!absl::IsNotFound(client.status())) {
      return client;
    }
    // If NotFound, fall back to default creation.
  }

  // Default PjRt creation logic.
  if (platform == "cpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtCpuClient(xla::CpuClientOptions()));
    return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  } else if (platform == "tpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client, xla::GetXlaPjrtTpuClient());
    return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  } else if (platform == "gpu") {
    TF_ASSIGN_OR_RETURN(auto pjrt_client,
                        xla::GetXlaPjrtGpuClient(xla::GpuClientOptions()));
    return xla::ifrt::PjRtClient::Create(std::move(pjrt_client));
  }

  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported platform: ", platform));
}

}  // namespace neptune

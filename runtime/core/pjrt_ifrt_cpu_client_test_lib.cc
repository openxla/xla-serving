#include <memory>
#include <utility>

#include "third_party/absl/status/statusor.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/cpu_client_options.h"
#include "third_party/tensorflow/compiler/xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/client.h"
#include "third_party/tensorflow/compiler/xla/python/ifrt/test_util.h"
#include "third_party/tensorflow/compiler/xla/python/pjrt_ifrt/pjrt_client.h"
#include "third_party/tensorflow/compiler/xla/tsl/platform/statusor.h"

namespace jsv {
namespace {

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
           xla::CpuClientOptions options;
           options.cpu_device_count = 8;
           TF_ASSIGN_OR_RETURN(auto pjrt_client,
                               xla::GetXlaPjrtCpuClient(options));
           return std::shared_ptr<xla::ifrt::Client>(
               xla::ifrt::PjRtClient::Create(std::move(pjrt_client)));
         }),
     true);

}  // namespace
}  // namespace jsv

#include <memory>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "xla/pjrt/plugin/xla_cpu/cpu_client_options.h"  // from @xla
#include "xla/pjrt/plugin/xla_cpu/xla_cpu_pjrt_client.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

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

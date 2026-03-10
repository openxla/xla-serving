#include <memory>
#include <utility>

#include "absl/status/statusor.h"  // from @com_google_absl
#include "xla/pjrt/c_api_client/pjrt_c_api_client.h"  // from @xla
#include "xla/pjrt/pjrt_client.h"  // from @xla
#include "xla/pjrt/plugin/plugin_names.h"  // from @xla
#include "xla/python/ifrt/client.h"  // from @xla
#include "xla/python/ifrt/test_util.h"  // from @xla
#include "xla/python/pjrt_ifrt/pjrt_client.h"  // from @xla
#include "xla/tsl/platform/statusor.h"  // from @xla

namespace jsv {
namespace {

const bool kUnused =
    (xla::ifrt::test_util::RegisterClientFactory(
         []() -> absl::StatusOr<std::shared_ptr<xla::ifrt::Client>> {
           TF_ASSIGN_OR_RETURN(
               std::unique_ptr<xla::PjRtClient> pjrt_client,
               xla::GetCApiClient(kCpuPjrtName,
                                  /*create_options=*/{}));
           xla::ifrt::PjRtClient::CreateOptions options;
           options.pjrt_client = std::move(pjrt_client);
           TF_ASSIGN_OR_RETURN(
               std::unique_ptr<xla::ifrt::PjRtClient> ifrt_client,
               xla::ifrt::PjRtClient::Create(std::move(options)));
           return std::shared_ptr<xla::ifrt::Client>(ifrt_client.release());
         }),
     true);

}  // namespace
}  // namespace jsv
